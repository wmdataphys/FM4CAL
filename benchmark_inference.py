import torch
import numpy as np
import time
import argparse
import logging
import json
from datetime import datetime
from tqdm import tqdm
import torch.profiler as profiler
import warnings
import logging
import os
import torch.nn.functional as F
import bitsandbytes
import torch.nn as nn

os.environ['TORCH_LOGS'] = '-dynamo'

from models.GPT import ECAL_GPT

def quantize_model_bnb(model, logger):
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes not found.")
        return model
    logger.info("Applying bitsandbytes INT8 quantization...")
    quantized_count = 0
    skipped_count = 0
    def replace_linear_with_int8(module, name_prefix=""):
        nonlocal quantized_count, skipped_count
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if isinstance(child, nn.Linear):
                if child.in_features < 64 or child.out_features < 64:
                    skipped_count += 1
                    continue
                has_bias = child.bias is not None
                new_layer = bnb.nn.Linear8bitLt(child.in_features, child.out_features, bias=has_bias, has_fp16_weights=False, threshold=6.0)
                new_layer.weight = bnb.nn.Int8Params(child.weight.data.contiguous(), requires_grad=False, has_fp16_weights=False)
                if has_bias:
                    new_layer.bias = nn.Parameter(child.bias.data.clone())
                setattr(module, name, new_layer)
                quantized_count += 1
            else:
                replace_linear_with_int8(child, full_name)
    replace_linear_with_int8(model)
    logger.info(f"Quantized {quantized_count} Linear layers to INT8, skipped {skipped_count}")
    logger.info("Running calibration forward pass...")
    model.eval()
    with torch.inference_mode():
        dummy_energy = torch.randn(2, 1, device=model.device)
        dummy_material = torch.zeros(2, dtype=torch.long, device=model.device)
        try:
            model.generate(initial_energy=dummy_energy, material_index=dummy_material, max_seq_len=10, temperature=1.0, use_kv_cache=True)
        except Exception as e:
            logger.warning(f"Calibration warning (may be OK): {e}")
    logger.info("INT8 quantization complete")
    return model


def quantize_model_bnb_selective(model, logger, quantize_attention=True, quantize_mlp=True, quantize_heads=True):
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes not found.")
        return model
    logger.info(f"Applying selective INT8 quantization (attn={quantize_attention}, mlp={quantize_mlp}, heads={quantize_heads})...")
    quantized_count = 0
    def should_quantize(name):
        name_lower = name.lower()
        if any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'qkv_proj', 'kv_proj', 'c_proj']):
            return quantize_attention
        if any(x in name_lower for x in ['ff', 'mlp', 'expert']):
            return quantize_mlp
        if any(x in name_lower for x in ['logits_head', 'energy_head']):
            return quantize_heads
        if 'embedding' in name_lower:
            return False
        return True
    def replace_linear(module, name_prefix=""):
        nonlocal quantized_count
        for name, child in list(module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if isinstance(child, nn.Linear):
                if not should_quantize(full_name) or child.in_features < 64 or child.out_features < 64:
                    continue
                has_bias = child.bias is not None
                new_layer = bnb.nn.Linear8bitLt(child.in_features, child.out_features, bias=has_bias, has_fp16_weights=False, threshold=6.0)
                new_layer.weight = bnb.nn.Int8Params(child.weight.data.contiguous(), requires_grad=False, has_fp16_weights=False)
                if has_bias:
                    new_layer.bias = nn.Parameter(child.bias.data.clone())
                setattr(module, name, new_layer)
                quantized_count += 1
            else:
                replace_linear(child, full_name)
    replace_linear(model)
    logger.info(f"Quantized {quantized_count} Linear layers to INT8")
    return model


def make_fused_forward(attn_module):
    def fused_forward(x, attn_mask=None, key_padding_mask=None, need_weights=False, past_kv=None, LoRA_module=None):
        B, T_new, E = x.shape
        H, D = attn_module.num_heads, attn_module.head_dim
        qkv = attn_module.qkv_proj(x).view(B, T_new, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if LoRA_module is not None:
            IA3_K, IA3_V = LoRA_module.get_IA3_KV()
            delta_Q, delta_K, delta_V = LoRA_module(x)
            q = q + delta_Q.view(B, T_new, H, D).transpose(1, 2)
            k = (k + delta_K.view(B, T_new, H, D).transpose(1, 2)) * IA3_K
            v = (v + delta_V.view(B, T_new, H, D).transpose(1, 2)) * IA3_V
        if past_kv is not None:
            cache_k, cache_v = past_kv['k'], past_kv['v']
            curr_len = past_kv['seq_len']
            cache_k[:, :, curr_len:curr_len+T_new] = k
            cache_v[:, :, curr_len:curr_len+T_new] = v
            new_len = curr_len + T_new
            k, v = cache_k[:, :, :new_len], cache_v[:, :, :new_len]
            past_kv['seq_len'] = new_len
            updated_cache, is_decode = past_kv, True
        else:
            updated_cache, is_decode = None, False
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :]
            attn_mask = kpm if attn_mask is None else (attn_mask | kpm)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=attn_module.dropout.p if attn_module.training else 0.0, is_causal=not is_decode)
        return (out.transpose(1, 2).contiguous().view(B, T_new, E), updated_cache)
    return fused_forward


def fuse_qkv_weights(model):
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'q_proj'):
            attn = layer.attn
            if i == 0:
                continue
            w_q, w_k, w_v = attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data
            embed_dim = w_q.size(0)
            fused_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False).to(w_q.device)
            with torch.inference_mode():
                fused_proj.weight.data[:embed_dim] = w_q
                fused_proj.weight.data[embed_dim:2*embed_dim] = w_k
                fused_proj.weight.data[2*embed_dim:] = w_v
            attn.qkv_proj = fused_proj
            attn.forward = make_fused_forward(attn)
            del attn.q_proj, attn.k_proj, attn.v_proj
    return model


def fuse_cross_attention_kv(model):
    class FusedCrossAttnForward:
        def __init__(self, attn_module):
            self.attn = attn_module
        def __call__(self, x, e_embed, attn_mask=None, key_padding_mask=None, need_weights=False, past_kv=None, LoRA_module=None):
            B, T, E = x.shape
            H, D = self.attn.num_heads, self.attn.head_dim
            kv = self.attn.kv_proj(x).view(B, T, 2, H, D)
            k, v = kv.unbind(dim=2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            q = self.attn.q_proj(e_embed).view(B, T, H, D).transpose(1, 2)
            if LoRA_module is not None:
                IA3_K, IA3_V = LoRA_module.get_IA3_KV()
                delta_Q, delta_K, delta_V = LoRA_module(x, e_embed=e_embed)
                q = q + delta_Q.view(B, T, H, D).transpose(1, 2)
                k = (k + delta_K.view(B, T, H, D).transpose(1, 2)) * IA3_K
                v = (v + delta_V.view(B, T, H, D).transpose(1, 2)) * IA3_V
            if past_kv is not None:
                cache_k, cache_v = past_kv['k'], past_kv['v']
                curr_len = past_kv['seq_len']
                cache_k[:, :, curr_len:curr_len+T] = k
                cache_v[:, :, curr_len:curr_len+T] = v
                new_len = curr_len + T
                k, v = cache_k[:, :, :new_len], cache_v[:, :, :new_len]
                past_kv['seq_len'] = new_len
                updated_cache, is_decode = past_kv, True
            else:
                updated_cache, is_decode = None, False
            if key_padding_mask is not None:
                kpm = key_padding_mask[:, None, None, :]
                attn_mask = kpm if attn_mask is None else (attn_mask | kpm)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn.dropout.p if self.attn.training else 0.0, is_causal=not is_decode)
            out = out.transpose(1, 2).reshape(B, T, E)
            return (out, None) if need_weights else (out, updated_cache)
    layer0 = model.layers[0]
    if hasattr(layer0, 'attn') and hasattr(layer0.attn, 'k_proj'):
        attn = layer0.attn
        w_k, w_v = attn.k_proj.weight.data, attn.v_proj.weight.data
        E = w_k.size(0)
        kv_fused = nn.Linear(E, 2*E, bias=False, device=w_k.device)
        with torch.inference_mode():
            kv_fused.weight.data = torch.cat([w_k, w_v], dim=0)
        attn.kv_proj = kv_fused
        attn.forward = FusedCrossAttnForward(attn)
        del attn.k_proj, attn.v_proj
    return model


def log_sdpa_backends(logger):
    if not torch.cuda.is_available():
        logger.info("CUDA not available")
        return
    logger.info("===== SDPA Backend Status =====")
    logger.info(f"Flash SDP enabled       : {torch.backends.cuda.flash_sdp_enabled()}")
    logger.info(f"Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    logger.info(f"Math SDP enabled        : {torch.backends.cuda.math_sdp_enabled()}")
    logger.info(f"PyTorch version         : {torch.__version__}")
    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU                     : {props.name}")
    logger.info(f"Compute capability      : {props.major}.{props.minor}")
    logger.info("================================")


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_peak_memory_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def setup_logger():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"inference_benchmark_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logger = logging.getLogger("benchmark")
    logger.info(f"Logging to file: {log_file}")
    return logger


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_model(model, initial_energy, material_index, use_kv_cache, logger=None):
    logger.info(f"\n===== Profiling (KV Cache = {use_kv_cache}) =====")
    model.eval()
    B_profile = min(16, initial_energy.shape[0])
    ie_small = initial_energy[:B_profile].clone()
    mi_small = material_index[:B_profile].clone()
    logger.info("Warming up...")
    with torch.inference_mode():
        model.generate(initial_energy=ie_small, material_index=mi_small, max_seq_len=50, temperature=1.0, use_kv_cache=use_kv_cache)
    torch.cuda.synchronize()
    logger.info("Profiling...")
    with torch.inference_mode():
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False, with_flops=True) as prof:
            model.generate(initial_energy=ie_small, material_index=mi_small, max_seq_len=2100, temperature=1.0, use_kv_cache=use_kv_cache)
            torch.cuda.synchronize()
    logger.info("\n----- Top 25 CUDA Kernels by GPU Time -----")
    logger.info(f"\n{prof.key_averages().table(sort_by='cuda_time_total', row_limit=25)}")
    logger.info("\n----- Top 25 by Self CUDA Time -----")
    logger.info(f"\n{prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=25)}")
    trace_file = f"profile_trace_kv{use_kv_cache}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    prof.export_chrome_trace(trace_file)
    logger.info(f"\nTrace saved to: {trace_file} (view with chrome://tracing)")
    return prof


def benchmark(model, initial_energy, material_index, *, use_kv_cache, use_cuda_graph=False, iters=50, warmup=10, max_seq_len=None, logger=None):
    if max_seq_len is None:
        max_seq_len = 2700
    model.eval()
    if use_cuda_graph:
        generate_fn = lambda: model.generate_with_cuda_graph(initial_energy=initial_energy, material_index=material_index, max_seq_len=max_seq_len, temperature=1.0)
    else:
        generate_fn = lambda: model.generate(initial_energy=initial_energy, material_index=material_index, max_seq_len=max_seq_len, temperature=1.0, use_kv_cache=use_kv_cache)
    with torch.inference_mode():
        for _ in tqdm(range(warmup), desc=f"Warmup (KV={use_kv_cache}, CUDAGraph={use_cuda_graph})", leave=False):
            generate_fn()
    sync()
    reset_gpu_memory()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in tqdm(range(iters), desc=f"Benchmark (KV={use_kv_cache}, CUDAGraph={use_cuda_graph})", leave=False):
            generate_fn()
    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / iters, get_peak_memory_mb()


def benchmark_all_modes(model, initial_energy, material_index, logger, iters=50, max_seq_len=None):
    logger.info("\n===== Benchmarking All Generation Modes =====")
    results = {}
    for label, kv, cg in [("no_cache", False, False), ("kv_cache", True, False), ("cuda_graph", True, True)]:
        logger.info(f"\n--- Mode: {label} ---")
        try:
            t, mem = benchmark(model, initial_energy, material_index, use_kv_cache=kv, use_cuda_graph=cg, iters=iters, max_seq_len=max_seq_len, logger=logger)
            results[label] = {'time': t, 'memory': mem}
            logger.info(f"  Time: {t*1000:.2f}ms | Memory: {mem:.1f}MB")
        except Exception as e:
            logger.error(f"  {label} failed: {e}")
            results[label] = None
    logger.info("\n===== Summary =====")
    baseline = results['no_cache']['time']
    for mode, data in results.items():
        if data is not None:
            logger.info(f"{mode:15s}: {data['time']*1000:8.2f}ms | {baseline/data['time']:5.2f}x speedup | {data['memory']:.1f}MB")
    return results


def benchmark_components(model, initial_energy, material_index, logger, max_seq_len=500):
    logger.info("\n===== Component-Level Timing Analysis =====")
    model.eval()
    device = initial_energy.device
    B = initial_energy.shape[0]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    times = {k: 0.0 for k in ['embedding_token', 'embedding_pos', 'embedding_energy', 'embedding_energy_pos', 'forward_pass', 'logits_pixel', 'logits_energy', 'softmax_pixel', 'softmax_energy', 'multinomial_pixel', 'multinomial_energy', 'eos_handling']}
    if initial_energy.dim() == 1:
        initial_energy = initial_energy.unsqueeze(1)
    init_e_embed = model.initial_energy_embedding(initial_energy).unsqueeze(1).to(dtype)
    is_done = torch.zeros(B, dtype=torch.bool, device=device)
    idx_buffer = torch.zeros((B, max_seq_len + 1), device=device, dtype=torch.long)
    e_buffer = torch.zeros((B, max_seq_len + 1), device=device, dtype=torch.long)
    idx_buffer[:, 0] = model.SOS_token
    e_buffer[:, 0] = 0
    kv_caches = model._allocate_kv_cache(B, max_seq_len + 2, device, dtype)
    lora_modules = model._get_lora_modules_for_particle("gamma")
    logger.info(f"Running instrumented generation for {max_seq_len} steps...")
    with torch.inference_mode():
        with torch.amp.autocast('cuda', dtype=dtype):
            for step in range(max_seq_len):
                def timed(key, fn):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    result = fn()
                    torch.cuda.synchronize()
                    times[key] += time.perf_counter() - t0
                    return result
                cur = slice(step, step+1)
                tok_emb = timed('embedding_token', lambda: model.token_embedding(idx_buffer[:, cur]))
                pos_idx = torch.full((B, 1), step, device=device, dtype=torch.long)
                pos_emb = timed('embedding_pos', lambda: model.pos_embedding(pos_idx))
                e_emb = timed('embedding_energy', lambda: model.energy_embedding(e_buffer[:, cur]))
                e_pos_emb = timed('embedding_energy_pos', lambda: model.energy_pos_embedding(pos_idx))
                if step == 0:
                    x_t = torch.cat([init_e_embed, tok_emb + pos_emb], dim=1)
                    e_t = torch.cat([init_e_embed, e_emb + e_pos_emb], dim=1)
                else:
                    x_t, e_t = tok_emb + pos_emb, e_emb + e_pos_emb
                h_t, kv_caches = timed('forward_pass', lambda: model.forward_decode_step(x_t, e_t, material_index, kv_caches=kv_caches, is_first_step=(step==0), lora_modules=lora_modules))
                pixel_logits = timed('logits_pixel', lambda: model.logits_head(h_t[:, -1, :]))
                energy_logits = timed('logits_energy', lambda: model.energy_head(h_t[:, -1, :]))
                pixel_probs = timed('softmax_pixel', lambda: torch.softmax(pixel_logits, dim=-1, dtype=torch.float32))
                energy_probs = timed('softmax_energy', lambda: torch.softmax(energy_logits, dim=-1, dtype=torch.float32))
                idx_next = timed('multinomial_pixel', lambda: torch.multinomial(pixel_probs, num_samples=1))
                e_next = timed('multinomial_energy', lambda: torch.multinomial(energy_probs, num_samples=1))
                idx_next = idx_next.clamp(0, model.space_vocab - 1)
                e_next = e_next.clamp(0, model.energy_vocab - 1)
                def eos_step():
                    nonlocal is_done
                    newly_done = (e_next.squeeze(1) == model.EOS_energy_token) | (idx_next.squeeze(1) == model.EOS_token)
                    pad_mask, eos_mask = is_done & ~newly_done, newly_done
                    is_done = is_done | newly_done
                    idx_store = idx_next.squeeze(1).clone()
                    idx_store.masked_fill_(eos_mask, model.EOS_token)
                    idx_store.masked_fill_(pad_mask, model.pad_token)
                    idx_buffer[:, step + 1] = idx_store
                    e_store = e_next.squeeze(1).clone()
                    e_store.masked_fill_(eos_mask, model.EOS_energy_token)
                    e_store.masked_fill_(pad_mask, model.energy_pad_token)
                    e_buffer[:, step + 1] = e_store
                timed('eos_handling', eos_step)
                if (step + 1) % 100 == 0 and torch.all(is_done):
                    logger.info(f"Early exit at step {step + 1}")
                    break
    total_time = sum(times.values())
    groups = {
        'Embeddings': ['embedding_token', 'embedding_pos', 'embedding_energy', 'embedding_energy_pos'],
        'Forward Pass': ['forward_pass'],
        'Logits Heads': ['logits_pixel', 'logits_energy'],
        'Softmax': ['softmax_pixel', 'softmax_energy'],
        'Sampling': ['multinomial_pixel', 'multinomial_energy'],
        'EOS Handling': ['eos_handling'],
    }
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Component':<30} {'Time (ms)':<12} {'Per Step (us)':<15} {'%':<8}")
    logger.info(f"{'='*60}")
    for component, t in sorted(times.items(), key=lambda x: -x[1]):
        logger.info(f"{component:<30} {t*1000:<12.2f} {(t/max_seq_len)*1e6:<15.1f} {100*t/total_time:<8.1f}")
    logger.info(f"{'-'*60}")
    logger.info(f"{'TOTAL':<30} {total_time*1000:<12.2f}")
    logger.info(f"\n{'='*60}")
    group_times = {g: sum(times[c] for c in comps) for g, comps in groups.items()}
    for g, gt in sorted(group_times.items(), key=lambda x: -x[1]):
        logger.info(f"{g:<30} {gt*1000:<12.2f} {100*gt/total_time:<8.1f}")
    logger.info(f"{'='*60}")
    logger.info("\n===== Optimization Recommendations =====")
    if group_times['Sampling'] / total_time > 0.15:
        logger.info("-> Sampling >15%: consider Gumbel-max or top-k sampling.")
    if group_times['Softmax'] / total_time > 0.10:
        logger.info("-> Softmax >10%: consider fused softmax+sampling.")
    if group_times['Embeddings'] / total_time > 0.15:
        logger.info("-> Embeddings >15%: consider fusing token+position embeddings.")
    if group_times['Logits Heads'] / total_time > 0.20:
        logger.info("-> Logits heads >20%: consider quantization or smaller vocab projection.")
    if group_times['Forward Pass'] / total_time < 0.30:
        logger.info("-> Forward pass <30%: model is overhead-bound. CUDA graphs should help.")
    return times, group_times


def main(args):
    logger = setup_logger()
    warnings.filterwarnings('ignore', message='.*skipping cudagraphs.*')
    logging.getLogger('torch._inductor.utils').setLevel(logging.ERROR)
    logging.getLogger('torch._dynamo').setLevel(logging.ERROR)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    energies = np.load(args.energy_file).astype(np.float32)
    energies = np.random.choice(energies, size=args.batch_size, replace=True)
    initial_energy = torch.tensor(energies, device=device).contiguous()
    material_index = torch.full((args.batch_size,), 1, dtype=torch.long, device=device).contiguous()

    # Build model from config (mirrors generate.py exactly)
    material_list = config['material_list']
    num_experts = len(material_list)
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else config['model']['max_seq_length']

    model = ECAL_GPT(
        config['model']['vocab_size'],
        max_seq_len,
        config['model']['embed_dim'],
        attn_heads=config['model']['attn_heads'],
        num_blocks=config['model']['num_blocks'],
        hidden_units=config['model']['hidden_units'],
        digitize_energy=bool(config['digitize_energy']),
        mlp_scale=config['model']['mlp_scale'],
        energy_vocab=config['model']['energy_vocab'],
        drop_rates=config['model']['drop_rates'],
        use_MoE=bool(config['model']['use_MoE']),
        num_experts=num_experts,
        material_list=material_list,
        base_model_type=config['base_model_type'],
        particle_list=config['particle_list'],
        LoRA_r=config['model']['LoRA_r'],
        LoRA_alpha=config['model']['LoRA_alpha'],
        enable_head_LoRA=config['model']['enable_head_LoRA'],
        enable_vocab_LoRA=config['model']['enable_vocab_LoRA'],
        enable_embedding_adapter=config['model']['enable_embedding_adapter'],
        vocab_LoRA_scale=config['model']['vocab_LoRA_scale'],
        learnable_vocabs=config['model']['learnable_vocabs'],
        use_RoPE=config['model']['use_RoPE'] and not args.no_rope,
        is_expanded=config['model']['is_expanded'],
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["net_state_dict"], strict=True)
    logger.info("Model loaded successfully.")

    model = model.to(device)
    model.device = device

    if args.fuse_qkv:
        logger.info("Fusing QKV projections...")
        model = fuse_qkv_weights(model)
        model = fuse_cross_attention_kv(model)
        logger.info("QKV fusion complete")

    if args.quantize:
        model = quantize_model_bnb(model, logger)
        model.set_skip_compile(True)
    elif args.quantize_selective:
        model = quantize_model_bnb_selective(model, logger, quantize_attention=not args.no_quantize_attention, quantize_mlp=not args.no_quantize_mlp, quantize_heads=not args.no_quantize_heads)
        model.set_skip_compile(True)

    if args.compile and device == "cuda" and not args.cuda_graph:
        logger.info("Compiling model with torch.compile(...)")
        model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    if device == "cpu":
        logger.info("Running on CPU - disabling torch.compile")
        model.set_skip_compile(True)

    log_sdpa_backends(logger)
    logger.info("===== Inference Benchmark =====")
    logger.info(f"Device         : {device}")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info(f"Iterations     : {args.iters}")
    logger.info(f"Max seq len    : {max_seq_len}")
    logger.info("================================")

    if args.instrument:
        benchmark_components(model, initial_energy, material_index, logger, max_seq_len=args.instrument_steps)
        return

    if args.profile:
        profile_model(model, initial_energy, material_index, use_kv_cache=True, logger=logger)
        return

    if args.benchmark_all:
        benchmark_all_modes(model, initial_energy, material_index, logger, iters=args.iters, max_seq_len=max_seq_len)
        return

    if args.cuda_graph:
        t, peak_mem_mb = benchmark(model, initial_energy, material_index, use_kv_cache=True, use_cuda_graph=True, iters=args.iters, max_seq_len=max_seq_len, logger=logger)
        label = "CUDA Graph    "
    else:
        if device == "cpu":
            torch.set_autocast_enabled(False)
        t, peak_mem_mb = benchmark(model, initial_energy, material_index, use_kv_cache=True, use_cuda_graph=False, iters=args.iters, max_seq_len=max_seq_len, logger=logger)
        label = "KV Cache = True"

    samples_per_sec = args.batch_size / t
    ms_per_sample = (t / args.batch_size) * 1000
    logger.info(f"{label} | Batch latency = {t*1000:.2f} ms | Per-sample = {ms_per_sample:.2f} ms | Throughput = {samples_per_sec:.2f} samples/s | Peak GPU Memory = {peak_mem_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file (e.g. config/config.json)")
    parser.add_argument("--energy_file", default="initial_energies.npy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=70)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max_seq_len", type=int, default=None, help="Max sequence length. Defaults to max_seq_length in config.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fuse_qkv", action="store_true", help="Fuse Q, K, V projections for faster inference")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization using bitsandbytes")
    parser.add_argument("--quantize_selective", action="store_true", help="Apply selective INT8 quantization")
    parser.add_argument("--no_quantize_attention", action="store_true")
    parser.add_argument("--no_quantize_mlp", action="store_true")
    parser.add_argument("--no_quantize_heads", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Run profiler to identify bottlenecks")
    parser.add_argument("--cuda_graph", action="store_true", help="Use CUDA graphs for maximum inference speed")
    parser.add_argument("--benchmark_all", action="store_true", help="Benchmark all modes: no cache, KV cache, CUDA graph")
    parser.add_argument("--instrument", action="store_true", help="Run component-level timing analysis")
    parser.add_argument("--instrument_steps", type=int, default=500)
    parser.add_argument("--no_rope", action="store_true", help="Disable RoPE (for profiling impact)")
    args = parser.parse_args()
    main(args)
