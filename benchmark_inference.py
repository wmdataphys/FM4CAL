import torch
import numpy as np
import time
import argparse
import logging
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
# import quanto

def quantize_model_bnb(model, logger):
    """
    Apply bitsandbytes INT8 quantization to Linear layers.
    Works well on RTX 4090 and other Ampere/Ada GPUs.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes not found. Install with: pip install bitsandbytes")
        return model
    
    logger.info("Applying bitsandbytes INT8 quantization...")
    
    quantized_count = 0
    skipped_count = 0
    
    def replace_linear_with_int8(module, name_prefix=""):
        nonlocal quantized_count, skipped_count
        
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            if isinstance(child, nn.Linear):
                # Skip small layers (not worth quantizing)
                if child.in_features < 64 or child.out_features < 64:
                    skipped_count += 1
                    continue
                
                # Create INT8 linear layer
                has_bias = child.bias is not None
                
                # bitsandbytes Linear8bitLt for inference
                new_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    has_fp16_weights=False,  # Use INT8 weights
                    threshold=6.0,  # Outlier threshold
                )
                
                # Copy weights (will be quantized on first forward pass)
                new_layer.weight = bnb.nn.Int8Params(
                    child.weight.data.contiguous(),
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                
                if has_bias:
                    new_layer.bias = nn.Parameter(child.bias.data.clone())
                
                # Replace the layer
                setattr(module, name, new_layer)
                quantized_count += 1
                
            else:
                # Recurse into child modules
                replace_linear_with_int8(child, full_name)
    
    replace_linear_with_int8(model)
    
    logger.info(f"Quantized {quantized_count} Linear layers to INT8")
    logger.info(f"Skipped {skipped_count} small layers")
    
    # Run a dummy forward pass to trigger quantization
    logger.info("Running calibration forward pass...")
    model.eval()
    with torch.inference_mode():
        dummy_energy = torch.randn(2, 1, device=model.device)
        dummy_material = torch.zeros(2, dtype=torch.long, device=model.device)
        try:
            model.generate(
                initial_energy=dummy_energy,
                material_index=dummy_material,
                max_seq_len=10,
                temperature=1.0,
                use_kv_cache=True,
            )
        except Exception as e:
            logger.warning(f"Calibration warning (may be OK): {e}")
    
    logger.info("INT8 quantization complete")
    return model


def quantize_model_bnb_selective(model, logger, quantize_attention=True, quantize_mlp=True, quantize_heads=True):
    """
    Selective INT8 quantization - choose which parts to quantize.
    More control over quality vs speed tradeoff.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("bitsandbytes not found. Install with: pip install bitsandbytes")
        return model
    
    logger.info("Applying selective bitsandbytes INT8 quantization...")
    logger.info(f"  Quantize attention: {quantize_attention}")
    logger.info(f"  Quantize MLP: {quantize_mlp}")
    logger.info(f"  Quantize heads: {quantize_heads}")
    
    quantized_count = 0
    
    def should_quantize(name):
        """Determine if a layer should be quantized based on its name."""
        name_lower = name.lower()
        
        # Attention projections
        if any(x in name_lower for x in ['q_proj', 'k_proj', 'v_proj', 'qkv_proj', 'kv_proj', 'c_proj']):
            return quantize_attention
        
        # MLP/FFN layers
        if any(x in name_lower for x in ['ff', 'mlp', 'expert']):
            return quantize_mlp
        
        # Output heads
        if any(x in name_lower for x in ['logits_head', 'energy_head']):
            return quantize_heads
        
        # Embeddings - generally don't quantize
        if 'embedding' in name_lower:
            return False
        
        # Default: quantize
        return True
    
    def replace_linear(module, name_prefix=""):
        nonlocal quantized_count
        
        for name, child in list(module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            if isinstance(child, nn.Linear):
                if not should_quantize(full_name):
                    logger.debug(f"  Skipping: {full_name}")
                    continue
                
                if child.in_features < 64 or child.out_features < 64:
                    continue
                
                has_bias = child.bias is not None
                
                new_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                
                new_layer.weight = bnb.nn.Int8Params(
                    child.weight.data.contiguous(),
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                
                if has_bias:
                    new_layer.bias = nn.Parameter(child.bias.data.clone())
                
                setattr(module, name, new_layer)
                quantized_count += 1
                logger.debug(f"  Quantized: {full_name}")
            else:
                replace_linear(child, full_name)
    
    replace_linear(model)
    logger.info(f"Quantized {quantized_count} Linear layers to INT8")
    
    return model

def make_fused_forward(attn_module):
    """Factory to avoid closure issues"""
    def fused_forward(x, attn_mask=None, key_padding_mask=None,
                    need_weights=False, past_kv=None, LoRA_module=None):  # Added LoRA_module
        B, T_new, E = x.shape
        H, D = attn_module.num_heads, attn_module.head_dim

        # Fused QKV: [B,T,3E] -> [3,B,H,T,D]
        qkv = attn_module.qkv_proj(x).view(B, T_new, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,T,D]
        
        # Apply LoRA if available
        if LoRA_module is not None:
            IA3_K, IA3_V = LoRA_module.get_IA3_KV()
            delta_Q, delta_K, delta_V = LoRA_module(x)
            dq = delta_Q.view(B, T_new, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
            dk = delta_K.view(B, T_new, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
            dv = delta_V.view(B, T_new, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
            q = q + dq
            k = (k + dk) * IA3_K
            v = (v + dv) * IA3_V
        
        # Normalize Q and K
        # if attn_module.qk_norm:
        #     k = torch.nn.functional.normalize(k, p=2, dim=-1)
        #     q = torch.nn.functional.normalize(q, p=2, dim=-1)
        
        # Handle KV cache
        if past_kv is not None:
            cache_k = past_kv['k']
            cache_v = past_kv['v']
            curr_len = past_kv['seq_len']
            
            cache_k[:, :, curr_len:curr_len+T_new] = k
            cache_v[:, :, curr_len:curr_len+T_new] = v
            
            new_len = curr_len + T_new
            k = cache_k[:, :, :new_len]
            v = cache_v[:, :, :new_len]
            
            past_kv['seq_len'] = new_len
            updated_cache = past_kv
            is_decode = True
        else:
            updated_cache = None
            is_decode = False
        
        # Compute attention
        # if attn_module.qk_norm:
        #     attn_scores = attn_module.g_scale * q @ k.transpose(2, 3)
        # else:
        #     attn_scores = attn_module.d_k * q @ k.transpose(2, 3)
        
        # if attn_mask is not None:
        #     attn_scores.masked_fill_(attn_mask, -torch.inf)
        
        if key_padding_mask is not None:
            # key_padding_mask: [B, T_k] with True=mask
            kpm = key_padding_mask[:, None, None, :]  # [B,1,1,T_k]
            attn_mask = kpm if attn_mask is None else (attn_mask | kpm)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=attn_module.dropout.p if attn_module.training else 0.0,
            is_causal=not is_decode,  # decode usually False; you can also keep it False if cache truncation handles causality
        )  # [B,H,T_new,D]

        attn_output = out.transpose(1, 2).contiguous().view(B, T_new, E)
        
        # if need_weights:
        #     return attn_output, attn_scores
        # else:
        return (attn_output, updated_cache)
    
    return fused_forward


def fuse_qkv_weights(model):
    """
    Fuse separate Q, K, V projections into single QKV projection.
    Only for MHSA (self-attention) layers, not CrossAttention.
    """
    import torch.nn as nn
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'q_proj'):
            attn = layer.attn
            
            # Skip CrossAttention (layer 0), only fuse MHSA
            if i == 0:
                continue
            
            # Get existing weights
            w_q = attn.q_proj.weight.data
            w_k = attn.k_proj.weight.data
            w_v = attn.v_proj.weight.data
            
            embed_dim = w_q.size(0)
            
            # Create fused projection
            fused_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False).to(w_q.device)
            
            # Stack weights: [3*E, E]
            with torch.inference_mode():
                fused_proj.weight.data[:embed_dim] = w_q
                fused_proj.weight.data[embed_dim:2*embed_dim] = w_k
                fused_proj.weight.data[2*embed_dim:] = w_v
            
            # Replace the projection
            attn.qkv_proj = fused_proj
            
            # Replace forward method using factory
            attn.forward = make_fused_forward(attn)
            
            # Delete old projections to save memory
            del attn.q_proj
            del attn.k_proj
            del attn.v_proj
    
    return model


def fuse_cross_attention_kv(model):
    """Fuse K,V projections in CrossAttention (layer 0)"""
    import torch.nn as nn
    import torch.nn.functional as F
    
    class FusedCrossAttnForward:
        def __init__(self, attn_module):
            self.attn = attn_module
            
        def __call__(self, x, e_embed, attn_mask=None, key_padding_mask=None,
                    need_weights=False, past_kv=None, LoRA_module=None):
            B, T, E = x.shape
            H = self.attn.num_heads
            D = self.attn.head_dim
            
            # Fused K,V from x
            kv = self.attn.kv_proj(x).view(B, T, 2, H, D)
            k, v = kv.unbind(dim=2)  # [B, T, H, D] each
            k = k.transpose(1, 2)    # [B, H, T, D]
            v = v.transpose(1, 2)    # [B, H, T, D]
            
            # Separate Q from e_embed
            q = self.attn.q_proj(e_embed).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
            
            # Apply LoRA if available
            if LoRA_module is not None:
                IA3_K, IA3_V = LoRA_module.get_IA3_KV()
                delta_Q, delta_K, delta_V = LoRA_module(x, e_embed=e_embed)
                q = q + delta_Q.view(B, T, H, D).transpose(1, 2)
                k = (k + delta_K.view(B, T, H, D).transpose(1, 2)) * IA3_K
                v = (v + delta_V.view(B, T, H, D).transpose(1, 2)) * IA3_V
            
            # KV cache handling
            if past_kv is not None:
                cache_k, cache_v = past_kv['k'], past_kv['v']
                curr_len = past_kv['seq_len']
                
                cache_k[:, :, curr_len:curr_len+T] = k
                cache_v[:, :, curr_len:curr_len+T] = v
                
                new_len = curr_len + T
                k = cache_k[:, :, :new_len]
                v = cache_v[:, :, :new_len]
                past_kv['seq_len'] = new_len
                updated_cache = past_kv
                is_decode = True
            else:
                updated_cache = None
                is_decode = False
            
            # Handle key_padding_mask
            if key_padding_mask is not None:
                # key_padding_mask: [B, T_k] with True=mask
                kpm = key_padding_mask[:, None, None, :]  # [B, 1, 1, T_k]
                attn_mask = kpm if attn_mask is None else (attn_mask | kpm)
            
            # Use SDPA instead of manual attention
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn.dropout.p if self.attn.training else 0.0,
                is_causal=not is_decode,
            )  # [B, H, T, D]
            
            out = out.transpose(1, 2).reshape(B, T, E)  # [B, T, E]
            
            return (out, None) if need_weights else (out, updated_cache)
    
    layer0 = model.layers[0]
    if hasattr(layer0, 'attn') and hasattr(layer0.attn, 'k_proj'):
        attn = layer0.attn
        
        # Fuse K,V only (Q stays separate)
        w_k = attn.k_proj.weight.data
        w_v = attn.v_proj.weight.data
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
        logger.info("CUDA not available — SDPA disabled")
        return

    logger.info("===== SDPA Backend Status =====")
    logger.info(f"Flash SDP enabled       : {torch.backends.cuda.flash_sdp_enabled()}")
    logger.info(f"Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    logger.info(f"Math SDP enabled        : {torch.backends.cuda.math_sdp_enabled()}")

    # Torch version matters
    logger.info(f"PyTorch version         : {torch.__version__}")

    # GPU info
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

# -------------------------
# Logging setup
# -------------------------
def setup_logger():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"inference_benchmark_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger("benchmark")
    logger.info(f"Logging to file: {log_file}")
    return logger


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_model(
    model,
    initial_energy,
    material_index,
    use_kv_cache: bool,
    logger=None,
):
    """
    Profile a single generation run to identify bottlenecks.
    """
    logger.info(f"\n===== Profiling (KV Cache = {use_kv_cache}) =====")
    
    model.eval()
    
    # Use smaller batch and sequence for profiling to avoid OOM
    B_profile = min(16, initial_energy.shape[0])
    initial_energy_small = initial_energy[:B_profile].clone()
    material_index_small = material_index[:B_profile].clone()
    
    # Warmup first
    logger.info("Warming up...")
    with torch.inference_mode():
        model.generate(
            initial_energy=initial_energy_small,
            material_index=material_index_small,
            max_seq_len=50,
            temperature=1.0,
            use_kv_cache=use_kv_cache,
        )
    torch.cuda.synchronize()
    
    logger.info("Profiling...")
    with torch.inference_mode():
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
            with_flops=True,
        ) as prof:
            model.generate(
                initial_energy=initial_energy_small,
                material_index=material_index_small,
                max_seq_len=2100,  # Short sequence for profiling
                temperature=1.0,
                use_kv_cache=use_kv_cache,
            )
            torch.cuda.synchronize()
    
    # Print top operations by CUDA time (kernel time, not API calls)
    logger.info("\n----- Top 25 CUDA Kernels by GPU Time -----")
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=25,
    )
    logger.info(f"\n{table}")
    
    # Also show self CUDA time (excludes child calls)
    logger.info("\n----- Top 25 by Self CUDA Time -----")
    table2 = prof.key_averages().table(
        sort_by="self_cuda_time_total", 
        row_limit=25,
    )
    logger.info(f"\n{table2}")
    
    # Save trace
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = f"profile_trace_kv{use_kv_cache}_{timestamp}.json"
    prof.export_chrome_trace(trace_file)
    logger.info(f"\nDetailed trace saved to: {trace_file}")
    logger.info("View with chrome://tracing in Chrome/Edge browser")
    
    return prof


def benchmark(
    model,
    initial_energy,
    material_index,
    *,
    use_kv_cache: bool,
    use_cuda_graph: bool = False,  # New parameter
    iters: int = 50,
    warmup: int = 10,
    max_seq_len: int = None,
    logger=None,
):
    """
    Benchmarks ECAL_GPT.generate() only.
    AMP is handled internally by the model.
    """

    if max_seq_len is None:
        max_seq_len = model.pos_embedding.num_embeddings

    model.eval()

    # Select generation method
    if use_cuda_graph:
        generate_fn = lambda: model.generate_with_cuda_graph(
            initial_energy=initial_energy,
            material_index=material_index,
            max_seq_len=max_seq_len,
            temperature=1.0,
        )
    else:
        generate_fn = lambda: model.generate(
            initial_energy=initial_energy,
            material_index=material_index,
            max_seq_len=max_seq_len,
            temperature=1.0,
            use_kv_cache=use_kv_cache,
        )

    # Warmup
    with torch.inference_mode():
        for _ in tqdm(
            range(warmup),
            desc=f"Warmup (KV={use_kv_cache}, CUDAGraph={use_cuda_graph})",
            leave=False,
        ):
            generate_fn()

    sync()

    # Timed runs
    reset_gpu_memory()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in tqdm(
            range(iters),
            desc=f"Benchmark (KV={use_kv_cache}, CUDAGraph={use_cuda_graph})",
            leave=False,
        ):
            generate_fn()

    sync()
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / iters
    peak_mem_mb = get_peak_memory_mb()
    return avg_time, peak_mem_mb


def benchmark_all_modes(
    model,
    initial_energy,
    material_index,
    logger,
    iters: int = 50,
    max_seq_len: int = None,
):
    """
    Benchmark all three modes: no cache, KV cache, and CUDA graph.
    """
    logger.info("\n===== Benchmarking All Generation Modes =====")
    
    results = {}
    
    # Mode 1: No KV cache
    logger.info("\n--- Mode: No KV Cache ---")
    t, mem = benchmark(
        model, initial_energy, material_index,
        use_kv_cache=False,
        use_cuda_graph=False,
        iters=iters,
        max_seq_len=max_seq_len,
        logger=logger,
    )
    results['no_cache'] = {'time': t, 'memory': mem}
    logger.info(f"  Time: {t*1000:.2f}ms | Memory: {mem:.1f}MB")
    
    # Mode 2: KV cache (no CUDA graph)
    logger.info("\n--- Mode: KV Cache (no CUDA graph) ---")
    t, mem = benchmark(
        model, initial_energy, material_index,
        use_kv_cache=True,
        use_cuda_graph=False,
        iters=iters,
        max_seq_len=max_seq_len,
        logger=logger,
    )
    results['kv_cache'] = {'time': t, 'memory': mem}
    logger.info(f"  Time: {t*1000:.2f}ms | Memory: {mem:.1f}MB")
    
    # Mode 3: CUDA graph (includes KV cache)
    logger.info("\n--- Mode: CUDA Graph + KV Cache ---")
    try:
        t, mem = benchmark(
            model, initial_energy, material_index,
            use_kv_cache=True,
            use_cuda_graph=True,
            iters=iters,
            max_seq_len=max_seq_len,
            logger=logger,
        )
        results['cuda_graph'] = {'time': t, 'memory': mem}
        logger.info(f"  Time: {t*1000:.2f}ms | Memory: {mem:.1f}MB")
    except Exception as e:
        logger.error(f"  CUDA graph failed: {e}")
        results['cuda_graph'] = None
    
    # Summary
    logger.info("\n===== Summary =====")
    baseline = results['no_cache']['time']
    
    for mode, data in results.items():
        if data is not None:
            speedup = baseline / data['time']
            logger.info(f"{mode:15s}: {data['time']*1000:8.2f}ms | {speedup:5.2f}x speedup | {data['memory']:.1f}MB")
    
    return results


def benchmark_components(model, initial_energy, material_index, logger, max_seq_len=500):
    """
    Instrumented generation to measure time breakdown by component.
    Uses fewer steps for faster profiling.
    """
    logger.info("\n===== Component-Level Timing Analysis =====")
    
    model.eval()
    device = initial_energy.device
    B = initial_energy.shape[0]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Timing accumulators
    times = {
        'embedding_token': 0.0,
        'embedding_pos': 0.0,
        'embedding_energy': 0.0,
        'embedding_energy_pos': 0.0,
        'forward_pass': 0.0,
        'logits_pixel': 0.0,
        'logits_energy': 0.0,
        'softmax_pixel': 0.0,
        'softmax_energy': 0.0,
        'multinomial_pixel': 0.0,
        'multinomial_energy': 0.0,
        'eos_handling': 0.0,
    }
    
    # Setup
    if initial_energy.dim() == 1:
        initial_energy = initial_energy.unsqueeze(1)
    init_e_embed = model.initial_energy_embedding(initial_energy).unsqueeze(1).to(dtype)
    
    is_done = torch.zeros(B, dtype=torch.bool, device=device)
    idx_buffer = torch.zeros((B, max_seq_len + 1), device=device, dtype=torch.long)
    e_buffer = torch.zeros((B, max_seq_len + 1), device=device, dtype=torch.long)
    
    idx_buffer[:, 0] = model.SOS_token
    e_buffer[:, 0] = 0
    
    # Allocate KV cache
    kv_caches = model._allocate_kv_cache(B, max_seq_len + 2, device, dtype)
    lora_modules = model._get_lora_modules_for_particle("gamma")
    
    logger.info(f"Running instrumented generation for {max_seq_len} steps...")
    
    with torch.inference_mode():
        with torch.amp.autocast('cuda', dtype=dtype):
            for step in range(max_seq_len):
                # === TOKEN EMBEDDING ===
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if step == 0:
                    tok_emb = model.token_embedding(idx_buffer[:, 0:1])
                else:
                    tok_emb = model.token_embedding(idx_buffer[:, step:step+1])
                torch.cuda.synchronize()
                times['embedding_token'] += time.perf_counter() - t0
                
                # === POSITION EMBEDDING ===
                t0 = time.perf_counter()
                if step == 0:
                    pos_idx = torch.zeros((B, 1), device=device, dtype=torch.long)
                else:
                    pos_idx = torch.full((B, 1), step, device=device, dtype=torch.long)
                pos_emb = model.pos_embedding(pos_idx)
                torch.cuda.synchronize()
                times['embedding_pos'] += time.perf_counter() - t0
                
                # === ENERGY EMBEDDING ===
                t0 = time.perf_counter()
                if step == 0:
                    e_emb = model.energy_embedding(e_buffer[:, 0:1])
                else:
                    e_emb = model.energy_embedding(e_buffer[:, step:step+1])
                torch.cuda.synchronize()
                times['embedding_energy'] += time.perf_counter() - t0
                
                # === ENERGY POSITION EMBEDDING ===
                t0 = time.perf_counter()
                e_pos_emb = model.energy_pos_embedding(pos_idx)
                torch.cuda.synchronize()
                times['embedding_energy_pos'] += time.perf_counter() - t0
                
                # Combine embeddings
                if step == 0:
                    x_t = torch.cat([init_e_embed, tok_emb + pos_emb], dim=1)
                    e_t = torch.cat([init_e_embed, e_emb + e_pos_emb], dim=1)
                else:
                    x_t = tok_emb + pos_emb
                    e_t = e_emb + e_pos_emb
                
                # === FORWARD PASS ===
                t0 = time.perf_counter()
                h_t, kv_caches = model.forward_decode_step(
                    x_t, e_t, material_index,
                    kv_caches=kv_caches,
                    is_first_step=(step == 0),
                    lora_modules=lora_modules
                )
                torch.cuda.synchronize()
                times['forward_pass'] += time.perf_counter() - t0
                
                # === PIXEL LOGITS ===
                t0 = time.perf_counter()
                pixel_logits = model.logits_head(h_t[:, -1, :])
                torch.cuda.synchronize()
                times['logits_pixel'] += time.perf_counter() - t0
                
                # === ENERGY LOGITS ===
                t0 = time.perf_counter()
                energy_logits = model.energy_head(h_t[:, -1, :])
                torch.cuda.synchronize()
                times['logits_energy'] += time.perf_counter() - t0
                
                # === PIXEL SOFTMAX ===
                t0 = time.perf_counter()
                pixel_probs = torch.softmax(pixel_logits, dim=-1, dtype=torch.float32)
                torch.cuda.synchronize()
                times['softmax_pixel'] += time.perf_counter() - t0
                
                # === ENERGY SOFTMAX ===
                t0 = time.perf_counter()
                energy_probs = torch.softmax(energy_logits, dim=-1, dtype=torch.float32)
                torch.cuda.synchronize()
                times['softmax_energy'] += time.perf_counter() - t0
                
                # === PIXEL MULTINOMIAL ===
                t0 = time.perf_counter()
                idx_next = torch.multinomial(pixel_probs, num_samples=1)
                torch.cuda.synchronize()
                times['multinomial_pixel'] += time.perf_counter() - t0
                
                # === ENERGY MULTINOMIAL ===
                t0 = time.perf_counter()
                e_next = torch.multinomial(energy_probs, num_samples=1)
                torch.cuda.synchronize()
                times['multinomial_energy'] += time.perf_counter() - t0
                
                # Clamp to valid range
                idx_next = idx_next.clamp(0, model.space_vocab - 1)
                e_next = e_next.clamp(0, model.energy_vocab - 1)
                
                # === EOS HANDLING ===
                t0 = time.perf_counter()
                newly_done = (e_next.squeeze(1) == model.EOS_energy_token) | \
                            (idx_next.squeeze(1) == model.EOS_token)
                pad_mask = is_done & ~newly_done
                eos_mask = newly_done
                is_done = is_done | newly_done
                
                idx_store = idx_next.squeeze(1).clone()
                idx_store.masked_fill_(eos_mask, model.EOS_token)
                idx_store.masked_fill_(pad_mask, model.pad_token)
                idx_buffer[:, step + 1] = idx_store
                
                e_store = e_next.squeeze(1).clone()
                e_store.masked_fill_(eos_mask, model.EOS_energy_token)
                e_store.masked_fill_(pad_mask, model.energy_pad_token)
                e_buffer[:, step + 1] = e_store
                torch.cuda.synchronize()
                times['eos_handling'] += time.perf_counter() - t0
                
                # Early exit check (infrequent)
                if (step + 1) % 100 == 0:
                    if torch.all(is_done):
                        logger.info(f"Early exit at step {step + 1}")
                        break
    
    # Calculate totals and percentages
    total_time = sum(times.values())
    
    # Group related operations
    groups = {
        'Embeddings': ['embedding_token', 'embedding_pos', 'embedding_energy', 'embedding_energy_pos'],
        'Forward Pass': ['forward_pass'],
        'Logits Heads': ['logits_pixel', 'logits_energy'],
        'Softmax': ['softmax_pixel', 'softmax_energy'],
        'Sampling': ['multinomial_pixel', 'multinomial_energy'],
        'EOS Handling': ['eos_handling'],
    }
    
    # Print detailed breakdown
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Component':<30} {'Time (ms)':<12} {'Per Step (μs)':<15} {'%':<8}")
    logger.info(f"{'='*60}")
    
    for component, t in sorted(times.items(), key=lambda x: -x[1]):
        per_step_us = (t / max_seq_len) * 1_000_000
        pct = 100 * t / total_time
        logger.info(f"{component:<30} {t*1000:<12.2f} {per_step_us:<15.1f} {pct:<8.1f}")
    
    logger.info(f"{'-'*60}")
    logger.info(f"{'TOTAL':<30} {total_time*1000:<12.2f}")
    
    # Print grouped summary
    logger.info(f"\n{'='*60}")
    logger.info(f"{'Group':<30} {'Time (ms)':<12} {'%':<8}")
    logger.info(f"{'='*60}")
    
    group_times = {}
    for group_name, components in groups.items():
        group_time = sum(times[c] for c in components)
        group_times[group_name] = group_time
    
    for group_name, group_time in sorted(group_times.items(), key=lambda x: -x[1]):
        pct = 100 * group_time / total_time
        logger.info(f"{group_name:<30} {group_time*1000:<12.2f} {pct:<8.1f}")
    
    logger.info(f"{'='*60}")
    
    # Recommendations based on results
    logger.info("\n===== Optimization Recommendations =====")
    
    if group_times['Sampling'] / total_time > 0.15:
        logger.info("→ Sampling is >15% of time. Consider Gumbel-max sampling or top-k filtering.")
    
    if group_times['Softmax'] / total_time > 0.10:
        logger.info("→ Softmax is >10% of time. Consider fused softmax+sampling or top-k before softmax.")
    
    if group_times['Embeddings'] / total_time > 0.15:
        logger.info("→ Embeddings are >15% of time. Consider fusing token+position embeddings.")
    
    if group_times['Logits Heads'] / total_time > 0.20:
        logger.info("→ Logits heads are >20% of time. Consider quantization or smaller vocab projection.")
    
    if group_times['Forward Pass'] / total_time < 0.30:
        logger.info("→ Forward pass is <30% of time. Model is overhead-bound, not compute-bound.")
        logger.info("   CUDA graphs should help significantly.")
    
    return times, group_times


def main(args):
    logger = setup_logger()

    warnings.filterwarnings('ignore', message='.*skipping cudagraphs.*')
    logging.getLogger('torch._inductor.utils').setLevel(logging.ERROR)
    logging.getLogger('torch._dynamo').setLevel(logging.ERROR)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    energies = np.load(args.energy_file).astype(np.float32)
    energies = np.random.choice(energies, size=args.batch_size, replace=True)

    initial_energy = torch.tensor(energies, device=device).contiguous()

    material_index = torch.full(
        (args.batch_size,),
        1,
        dtype=torch.long,
        device=device,
    ).contiguous()

    model = ECAL_GPT(
        vocab_size=27003,
        seq_len=2100,
        embed_dim=256,
        attn_heads=[8, 8, 8],
        num_blocks=6,
        hidden_units=128,
        digitize_energy=True,
        mlp_scale=2,
        energy_vocab=25003,
        drop_rates=[0.0, 0.0, 0.0],
        use_MoE=True,
        num_experts=3,
        material_list=["G4_W", "G4_Ta", "G4_Pb"],
        device=device,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["net_state_dict"], strict=False)

    model = model.to(device)
    model.device = device

    if args.fuse_qkv:
        logger.info("Fusing QKV projections for optimization...")
        model = fuse_qkv_weights(model)
        model = fuse_cross_attention_kv(model)
        logger.info("QKV fusion complete")

    if args.quantize:
        model = quantize_model_bnb(model, logger)
        model.set_skip_compile(True)
    elif args.quantize_selective:
        model = quantize_model_bnb_selective(
            model, 
            logger,
            quantize_attention=not args.no_quantize_attention,
            quantize_mlp=not args.no_quantize_mlp,
            quantize_heads=not args.no_quantize_heads,
        )
        model.set_skip_compile(True)


    # Don't use torch.compile with CUDA graphs - they conflict
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
    logger.info("================================")
    if args.instrument:
        benchmark_components(
            model, initial_energy, material_index, logger,
            max_seq_len=args.instrument_steps
        )
        return
    if args.profile:
        profile_model(model, initial_energy, material_index, 
                     use_kv_cache=True, logger=logger)
        return

    # New: benchmark all modes
    if args.benchmark_all:
        benchmark_all_modes(
            model, initial_energy, material_index, logger,
            iters=args.iters,
            max_seq_len=args.max_seq_len,
        )
        return

    # Single mode benchmark
    if args.cuda_graph:
        t, peak_mem_mb = benchmark(
            model, initial_energy, material_index,
            use_kv_cache=True,
            use_cuda_graph=True,
            iters=args.iters,
            logger=logger,
        )
        samples_per_sec = args.batch_size / t
        ms_per_sample = (t / args.batch_size) * 1000
        logger.info(
            f"CUDA Graph     | "
            f"Batch latency = {t * 1000:.2f} ms | "
            f"Per-sample = {ms_per_sample:.2f} ms | "
            f"Throughput = {samples_per_sec:.2f} samples/s | "
            f"Peak GPU Memory = {peak_mem_mb:.1f} MB"
        )
    else:
        # Original benchmark loop
        if device == "cpu":
            torch.set_autocast_enabled(False)
        t, peak_mem_mb = benchmark(
            model, initial_energy, material_index,
            use_kv_cache=True,
            use_cuda_graph=False,
            iters=args.iters,
            logger=logger,
        )
        
        samples_per_sec = args.batch_size / t
        ms_per_sample = (t / args.batch_size) * 1000
        logger.info(
            f"KV Cache = {True:<5} | "
            f"Batch latency = {t * 1000:.2f} ms | "
            f"Per-sample = {ms_per_sample:.2f} ms | "
            f"Throughput = {samples_per_sec:.2f} samples/s | "
            f"Peak GPU Memory = {peak_mem_mb:.1f} MB"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--energy_file", default="initial_energies.npy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=70)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max_seq_len", type=int, default=2100,
                       help="Maximum sequence length for generation")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fuse_qkv", action="store_true",
                       help="Fuse Q, K, V projections for faster inference")
    parser.add_argument("--quantize", action="store_true",
                       help="Apply INT8 quantization using bitsandbytes")
    parser.add_argument("--quantize_selective", action="store_true",
                       help="Apply selective INT8 quantization (more control)")
    parser.add_argument("--no_quantize_attention", action="store_true",
                       help="Don't quantize attention layers (use with --quantize_selective)")
    parser.add_argument("--no_quantize_mlp", action="store_true",
                       help="Don't quantize MLP layers (use with --quantize_selective)")
    parser.add_argument("--no_quantize_heads", action="store_true",
                       help="Don't quantize output heads (use with --quantize_selective)")
    parser.add_argument("--profile", action="store_true",
                       help="Run profiler to identify bottlenecks")
    parser.add_argument("--cuda_graph", action="store_true",
                       help="Use CUDA graphs for maximum inference speed")
    parser.add_argument("--benchmark_all", action="store_true",
                       help="Benchmark all three modes: no cache, KV cache, CUDA graph")
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument("--instrument_steps", type=int, default=500)
    args = parser.parse_args()

    main(args)