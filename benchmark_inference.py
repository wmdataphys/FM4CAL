import torch
import numpy as np
import time
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
import torch.profiler as profiler

from models.GPT import ECAL_GPT

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
    
    with torch.inference_mode():
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            model.generate(
                initial_energy=initial_energy,
                material_index=material_index,
                max_seq_len=500,  # Fixed length for profiling
                temperature=1.0,
                use_kv_cache=use_kv_cache,
            )
    
    # Print top operations by CUDA time
    logger.info("\n----- Top 20 Operations by CUDA Time -----")
    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
    )
    logger.info(f"\n{table}")
    
    # Save detailed trace
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

    # Warmup
    with torch.inference_mode():
        for _ in tqdm(
            range(warmup),
            desc=f"Warmup (KV={use_kv_cache})",
            leave=False,
        ):
            model.generate(
                initial_energy=initial_energy,
                material_index=material_index,
                max_seq_len=max_seq_len,
                temperature=1.0,
                use_kv_cache=use_kv_cache,
            )

    sync()

    # Timed runs
    reset_gpu_memory()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in tqdm(
            range(iters),
            desc=f"Benchmark (KV={use_kv_cache})",
            leave=False,
        ):
            model.generate(
                initial_energy=initial_energy,
                material_index=material_index,
                max_seq_len=max_seq_len,
                temperature=1.0,
                use_kv_cache=use_kv_cache,
            )

    sync()
    t1 = time.perf_counter()

    avg_time = (t1 - t0) / iters
    peak_mem_mb = get_peak_memory_mb()
    return avg_time, peak_mem_mb


def benchmark_sequence_lengths(
    model,
    initial_energy,
    material_index,
    logger,
):
    """
    Test KV cache speedup at different sequence lengths.
    """
    logger.info("\n===== Testing KV Cache at Different Sequence Lengths =====")
    
    seq_lengths = [100, 300, 500, 1000, 1500]
    
    for max_len in seq_lengths:
        logger.info(f"\n--- Sequence Length: {max_len} ---")
        
        times = {}
        for kv in (True, False):
            t, mem = benchmark(
                model,
                initial_energy,
                material_index,
                use_kv_cache=kv,
                iters=10,
                warmup=3,
                max_seq_len=max_len,
                logger=logger,
            )
            times[kv] = t
            logger.info(
                f"  KV={kv:<5}: {t*1000:.2f}ms | "
                f"{mem:.1f}MB"
            )
        
        # Calculate speedup
        speedup = times[False] / times[True]
        logger.info(f"  Speedup with KV cache: {speedup:.2f}x")


def main(args):

    logger = setup_logger()

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # Load Ta-only initial energies
    energies = np.load(args.energy_file).astype(np.float32)
    energies = np.random.choice(
        energies, size=args.batch_size, replace=True
    )

    initial_energy = torch.tensor(
        energies, device=device
    ).contiguous()

    material_index = torch.full(
        (args.batch_size,),
        1,  # G4_Ta
        dtype=torch.long,
        device=device,
    ).contiguous()

    model = ECAL_GPT(
        vocab_size=27003,
        seq_len=1700,
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
    model.load_state_dict(checkpoint["net_state_dict"], strict=True)

    if args.compile and device == "cuda":
        logger.info("Compiling model with torch.compile(...)")
        model = torch.compile(
            model, mode="reduce-overhead", dynamic=True
        )
    
    log_sdpa_backends(logger)
    logger.info("===== Inference Benchmark (MoE / Ta-only) =====")
    logger.info(f"Device         : {device}")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info(f"Iterations     : {args.iters}")
    logger.info("Material       : G4_Ta")
    logger.info(
        f"Experts active : {model.num_experts // model.num_classes}"
    )
    logger.info("=============================================")

    # Run profiling if requested
    if args.profile:
        logger.info("\n" + "="*60)
        logger.info("PROFILING MODE")
        logger.info("="*60)
        
        # Profile both KV cache on and off
        profile_model(model, initial_energy, material_index, 
                     use_kv_cache=True, logger=logger)
        profile_model(model, initial_energy, material_index, 
                     use_kv_cache=False, logger=logger)
        
        logger.info("\nProfiler traces saved. Exiting after profiling.")
        return

    # Test sequence length scaling if requested
    if args.test_seq_lengths:
        benchmark_sequence_lengths(
            model, initial_energy, material_index, logger
        )
        return

    # Standard benchmark
    for kv in (True, False):
        t, peak_mem_mb = benchmark(
            model,
            initial_energy,
            material_index,
            use_kv_cache=kv,
            iters=args.iters,
            logger=logger,
        )
        
        samples_per_sec = args.batch_size / t
        ms_per_sample = (t / args.batch_size) * 1000
        logger.info(
            f"KV Cache = {kv:<5} | "
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
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true",
                       help="Run profiler to identify bottlenecks")
    parser.add_argument("--test_seq_lengths", action="store_true",
                       help="Test KV cache speedup at different sequence lengths")
    args = parser.parse_args()

    main(args)