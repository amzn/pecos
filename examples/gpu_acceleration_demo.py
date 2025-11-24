#!/usr/bin/env python3
"""
GPU Acceleration Demo for PECOS

This script demonstrates the GPU-accelerated features for large-scale datasets (10M+ rows).

Key features demonstrated:
1. Automatic GPU detection and batch size optimization
2. GPU-accelerated prediction post-processing
3. Efficient DataLoader configuration for large datasets
4. Performance profiling and benchmarking

Usage:
    python examples/gpu_acceleration_demo.py
"""
import logging
import numpy as np
import scipy.sparse as smat
from pecos.utils import torch_util, benchmark_util

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def demo_gpu_detection():
    """Demo 1: GPU detection and memory info"""
    LOGGER.info("=" * 80)
    LOGGER.info("DEMO 1: GPU Detection and Memory Info")
    LOGGER.info("=" * 80)

    benchmark_util.log_system_info()

    mem_info = torch_util.get_gpu_memory_info()
    LOGGER.info(f"\nGPU Memory Status:")
    LOGGER.info(f"  Allocated: {mem_info['allocated']:.2f} GB")
    LOGGER.info(f"  Reserved: {mem_info['reserved']:.2f} GB")
    LOGGER.info(f"  Free: {mem_info['free']:.2f} GB")
    LOGGER.info(f"  Total: {mem_info['total']:.2f} GB")


def demo_auto_batch_sizing():
    """Demo 2: Automatic batch size optimization"""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("DEMO 2: Automatic Batch Size Optimization")
    LOGGER.info("=" * 80)

    # Test different dataset sizes
    dataset_sizes = [10000, 100000, 1000000, 10000000]

    for size in dataset_sizes:
        optimal_bs = torch_util.auto_batch_size(size, base_batch_size=32)
        estimated_bs = benchmark_util.estimate_optimal_batch_size(
            dataset_size=size,
            num_labels=5000,
            hidden_dim=768,
            max_seq_length=128
        )
        LOGGER.info(f"Dataset size: {size:>10,}")
        LOGGER.info(f"  Auto batch size: {optimal_bs}")
        LOGGER.info(f"  Estimated batch size: {estimated_bs}")


def demo_gpu_sparse_operations():
    """Demo 3: GPU-accelerated sparse matrix operations"""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("DEMO 3: GPU-Accelerated Sparse Operations")
    LOGGER.info("=" * 80)

    # Create a sample large sparse matrix (simulating 100K rows, 5K labels)
    num_rows = 100000
    num_cols = 5000
    nnz_per_row = 50

    LOGGER.info(f"Creating sparse matrix: {num_rows} x {num_cols} with ~{nnz_per_row} nnz per row")

    # Generate random COO data
    row_idx = np.repeat(np.arange(num_rows), nnz_per_row)
    col_idx = np.random.randint(0, num_cols, size=len(row_idx))
    val = np.random.randn(len(row_idx)).astype(np.float32)

    # Benchmark CPU vs GPU
    import time

    # CPU version
    LOGGER.info("\nCPU Version:")
    start = time.time()
    cpu_csr = smat.csr_matrix((val, (row_idx, col_idx)), shape=(num_rows, num_cols))
    cpu_time = time.time() - start
    LOGGER.info(f"  Time: {cpu_time:.3f}s")

    # GPU version (if available)
    try:
        import torch
        if torch.cuda.is_available():
            LOGGER.info("\nGPU Version:")
            start = time.time()
            gpu_result = torch_util.torch_sorted_csr_from_coo(
                shape=(num_rows, num_cols),
                row_idx=row_idx,
                col_idx=col_idx,
                val=val,
                only_topk=10
            )
            gpu_time = time.time() - start
            LOGGER.info(f"  Time: {gpu_time:.3f}s")
            LOGGER.info(f"  Speedup: {cpu_time / gpu_time:.2f}x")
        else:
            LOGGER.info("\nGPU not available, skipping GPU benchmark")
    except Exception as e:
        LOGGER.warning(f"GPU benchmark failed: {e}")


def demo_performance_profiling():
    """Demo 4: Performance profiling"""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("DEMO 4: Performance Profiling")
    LOGGER.info("=" * 80)

    # Simulate some computation
    with benchmark_util.PerformanceProfiler("matrix_operations") as prof:
        # Simulate matrix operations
        import time
        time.sleep(0.1)

        # Create large matrices
        A = np.random.randn(10000, 1000).astype(np.float32)
        B = np.random.randn(1000, 500).astype(np.float32)

        # Matrix multiplication
        C = np.dot(A, B)

        prof.log_metric("samples_processed", 10000)
        prof.log_metric("matrix_shape", f"{A.shape} x {B.shape}")

    LOGGER.info("\nProfile completed! Summary was automatically logged above.")


def demo_usage_recommendations():
    """Demo 5: Usage recommendations for 10M+ datasets"""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("DEMO 5: Usage Recommendations for 10M+ Row Datasets")
    LOGGER.info("=" * 80)

    recommendations = """
    For optimal performance with 10M+ row datasets:

    1. TRAINING:
       - Use batch_size=32-128 depending on GPU memory
       - Enable gradient accumulation if OOM: gradient_accumulation_steps=2-4
       - Use pre-tokenization: pre_tokenize=True
       - Enable persistent_workers: automatically enabled for 1M+ datasets
       - Monitor GPU memory: Use torch_util.get_gpu_memory_info()

    2. INFERENCE:
       - Use larger batch sizes: batch_size=64-256
       - Enable GPU acceleration: It's automatic for 100K+ rows
       - Process in chunks if still OOM:
         ```python
         chunk_size = 1000000  # 1M at a time
         for i in range(0, len(data), chunk_size):
             chunk = data[i:i+chunk_size]
             predictions = model.predict(chunk, batch_size=128)
         ```

    3. MEMORY MANAGEMENT:
       - Pre-tokenize and save: Saves ~30% memory
       - Use mixed precision (FP16): Add to train_params
       - Clear cache between large operations:
         torch.cuda.empty_cache()

    4. DATALOADER OPTIMIZATION:
       - num_workers=4-8 for fast storage (SSD/NVMe)
       - pin_memory=True for fixed-size batches (automatic)
       - prefetch_factor=2 for better pipeline (automatic)

    5. MONITORING:
       - Use PerformanceProfiler for bottleneck detection
       - Monitor GPU utilization: nvidia-smi dmon
       - Check throughput: samples/sec should be >1000 for GPU

    Example Training Command:
    ```python
    from pecos.xmc.xtransformer import XTransformer

    # For 10M rows, 5K labels
    xtf = XTransformer.train(
        train_text,
        train_labels,
        train_params={
            "batch_size": 64,           # Auto-adjusted if needed
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 3,
            "learning_rate": 5e-5,
            "pre_tokenize": True,       # Highly recommended
            "batch_gen_workers": 8,     # For fast data loading
        }
    )
    ```

    Example Prediction:
    ```python
    # Will automatically use GPU acceleration
    predictions = xtf.predict(
        test_text,
        batch_size=128,              # Larger batch for inference
        batch_gen_workers=8,
    )
    ```
    """

    LOGGER.info(recommendations)


def main():
    """Run all demos"""
    LOGGER.info("\n" + "#" * 80)
    LOGGER.info("# PECOS GPU Acceleration Demo")
    LOGGER.info("# Optimized for Large-Scale Datasets (10M+ rows)")
    LOGGER.info("#" * 80)

    demo_gpu_detection()
    demo_auto_batch_sizing()
    demo_gpu_sparse_operations()
    demo_performance_profiling()
    demo_usage_recommendations()

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("All demos completed!")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
