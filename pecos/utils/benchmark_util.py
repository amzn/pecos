#  Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
#  with the License. A copy of the License is located at
#
#  http://aws.amazon.com/apache2.0/
#
#  or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
#  OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
#  and limitations under the License.
"""
Performance benchmarking and profiling utilities for PECOS.

Optimized for large-scale datasets (10M+ rows) with GPU acceleration.
"""
import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


class PerformanceProfiler:
    """Context manager for profiling performance of code blocks.

    Tracks execution time, GPU memory usage, and throughput metrics.

    Usage:
        with PerformanceProfiler("training") as prof:
            # your code here
            prof.log_metric("samples_processed", 10000)

        print(prof.get_summary())
    """

    def __init__(self, name: str, auto_log: bool = True):
        """Initialize profiler.

        Args:
            name (str): Name of the profiling section
            auto_log (bool): If True, automatically log summary on exit. Default True
        """
        self.name = name
        self.auto_log = auto_log
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.gpu_mem_start = None
        self.gpu_mem_end = None

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.time()

        # Record GPU memory at start
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_mem_start = torch.cuda.memory_allocated() / 1024**3  # GB
        except Exception:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and optionally log results."""
        self.end_time = time.time()

        # Record GPU memory at end
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_mem_end = torch.cuda.memory_allocated() / 1024**3  # GB
        except Exception:
            pass

        if self.auto_log:
            LOGGER.info(self.get_summary())

    def log_metric(self, metric_name: str, value: float):
        """Log a custom metric.

        Args:
            metric_name (str): Name of the metric
            value (float): Value of the metric
        """
        self.metrics[metric_name] = value

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def get_summary(self) -> str:
        """Get formatted summary of profiling results."""
        elapsed = self.get_elapsed_time()
        summary_lines = [f"[{self.name}] Performance Summary:"]
        summary_lines.append(f"  Elapsed Time: {elapsed:.2f}s")

        if self.gpu_mem_start is not None and self.gpu_mem_end is not None:
            mem_used = self.gpu_mem_end - self.gpu_mem_start
            summary_lines.append(f"  GPU Memory Used: {mem_used:.2f} GB")
            summary_lines.append(f"  GPU Memory Total: {self.gpu_mem_end:.2f} GB")

        if "samples_processed" in self.metrics:
            throughput = self.metrics["samples_processed"] / elapsed
            summary_lines.append(f"  Throughput: {throughput:.1f} samples/sec")

        for metric, value in self.metrics.items():
            if metric != "samples_processed":
                summary_lines.append(f"  {metric}: {value}")

        return "\n".join(summary_lines)


@contextmanager
def profile_block(name: str):
    """Simple context manager for quick profiling.

    Args:
        name (str): Name of the block to profile

    Usage:
        with profile_block("data loading"):
            # your code here
            pass
    """
    profiler = PerformanceProfiler(name, auto_log=True)
    with profiler:
        yield profiler


def benchmark_inference(
    model,
    X_text,
    batch_sizes=[8, 16, 32, 64],
    num_runs=3,
    warmup_runs=1
) -> Dict[int, Dict[str, float]]:
    """Benchmark inference performance across different batch sizes.

    Args:
        model: TransformerMatcher model
        X_text: Input text data (tokenized)
        batch_sizes (list): List of batch sizes to test
        num_runs (int): Number of runs per batch size. Default 3
        warmup_runs (int): Number of warmup runs. Default 1

    Returns:
        dict: Results dictionary with structure:
            {batch_size: {"time": avg_time, "throughput": samples_per_sec, "gpu_mem": peak_gb}}
    """
    import torch

    results = {}

    for batch_size in batch_sizes:
        LOGGER.info(f"Benchmarking batch_size={batch_size}...")

        times = []
        gpu_mems = []

        # Warmup
        for _ in range(warmup_runs):
            _ = model.predict(X_text, batch_size=batch_size, only_embeddings=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Actual runs
        for run in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            _ = model.predict(X_text, batch_size=batch_size, only_embeddings=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - start_time

            times.append(elapsed)

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                gpu_mems.append(peak_mem)

        avg_time = np.mean(times)
        num_samples = len(X_text["input_ids"])
        throughput = num_samples / avg_time

        results[batch_size] = {
            "time": avg_time,
            "throughput": throughput,
            "gpu_mem": np.mean(gpu_mems) if gpu_mems else 0
        }

        LOGGER.info(
            f"  batch_size={batch_size}: {avg_time:.2f}s, "
            f"{throughput:.1f} samples/s, "
            f"{results[batch_size]['gpu_mem']:.2f} GB"
        )

    return results


def estimate_optimal_batch_size(
    dataset_size: int,
    num_labels: int,
    hidden_dim: int = 768,
    max_seq_length: int = 128,
    available_gpu_memory_gb: Optional[float] = None
) -> int:
    """Estimate optimal batch size for large datasets.

    Uses heuristics based on dataset size, model architecture, and GPU memory.

    Args:
        dataset_size (int): Number of training samples
        num_labels (int): Number of output labels
        hidden_dim (int): Transformer hidden dimension. Default 768
        max_seq_length (int): Maximum sequence length. Default 128
        available_gpu_memory_gb (float, optional): Available GPU memory in GB.
                                                     Default None to auto-detect

    Returns:
        int: Recommended batch size
    """
    # Auto-detect GPU memory if not provided
    if available_gpu_memory_gb is None:
        try:
            import torch
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                # Use 80% of available memory
                available_gpu_memory_gb = total_mem * 0.8
            else:
                available_gpu_memory_gb = 0
        except Exception:
            available_gpu_memory_gb = 0

    if available_gpu_memory_gb == 0:
        # CPU-only, use conservative batch size
        return min(8, max(1, dataset_size // 1000))

    # Estimate memory per sample (in GB)
    # Formula: seq_len * hidden_dim * 4 bytes (float32) * 2 (forward + backward)
    mem_per_sample = (max_seq_length * hidden_dim * 4 * 2) / 1024**3

    # Add overhead for model weights and gradients
    model_overhead = (hidden_dim * num_labels * 4) / 1024**3  # Simplified
    available_for_batch = available_gpu_memory_gb - model_overhead - 2  # 2GB safety margin

    if available_for_batch <= 0:
        return 1

    # Calculate batch size
    estimated_batch_size = int(available_for_batch / mem_per_sample)

    # Apply heuristics based on dataset size
    if dataset_size > 10000000:  # 10M+
        # Very large dataset: use larger batches for efficiency
        batch_size = max(32, min(estimated_batch_size, 128))
    elif dataset_size > 1000000:  # 1M+
        # Large dataset: moderate batch size
        batch_size = max(16, min(estimated_batch_size, 64))
    else:
        # Smaller dataset: can use smaller batches
        batch_size = max(8, min(estimated_batch_size, 32))

    LOGGER.info(
        f"Estimated optimal batch size: {batch_size} "
        f"(dataset_size={dataset_size}, available_memory={available_gpu_memory_gb:.2f}GB)"
    )

    return batch_size


def log_system_info():
    """Log system information including GPU details."""
    import platform

    LOGGER.info("System Information:")
    LOGGER.info(f"  Platform: {platform.system()} {platform.release()}")
    LOGGER.info(f"  Python: {platform.python_version()}")

    try:
        import torch
        LOGGER.info(f"  PyTorch: {torch.__version__}")
        LOGGER.info(f"  CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            LOGGER.info(f"  CUDA Version: {torch.version.cuda}")
            LOGGER.info(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                LOGGER.info(f"  GPU {i}: {props.name}")
                LOGGER.info(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
                LOGGER.info(f"    Compute Capability: {props.major}.{props.minor}")
    except ImportError:
        LOGGER.warning("  PyTorch not installed")

    try:
        import numpy as np
        LOGGER.info(f"  NumPy: {np.__version__}")
    except ImportError:
        pass

    try:
        import scipy
        LOGGER.info(f"  SciPy: {scipy.__version__}")
    except ImportError:
        pass
