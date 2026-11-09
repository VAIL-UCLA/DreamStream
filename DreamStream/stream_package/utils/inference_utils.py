import numpy as np
import torch
from PIL import Image


def prepare_inputs_fun(control_video, ref_image, size=(832, 480), device="cuda"):
    if isinstance(control_video, torch.Tensor):
        control_video = control_video.to(device, dtype=torch.bfloat16)
    elif control_video is not None:
        raise ValueError("Unsupported control video format. Please provide a tensor.")

    if isinstance(ref_image, str):
        ref_image = Image.open(ref_image).convert("RGB").resize(size, Image.BICUBIC)
        ref_image = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).unsqueeze(0).to(device)
        ref_image = 2 * (ref_image / 255.0) - 1  # to [-1, 1], 1, 3, H, W
    elif isinstance(ref_image, Image.Image):
        ref_image = ref_image.convert("RGB").resize(size, Image.BICUBIC)
        ref_image = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).unsqueeze(0).to(device)
        ref_image = 2 * (ref_image / 255.0) - 1
    elif isinstance(ref_image, torch.Tensor):
        ref_image = ref_image.to(device)
    elif ref_image is not None:
        raise ValueError("Unsupported reference image format. Please provide a (path to) PIL Image or a tensor.")

    return control_video, ref_image


def initialize_kv_cache(batch_size, seq_len, num_layers, num_heads, head_dim, dtype, device):
    """
    Initialize KV cache for the Wan model.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_layers: Number of model layers
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dtype: Data type for the cache
        device: Device to create the cache on

    Returns:
        List of dictionaries containing 'k' and 'v' tensors for each layer
    """
    kv_cache = []
    for _ in range(num_layers):
        kv_cache.append(
            {
                "k": torch.zeros([batch_size, seq_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, seq_len, num_heads, head_dim], dtype=dtype, device=device),
            }
        )
    return kv_cache


def initialize_img_crossattn_cache(batch_size, seq_len, num_layers, num_heads, head_dim, dtype, device):
    """
    Initialize cross-attention cache for the image control model.

    Args:
        batch_size: Batch size
        seq_len: Text sequence length
        num_layers: Number of model layers
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        dtype: Data type for the cache
        device: Device to create the cache on

    Returns:
        List of dictionaries containing cross-attention cache for each layer
    """
    crossattn_cache = []
    for _ in range(num_layers):
        crossattn_cache.append(
            {
                "k_txt": torch.zeros([batch_size, seq_len, num_heads, head_dim], dtype=dtype, device=device),
                "v_txt": torch.zeros([batch_size, seq_len, num_heads, head_dim], dtype=dtype, device=device),
                "k_img": torch.zeros([batch_size, 257, num_heads, head_dim], dtype=dtype, device=device),
                "v_img": torch.zeros([batch_size, 257, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False,
            }
        )
    return crossattn_cache


class GPUTimer:
    """Minimal GPU timer for performance profiling with automatic enable/disable handling."""

    def __init__(self, enabled=True):
        """
        Initialize the GPU timer.

        Args:
            enabled: Whether timing is enabled. When False, all operations become no-ops.
        """
        self.enabled = enabled
        self.timings = {} if enabled else None
        self.stack = [] if enabled else None

    def __call__(self, name):
        """Allow using timer as context manager: with timer('section_name'):"""
        return self._Section(self, name)

    class _Section:
        """Context manager for timing a code section."""

        def __init__(self, timer, name):
            self.timer = timer
            self.name = name

        def __enter__(self):
            if self.timer.enabled:
                start = torch.cuda.Event(enable_timing=True)
                start.record()
                self.timer.stack.append((self.name, start))
            return self

        def __exit__(self, *args):
            if self.timer.enabled:
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                name, start = self.timer.stack.pop()
                if name not in self.timer.timings:
                    self.timer.timings[name] = []
                self.timer.timings[name].append((start, end))

    def report(self, prefix=""):
        """
        Generate and print timing report. Safe to call even when disabled.

        Args:
            prefix: Optional prefix for the report header (e.g., "[Stream 0] ")
        """
        if not self.enabled or not self.timings:
            return

        torch.cuda.synchronize()

        # Calculate section times
        section_times = {}
        for name, events in self.timings.items():
            section_times[name] = sum(start.elapsed_time(end) for start, end in events)

        if not section_times:
            return

        # Total time is sum of all sections (assuming non-overlapping)
        total_time = sum(section_times.values())

        # Print report
        print(f"\n{prefix}Timing breakdown (ms):")
        print(f"  Total time: {total_time:.2f}")

        # Sort by time (descending) for better readability
        for name, time_ms in sorted(section_times.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_ms / total_time * 100) if total_time > 0 else 0
            print(f"  {name}: {time_ms:.2f} ({percentage:.1f}%)")

    def reset(self):
        """Clear all timings. Safe to call even when disabled."""
        if self.enabled and self.timings:
            self.timings.clear()
            self.stack.clear()
