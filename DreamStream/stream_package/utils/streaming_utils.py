"""Streaming utility functions for control video-based generation."""

from typing import Tuple, Union


def calculate_frame_counts(stream_idx: int, num_frame_per_block: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Calculate pixel frame range for a VAE block.

    VAE encoding: 4n+1 pixel frames -> n+1 latent frames.
    """
    if isinstance(num_frame_per_block, int):
        if stream_idx == 0:
            pixel_frames = 1 + (num_frame_per_block - 1) * 4
            return 0, pixel_frames
        else:
            first_block_frames = 1 + (num_frame_per_block - 1) * 4
            start = first_block_frames + (stream_idx - 1) * (num_frame_per_block * 4)
            end = start + num_frame_per_block * 4
            return start, end
    else:
        first_size, subsequent_size = num_frame_per_block
        if stream_idx == 0:
            if first_size == 1:
                pixel_frames = 1
            else:
                pixel_frames = 1 + (first_size - 1) * 4
            return 0, pixel_frames
        else:
            if first_size == 1:
                first_block_frames = 1
            else:
                first_block_frames = 1 + (first_size - 1) * 4

            start = first_block_frames + (stream_idx - 1) * (subsequent_size * 4)
            end = start + subsequent_size * 4
            return start, end
