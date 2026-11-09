"""
TinyVAE: Refactored Tiny AutoEncoder for Video Generation
Adapted from https://github.com/madebyollin/taehv/blob/main/taehv.py
(MIT License, Copyright (c) 2025 Ollin Boer Bohan; see NOTICE)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


class Clamp(nn.Module):
    """Clamp activation to prevent extreme values."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Memory block for temporal coherence in causal video processing.

    Concatenates current and previous frame features for temporal modeling.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in * 2, n_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Process current frame with memory from previous frame."""
        return self.act(self.conv(torch.cat([x, memory], dim=1)) + self.skip(x))


class TPool(nn.Module):
    """Temporal pooling layer to reduce temporal dimension."""

    def __init__(self, n_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_channels * stride, n_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool temporal dimension by combining consecutive frames."""
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    """Temporal growth layer to expand temporal dimension."""

    def __init__(self, n_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_channels, n_channels * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand temporal dimension by generating multiple frames."""
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class TinyVAE(ModelMixin, ConfigMixin):
    """Tiny causal video autoencoder with streaming encode/decode (taew2_2 layout)."""

    def __init__(self):
        super().__init__()

        # 2x2-patchified RGB (12 channels) in, 48-channel Wan2.2 latents.
        self.patch_size = 2
        self.get_version1(image_channels=12, latent_channels=48)

        self.frames_to_pad_or_trim = 3
        self._encoder_cache = None
        self._decoder_cache = None

    def get_version1(self, image_channels=3, latent_channels=16):
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            TPool(64, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            Clamp(),
            nn.Conv2d(latent_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            MemBlock(256, 256),
            MemBlock(256, 256),
            MemBlock(256, 256),
            nn.Upsample(scale_factor=2),
            TGrow(256, 1),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            MemBlock(128, 128),
            MemBlock(128, 128),
            MemBlock(128, 128),
            nn.Upsample(scale_factor=2),
            TGrow(128, 2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            nn.Upsample(scale_factor=2),
            TGrow(64, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, image_channels, kernel_size=3, padding=1),
        )

    def clean_encoder_cache(self):
        """Clear encoder cache for streaming. Compatible with WanVAEWrapper interface."""
        self._encoder_cache = None

    def clean_decoder_cache(self):
        """Clear decoder cache for streaming. Compatible with WanVAEWrapper interface."""
        self._decoder_cache = None

    @classmethod
    def from_pretrained(cls, checkpoint_path: Optional[str] = None, **kwargs) -> "TinyVAE":
        """Load a TinyVAE from a taew2_2 checkpoint (.safetensors or .pt)."""
        state_dict = None
        if checkpoint_path is not None:
            if str(checkpoint_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            # remove module. if it exists
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model = cls()
        if state_dict is not None:
            model.load_state_dict(state_dict, strict=True)
        return model

    def encode(
        self,
        x: torch.Tensor,
        start: bool = True,
        end: bool = True,
        to_0_1: bool = True,
    ) -> torch.Tensor:
        """Encode video frames to latent representation.

        Args:
            x: Input tensor of shape (N, T, C, H, W)
            start: Clear encoder cache at start (for streaming)
            end: Clear encoder cache at end (for streaming)
            to_0_1: If True, expect input in [-1, 1] and convert to [0, 1]

        Returns:
            Latent tensor of shape (N, T', C', H', W')
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor (NTCHW), got {x.ndim}D")

        if to_0_1:
            x = (x + 1) / 2

        if x.shape[1] % 4 == 1 and self.frames_to_pad_or_trim > 0:
            first_frame = x[:, :1]
            padding = first_frame.repeat(1, self.frames_to_pad_or_trim, 1, 1, 1)
            x = torch.cat([padding, x], dim=1)

        if self.patch_size > 1:
            x = F.pixel_unshuffle(x.flatten(0, 1), self.patch_size).unflatten(0, x.shape[:2])

        return self._encode_streaming(x, start, end)

    def decode(
        self,
        z: torch.Tensor,
        start: bool = True,
        end: bool = True,
        to_neg_1_1: bool = True,
    ) -> torch.Tensor:
        """Decode latent representation to video frames.

        Args:
            z: Latent tensor of shape (N, T, C, H, W)
            start: Clear decoder cache at start (for streaming)
            end: Clear decoder cache at end (for streaming)
            to_neg_1_1: If True, convert output from [0, 1] to [-1, 1]

        Returns:
            Video tensor of shape (N, T', C', H', W')
        """
        if z.ndim != 5:
            raise ValueError(f"Expected 5D tensor (NTCHW), got {z.ndim}D")

        result = self._decode_streaming(z, start, end)

        if self.patch_size > 1:
            result = F.pixel_shuffle(result.flatten(0, 1), self.patch_size).unflatten(0, result.shape[:2])

        # Use in-place operations to avoid extra memory allocation
        result.clamp_(0, 1)

        if to_neg_1_1:
            result = 2 * result - 1

        return result

    def _encode_streaming(self, x: torch.Tensor, start: bool, end: bool) -> torch.Tensor:
        """Frame-by-frame encoding using work queue for memory efficiency."""
        N, T, C, H, W = x.shape

        # Initialize cache if starting
        if start:
            self.clean_encoder_cache()
            self._encoder_cache = [None] * len(self.encoder)

        layer_memory = self._encoder_cache if self._encoder_cache is not None else [None] * len(self.encoder)

        # Split input into individual frames
        frames = x.reshape(N, T * C, H, W).chunk(T, dim=1)

        work_queue = [(frame, 0) for frame in frames]
        output_frames = []

        while work_queue:
            frame_data, layer_idx = work_queue.pop(0)

            if layer_idx >= len(self.encoder):
                output_frames.append(frame_data)
                continue

            layer = self.encoder[layer_idx]

            if isinstance(layer, MemBlock):
                if layer_memory[layer_idx] is None:
                    output = layer(frame_data, frame_data * 0)
                    layer_memory[layer_idx] = frame_data
                else:
                    output = layer(frame_data, layer_memory[layer_idx])
                    layer_memory[layer_idx].copy_(frame_data)

                work_queue.insert(0, (output, layer_idx + 1))

            elif isinstance(layer, TPool):
                if layer_memory[layer_idx] is None:
                    layer_memory[layer_idx] = []

                layer_memory[layer_idx].append(frame_data)

                if len(layer_memory[layer_idx]) == layer.stride:
                    N, C, H, W = frame_data.shape
                    stacked = torch.cat(layer_memory[layer_idx], dim=1).view(N * layer.stride, C, H, W)
                    pooled = layer(stacked)
                    layer_memory[layer_idx] = []
                    work_queue.insert(0, (pooled, layer_idx + 1))

            else:
                output = layer(frame_data)
                work_queue.insert(0, (output, layer_idx + 1))

        # Save cache state if not ending
        if not end:
            self._encoder_cache = layer_memory
        else:
            self.clean_encoder_cache()

        if output_frames:
            return torch.stack(output_frames, dim=1)
        else:
            raise ValueError("No output frames produced")

    def _decode_streaming(self, z: torch.Tensor, start: bool, end: bool) -> torch.Tensor:
        """Frame-by-frame decoding using work queue for memory efficiency."""
        N, T, C, H, W = z.shape

        # Initialize cache if starting
        if start:
            self.clean_decoder_cache()
            self._decoder_cache = [None] * len(self.decoder)

        layer_memory = self._decoder_cache if self._decoder_cache is not None else [None] * len(self.decoder)

        # Split input into individual frames
        frames = z.reshape(N, T * C, H, W).chunk(T, dim=1)

        work_queue = [(frame, 0) for frame in frames]
        output_frames = []

        while work_queue:
            frame_data, layer_idx = work_queue.pop(0)

            if layer_idx >= len(self.decoder):
                output_frames.append(frame_data)
                continue

            layer = self.decoder[layer_idx]

            if isinstance(layer, MemBlock):
                if layer_memory[layer_idx] is None:
                    output = layer(frame_data, frame_data * 0)
                    layer_memory[layer_idx] = frame_data
                else:
                    output = layer(frame_data, layer_memory[layer_idx])
                    layer_memory[layer_idx].copy_(frame_data)

                work_queue.insert(0, (output, layer_idx + 1))

            elif isinstance(layer, TGrow):
                output = layer(frame_data)
                NT, C, H, W = output.shape
                N = frame_data.shape[0]
                # Split expanded frames (reverse for correct ordering with depth-first)
                for frame_slice in reversed(output.view(N, layer.stride * C, H, W).chunk(layer.stride, dim=1)):
                    work_queue.insert(0, (frame_slice, layer_idx + 1))

            else:
                output = layer(frame_data)
                work_queue.insert(0, (output, layer_idx + 1))

        # Save cache state if not ending
        if not end:
            self._decoder_cache = layer_memory
        else:
            self.clean_decoder_cache()

        if output_frames:
            result = torch.stack(output_frames, dim=1)

            # Only trim initial frames from first chunk
            if start and self.frames_to_pad_or_trim > 0:
                frames_to_trim = min(self.frames_to_pad_or_trim, result.shape[1])
                result = result[:, frames_to_trim:]

            return result
        else:
            raise ValueError("No output frames produced")
