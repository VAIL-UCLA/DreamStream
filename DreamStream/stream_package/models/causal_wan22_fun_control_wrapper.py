"""Base class for the streaming Wan 2.2 ti2v FunControl pipeline."""

from __future__ import annotations

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from stream_package.models.causal_wan22_wrapper import CausalWan22DiffusionWrapper
from stream_package.models.wan_wrapper import WanTextEncoderWrapper, WanVAEWrapper


class CausalWan22FunControlInferencePipeline(ModelMixin, ConfigMixin):
    """Wan 2.2 ti2v FunControl pipeline: text encoder, VAE and causal diffusion model."""

    def __init__(
        self,
        text_encoder,
        vae,
        diffusion,
        denoising_step_list=[1000, 927, 771, 0],
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.diffusion = diffusion
        self.denoising_step_list = denoising_step_list
        self.kv_cache = None
        self.crossattn_cache = None

    @classmethod
    def from_pretrained(cls, wan_root: str, **kwargs):
        diffusion = CausalWan22DiffusionWrapper.from_pretrained(wan_root, **kwargs)
        use_tvae_decoder = kwargs.pop("use_tvae_decoder", False)
        use_tvae_encoder = kwargs.pop("use_tvae_encoder", False)
        tvae_ckpt = kwargs.pop("tvae_ckpt", None)
        vae = WanVAEWrapper.from_pretrained(
            wan_root, use_tvae_decoder=use_tvae_decoder,
            use_tvae_encoder=use_tvae_encoder, tvae_ckpt=tvae_ckpt,
        )
        text_encoder = WanTextEncoderWrapper.from_pretrained(wan_root)
        denoising_step_list = kwargs.get("denoising_step_list", [1000, 927, 771, 0])
        return cls(text_encoder, vae, diffusion, denoising_step_list)

    def calc_block_frames(self, num_blocks, num_frame_per_block, start_block_idx=0):
        if isinstance(num_frame_per_block, int):
            return num_blocks * num_frame_per_block
        first_size, subsequent_size = num_frame_per_block
        frames = 0
        for b in range(num_blocks):
            frames += first_size if start_block_idx + b == 0 else subsequent_size
        return frames

    def reset_all_caches(self):
        if self.kv_cache is None:
            return
        del self.kv_cache
        del self.crossattn_cache
        self.kv_cache = None
        self.crossattn_cache = None
        if hasattr(self, "cache"):
            del self.cache
        torch.cuda.empty_cache()
