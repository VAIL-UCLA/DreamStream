"""Causal Wan 2.2 ti2v diffusion wrapper.

The input channels and ``ref_conv`` presence are detected from the DreamStream
checkpoint, so it loads against the base model's ``config.json``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin

from stream_package.models.causal_wan22_model import CausalWan22Model
from stream_package.models.causal_wan_wrapper import CausalWanDiffusionWrapper

logger = logging.getLogger(__name__)


class CausalWan22DiffusionWrapper(CausalWanDiffusionWrapper):
    """Diffusion wrapper around :class:`CausalWan22Model` (ti2v only)."""

    def __init__(self, model):
        ModelMixin.__init__(self)
        self.model = model
        self.frame_seq_len = None

    @classmethod
    def from_pretrained(cls, wan_root: str, **kwargs):
        diffusion_ckpt = kwargs.pop("diffusion_ckpt", None)
        new_in_dim = kwargs.pop("new_in_dim", None)
        add_ref_conv_override = kwargs.pop("add_ref_conv", None)

        # Read new_in_dim / add_ref_conv from the checkpoint unless given.
        ckpt_state = None
        if diffusion_ckpt is not None:
            # mmap avoids holding a second copy of the checkpoint in RAM.
            ckpt_state = torch.load(diffusion_ckpt, map_location="cpu", weights_only=True, mmap=True)
            if new_in_dim is None and "patch_embedding.weight" in ckpt_state:
                new_in_dim = ckpt_state["patch_embedding.weight"].shape[1]
            if add_ref_conv_override is None:
                add_ref_conv_override = any(k.startswith("ref_conv.") for k in ckpt_state)

        model = CausalWan22Model.from_pretrained_wan22(wan_root)

        if new_in_dim is not None and new_in_dim != model.in_dim:
            with torch.no_grad():
                new_patch_embedding = nn.Conv3d(
                    new_in_dim, model.dim,
                    kernel_size=model.patch_size, stride=model.patch_size,
                )
                nn.init.xavier_uniform_(new_patch_embedding.weight.flatten(1))
                if new_in_dim > model.in_dim:
                    new_patch_embedding.weight[:, : model.in_dim, :, :, :].copy_(
                        model.patch_embedding.weight
                    )
                    if new_in_dim == model.in_dim * 2:
                        new_patch_embedding.weight[:, model.in_dim :, :, :, :].copy_(
                            model.patch_embedding.weight
                        )
                else:
                    new_patch_embedding.weight.copy_(
                        model.patch_embedding.weight[:, :new_in_dim, :, :, :]
                    )
                if model.patch_embedding.bias is not None:
                    new_patch_embedding.bias.copy_(model.patch_embedding.bias)
                model.patch_embedding = new_patch_embedding
                model.in_dim = new_in_dim

        if add_ref_conv_override is False and model.ref_conv is not None:
            model.ref_conv = None
            model.add_ref_conv = False
        elif add_ref_conv_override is True and model.ref_conv is None:
            raise ValueError(
                "add_ref_conv=True override is not supported when wan_root "
                "config.json has add_ref_conv=False; load from a wan_root "
                "that ships ref_conv weights instead."
            )

        model.eval().requires_grad_(False)
        if ckpt_state is not None:
            logger.info(f"Loading Wan2.2 diffusion overlay: {diffusion_ckpt}")
            model.load_state_dict(ckpt_state, strict=False)

        return cls(model)
