"""Wan 2.2 model wrapper for Fun-5B-Control with ref_conv support.

Wan2.2-Fun-5B-Control uses in_dim=148 (noise 48 + control_video 48 + mask 4 + ref_padded 48)
and an additional ref_conv (Conv2d) that projects a reference image latent into the
transformer token space. This module provides Wan22Model, the base of
CausalWan22Model.
"""

import json
import logging
import os

import torch
import torch.nn as nn

from stream_package.models.wan_base.modules.model import WanModel

logger = logging.getLogger(__name__)


class Wan22Model(WanModel):
    """Wan 2.2 model variant with optional ref_conv for Fun-Control.

    Adds a Conv2d ref_conv layer that projects a reference image latent (single frame)
    into transformer token space and prepends it to the main sequence, providing an
    additional conditioning pathway beyond the channel-concatenated y input.
    """

    def __init__(self, add_ref_conv=False, in_dim_ref_conv=48, **kwargs):
        super().__init__(**kwargs)
        self.add_ref_conv = add_ref_conv
        self.in_dim_ref_conv = in_dim_ref_conv
        if add_ref_conv:
            self.ref_conv = nn.Conv2d(
                in_dim_ref_conv, self.dim,
                kernel_size=self.patch_size[1:],
                stride=self.patch_size[1:],
            )
            nn.init.xavier_uniform_(self.ref_conv.weight.flatten(1))
            if self.ref_conv.bias is not None:
                nn.init.zeros_(self.ref_conv.bias)
        else:
            self.ref_conv = None

    @classmethod
    def from_pretrained_wan22(cls, wan_root, **kwargs):
        """Load from Wan2.2-Fun-5B-Control checkpoint directory.

        Reads the checkpoint config.json to determine architecture params
        (including add_ref_conv, in_dim_ref_conv), constructs the model,
        and loads weights from safetensors or .pth files.
        """
        config_path = os.path.join(wan_root, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        wan_params = {}
        for key in [
            "model_type", "patch_size", "text_len", "in_dim", "dim",
            "ffn_dim", "freq_dim", "text_dim", "out_dim", "num_heads",
            "num_layers", "window_size", "qk_norm", "cross_attn_norm", "eps",
        ]:
            if key in config:
                wan_params[key] = config[key]

        add_ref_conv = config.get("add_ref_conv", False)
        in_dim_ref_conv = config.get("in_dim_ref_conv", 48)

        model = cls(
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            **wan_params,
        )

        index_path = os.path.join(wan_root, "diffusion_pytorch_model.safetensors.index.json")
        single_st = os.path.join(wan_root, "diffusion_pytorch_model.safetensors")

        if os.path.exists(index_path):
            from safetensors.torch import load_file
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for shard in shard_files:
                state_dict.update(load_file(os.path.join(wan_root, shard)))
        elif os.path.exists(single_st):
            from safetensors.torch import load_file
            state_dict = load_file(single_st)
        else:
            pth_candidates = sorted(
                f for f in os.listdir(wan_root) if f.endswith(".pth") and "diffusion" in f.lower()
            )
            if pth_candidates:
                state_dict = torch.load(
                    os.path.join(wan_root, pth_candidates[0]), map_location="cpu", weights_only=False
                )
            else:
                raise FileNotFoundError(f"No diffusion weight files found in {wan_root}")

        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys when loading Wan22Model: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(
                f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:10]}..."
            )

        return model
