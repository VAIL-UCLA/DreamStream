"""Causal Wan 2.2 ti2v backbone.

Subclass of :class:`Wan22Model` that swaps the bidirectional attention blocks
for the KV-cached causal attention blocks. The model_type is locked to
``ti2v``: Wan 2.2 ships only the ti2v variant (``z_dim=48``, spatial
down-factor 16, max latent frames 31). Only streaming inference (with a
KV cache) is supported.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from stream_package.models.causal_wan_model import (
    CausalHead,
    CausalWanAttentionBlock,
)
from stream_package.models.wan2_2_wrapper import Wan22Model
from stream_package.models.wan_base.modules.model import (
    rope_params,
    sinusoidal_embedding_1d,
)


class CausalWan22Model(Wan22Model):
    """Causal Wan 2.2 ti2v diffusion backbone."""

    _no_split_modules = ["CausalWanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=48,
        dim=3072,
        ffn_dim=14336,
        freq_dim=256,
        text_dim=4096,
        out_dim=48,
        num_heads=24,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        add_ref_conv=False,
        in_dim_ref_conv=48,
        model_type=None,
    ):
        if model_type is not None and model_type != "ti2v":
            raise ValueError(
                f"CausalWan22Model is locked to model_type='ti2v', got {model_type!r}"
            )
        super().__init__(
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            model_type="ti2v",
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )

        cross_attn_type = "t2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps
                )
                for _ in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        self._init_causal_weights()

    def _init_causal_weights(self) -> None:
        for m in self.blocks.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.head.head, nn.Linear):
            nn.init.zeros_(self.head.head.weight)
            if self.head.head.bias is not None:
                nn.init.zeros_(self.head.head.bias)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        current_end: int = 0,
        streaming_kwargs: dict = None,
    ):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        num_frames = x[0].shape[1]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)

        if t.dim() == 1:
            t = t.expand(t.size(0), num_frames)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        for block_index, block in enumerate(self.blocks):
            block_kwargs = {
                **kwargs,
                "kv_cache": kv_cache[block_index],
                "crossattn_cache": crossattn_cache[block_index] if crossattn_cache is not None else None,
                "current_start": current_start,
                "current_end": current_end,
                "streaming_kwargs": streaming_kwargs,
            }
            x = block(x, **block_kwargs)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(self, *args, **kwargs):
        clip_fea = kwargs.pop("clip_fea", None)
        if clip_fea is not None:
            raise ValueError("CausalWan22Model is ti2v-only; clip_fea is not supported.")

        if kwargs.get("kv_cache", None) is None:
            raise ValueError(
                "CausalWan22Model only supports streaming inference (kv_cache must be provided)."
            )
        kwargs.pop("num_frame_per_block", None)
        kwargs.pop("sink_size", None)
        kwargs.pop("history_cutoff", None)
        return self._forward_inference(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out
