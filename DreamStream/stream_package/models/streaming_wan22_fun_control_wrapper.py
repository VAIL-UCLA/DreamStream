"""Streaming Wan 2.2 ti2v FunControl pipeline.

Streaming rollout with one KV cache per non-zero denoising step.
"""

from __future__ import annotations

import math
import random
import sys

import torch

from stream_package.models.causal_wan22_fun_control_wrapper import (
    CausalWan22FunControlInferencePipeline,
)
from stream_package.utils.flow_match import FlowMatchScheduler
from stream_package.utils.inference_utils import (
    GPUTimer,
    initialize_img_crossattn_cache,
    initialize_kv_cache,
    prepare_inputs_fun,
)


_LATENT_CHANNELS = 48
_NUM_FRAMES = 31
_SPATIAL_DOWN = 16


class StreamingWan22FunControlInferencePipeline(CausalWan22FunControlInferencePipeline):
    """Streaming inference for Wan2.2 ti2v fun-control with per-step KV caches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache_per_step = None  # list[K] of layer-keyed KV caches

    def _roll_streaming_caches(self, sink_tokens: int, roll_tokens: int) -> None:
        for cache in self.kv_cache_per_step:
            for layer_idx in range(len(cache)):
                cache[layer_idx]["k"][:, sink_tokens:] = torch.roll(
                    cache[layer_idx]["k"][:, sink_tokens:],
                    shifts=-roll_tokens, dims=1,
                )
                cache[layer_idx]["v"][:, sink_tokens:] = torch.roll(
                    cache[layer_idx]["v"][:, sink_tokens:],
                    shifts=-roll_tokens, dims=1,
                )
                # Rolled keys sit in new slots, so their RoPE must be recomputed.
                cache[layer_idx]["k_roped_stale_from"] = sink_tokens

    def reset_all_caches(self):
        if self.kv_cache_per_step is not None:
            del self.kv_cache_per_step
            self.kv_cache_per_step = None
        super().reset_all_caches()

    def streaming_generation(
        self,
        stream_idx=0,
        input_prompt=None,
        control_video=None,
        ref_image=None,
        size=(832, 480),
        shift=6.0,
        denoising_step_list=None,
        num_frame_per_block=3,
        sink_size=0,
        history_cutoff=999,
        rope_range="limited_naive",
        seed=-1,
        enable_timing=False,
        use_precalculated_contexts=False,
        **kwargs,
    ):
        if stream_idx == 0:
            self.timer = GPUTimer(enabled=enable_timing)

            class Cache:
                pass

            self.cache = Cache()
            _, ref_image = prepare_inputs_fun(
                control_video=None,
                ref_image=ref_image,
                size=size,
                device=self.diffusion.device,
            )

            self.cache.th = size[1] // _SPATIAL_DOWN
            self.cache.tw = size[0] // _SPATIAL_DOWN
            self.cache.num_frames = _NUM_FRAMES
            self.cache.latent_channels = _LATENT_CHANNELS

            expected_control_channels = self.diffusion.model.in_dim - self.cache.latent_channels
            if expected_control_channels != self.cache.latent_channels:
                raise ValueError(
                    "streaming_generation supports the noise+control TI2V layout only: "
                    f"model.in_dim={self.diffusion.model.in_dim}, latent_channels={self.cache.latent_channels}."
                )

            self.cache.frame_seq_len = math.ceil((self.cache.th * self.cache.tw) / (2 * 2))
            self.diffusion.configure_seq_len(size=size)
            sink_frames = min(self.calc_block_frames(sink_size, num_frame_per_block), self.cache.num_frames)
            self.cache.streaming_kwargs = {
                "rope_range": rope_range,
                "sink_frames": sink_frames,
                "streaming_time_idx": None,
            }
            self.cache.sink_frames = sink_frames
            self.cache.sink_tokens = self.cache.sink_frames * self.cache.frame_seq_len
            self.cache.roll_frames = self.calc_block_frames(1, num_frame_per_block, start_block_idx=sink_size)
            self.cache.roll_tokens = self.cache.roll_frames * self.cache.frame_seq_len
            if history_cutoff >= 999:
                raise ValueError("history_cutoff cannot be 999 when streaming long video")
            self.cache.history_frames = self.calc_block_frames(
                history_cutoff, num_frame_per_block, start_block_idx=sink_size
            )
            self.cache.history_tokens = self.cache.history_frames * self.cache.frame_seq_len

            seq_len = self.cache.frame_seq_len * self.calc_block_frames(
                history_cutoff + sink_size + 1, num_frame_per_block
            )

            if denoising_step_list is None:
                self.cache.denoising_steps = self.denoising_step_list
            else:
                self.cache.denoising_steps = denoising_step_list
            assert self.cache.denoising_steps[-1] == 0
            self.cache.trajectory_steps = list(self.cache.denoising_steps)
            self.cache.step_timesteps = self.cache.trajectory_steps[:-1]
            num_steps = len(self.cache.step_timesteps)

            self.kv_cache_per_step = [
                initialize_kv_cache(
                    batch_size=1,
                    seq_len=seq_len,
                    num_layers=self.diffusion.model.num_layers,
                    num_heads=self.diffusion.model.num_heads,
                    head_dim=self.diffusion.model.dim // self.diffusion.model.num_heads,
                    dtype=torch.bfloat16,
                    device=self.diffusion.device,
                )
                for _ in range(num_steps)
            ]
            self.crossattn_cache = initialize_img_crossattn_cache(
                batch_size=1,
                seq_len=self.diffusion.model.text_len,
                num_layers=self.diffusion.model.num_layers,
                num_heads=self.diffusion.model.num_heads,
                head_dim=self.diffusion.model.dim // self.diffusion.model.num_heads,
                dtype=torch.bfloat16,
                device=self.diffusion.device,
            )

            self.cache.scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
            self.cache.scheduler.set_timesteps(1000, training=True)

            seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
            self.cache.seed_g = torch.Generator(device=self.diffusion.device).manual_seed(seed)

            self.cache.current_start = 0
            self.cache.current_end = 0

            with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
                with self.timer("text_encode"):
                    if use_precalculated_contexts:
                        self.cache.context = self.precomputed_text_emb
                    else:
                        self.cache.context = self.text_encoder([input_prompt])

                if ref_image is not None:
                    with self.timer("vae_encode"):
                        self.cache.z = self.vae.encode(ref_image.unsqueeze(1))
                else:
                    self.cache.z = None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            current_block_size = num_frame_per_block
            target_shape = (1, current_block_size, self.cache.latent_channels, self.cache.th, self.cache.tw)

            self.cache.current_start = self.cache.current_end
            self.cache.current_end = self.cache.current_end + self.cache.frame_seq_len * current_block_size

            if stream_idx > sink_size + history_cutoff:
                self._roll_streaming_caches(self.cache.sink_tokens, self.cache.roll_tokens)
                self.cache.current_start = self.cache.sink_tokens + self.cache.history_tokens
                self.cache.current_end = self.cache.current_start + current_block_size * self.cache.frame_seq_len

            with self.timer("control_process"):
                control_video_block, _ = prepare_inputs_fun(
                    control_video=control_video,
                    ref_image=None,
                    size=size,
                    device=self.diffusion.device,
                )
                control_emb = self.vae.encode(control_video_block)

            noise = torch.randn(
                target_shape, dtype=torch.bfloat16, device=self.diffusion.device, generator=self.cache.seed_g
            )

            control_latents = control_emb
            num_steps = len(self.cache.step_timesteps)

            with self.timer("denoising"):
                x = noise
                for j in range(num_steps):
                    t_j = self.cache.step_timesteps[j]
                    timestep = torch.ones(
                        [1, current_block_size], device=noise.device, dtype=torch.int64
                    ) * t_j
                    pred_x = x
                    if stream_idx == 0 and self.cache.z is not None:
                        timestep = timestep.clone()
                        timestep[:, :1] = 0
                        pred_x = pred_x.clone()
                        pred_x[:, :1] = self.cache.z

                    denoised_pred = self.diffusion(
                        x=pred_x,
                        t=timestep,
                        context=self.cache.context,
                        scheduler=self.cache.scheduler,
                        kv_cache=self.kv_cache_per_step[j],
                        crossattn_cache=self.crossattn_cache,
                        current_start=self.cache.current_start,
                        current_end=self.cache.current_end,
                        y=control_latents,
                        streaming_kwargs=self.cache.streaming_kwargs,
                    )

                    is_last_step = j == num_steps - 1
                    if not is_last_step:
                        t_next = self.cache.trajectory_steps[j + 1]
                        x_next = self.cache.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn(
                                denoised_pred.flatten(0, 1).shape,
                                device=self.diffusion.device,
                                dtype=denoised_pred.dtype,
                                generator=self.cache.seed_g,
                            ),
                            t_next * torch.ones([1], device=noise.device, dtype=torch.long),
                        ).unflatten(0, denoised_pred.shape[:2])
                        commit_t_val = t_next
                        commit_x = x_next
                    else:
                        x_next = denoised_pred
                        commit_t_val = 0
                        commit_x = x_next
                    x = x_next

                    commit_t = torch.ones(
                        [1, current_block_size], device=noise.device, dtype=torch.int64
                    ) * commit_t_val
                    if stream_idx == 0 and self.cache.z is not None:
                        commit_t = commit_t.clone()
                        commit_t[:, :1] = 0
                        commit_x = commit_x.clone()
                        commit_x[:, :1] = self.cache.z

                    self.diffusion(
                        x=commit_x,
                        t=commit_t,
                        context=self.cache.context,
                        scheduler=self.cache.scheduler,
                        kv_cache=self.kv_cache_per_step[j],
                        crossattn_cache=self.crossattn_cache,
                        current_start=self.cache.current_start,
                        current_end=self.cache.current_end,
                        y=control_latents,
                        streaming_kwargs=self.cache.streaming_kwargs,
                    )

            with self.timer("vae_decode"):
                videos = self.vae.decode(denoised_pred, start=(stream_idx == 0), end=False)

        self.timer.report(prefix=f"[Stream {stream_idx}] ")
        self.timer.reset()

        return videos[0]
