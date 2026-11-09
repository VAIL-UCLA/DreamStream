"""StreamingPipeline: programmatic API for streaming video generation."""

import json
import os

import torch

from stream_package.models.streaming_wan22_fun_control_wrapper import StreamingWan22FunControlInferencePipeline
from stream_package.utils.inference_utils import prepare_inputs_fun


class StreamingPipeline:
    """Programmatic API for streaming video generation with control conditioning."""

    def __init__(self, pipeline: StreamingWan22FunControlInferencePipeline, device: str = None):
        self.pipeline = pipeline
        self._device = torch.device(device) if device else next(pipeline.diffusion.parameters()).device
        self._precomputed_prompt = None

    @classmethod
    def from_pretrained(
        cls,
        wan_root: str,
        diffusion_ckpt: str = None,
        denoising_step_list: list = None,
        use_tvae_decoder: bool = False,
        use_tvae_encoder: bool = False,
        tvae_ckpt: str = None,
        device: str = "cuda:0",
    ):
        """Load the streaming Wan2.2 ti2v FunControl pipeline from local weights.

        Args:
            wan_root: Path to the Wan2.2 ti2v base model directory.
            diffusion_ckpt: Path to the diffusion checkpoint (plain state_dict).
            denoising_step_list: List of denoising timesteps.
            use_tvae_decoder: Replace VAE decoder with TinyVAE decoder.
            use_tvae_encoder: Replace VAE encoder with TinyVAE encoder.
            tvae_ckpt: Path to TinyVAE checkpoint.
            device: Target device.
        """
        with open(os.path.join(wan_root, "config.json")) as f:
            model_type = json.load(f).get("model_type")
        if model_type != "ti2v":
            raise ValueError(
                f"{wan_root}/config.json has model_type={model_type!r}; only Wan2.2 'ti2v' is supported."
            )

        pipe_kwargs = {}
        if denoising_step_list is not None:
            pipe_kwargs["denoising_step_list"] = denoising_step_list
        if diffusion_ckpt:
            pipe_kwargs["diffusion_ckpt"] = diffusion_ckpt
        if use_tvae_decoder:
            pipe_kwargs["use_tvae_decoder"] = True
            pipe_kwargs["tvae_ckpt"] = tvae_ckpt
        if use_tvae_encoder:
            pipe_kwargs["use_tvae_encoder"] = True
            if not use_tvae_decoder:
                pipe_kwargs["tvae_ckpt"] = tvae_ckpt

        pipeline = StreamingWan22FunControlInferencePipeline.from_pretrained(wan_root, **pipe_kwargs)

        pipeline.diffusion.to(device)
        pipeline.vae.to(device)
        pipeline.text_encoder.to(device)

        return cls(pipeline, device=str(device))

    def streaming_step(
        self,
        stream_idx: int,
        prompt: str,
        control_block,
        ref_image=None,
        size: tuple = (832, 480),
        **kwargs,
    ):
        """Single streaming block generation. Returns video tensor for this block."""
        return self.pipeline.streaming_generation(
            stream_idx=stream_idx,
            input_prompt=prompt,
            control_video=control_block,
            ref_image=ref_image,
            size=size,
            **kwargs,
        )

    @torch.no_grad()
    def warmup(
        self,
        prompt: str,
        ref_image,
        control_block,
        size: tuple = (832, 480),
        num_dummy_blocks: int = 2,
        **kwargs,
    ):
        """Warm up: pre-compute embeddings and trigger torch.compile tracing.

        Runs dummy streaming blocks to warm up torch.compile graphs and CUDA
        allocator, then resets streaming state. After warmup, the real session
        starts with pre-computed embeddings and warm compile caches.

        Args:
            prompt: Text prompt for embedding precomputation.
            ref_image: Reference image (PIL, path, or tensor).
            control_block: A control video block (used for dummy generation).
            size: (width, height) tuple.
            num_dummy_blocks: Number of dummy streaming blocks to run (1-3).
        """
        print(f"[warmup] Pre-computing embeddings on {self._device}...")

        # Prepare ref_image tensor
        _, ref_image_tensor = prepare_inputs_fun(
            control_video=None, ref_image=ref_image, size=size, device=self._device,
        )

        self.set_prompt(prompt)

        print(f"[warmup] Running {num_dummy_blocks + 1} dummy blocks (block 0 + {num_dummy_blocks} streaming)...")

        # Run dummy block 0 + streaming blocks to trigger compile tracing
        stream_kwargs = dict(
            size=size,
            use_precalculated_contexts=True,
            **kwargs,
        )

        for i in range(1 + num_dummy_blocks):
            self.pipeline.streaming_generation(
                stream_idx=i,
                input_prompt=prompt,
                control_video=control_block,
                ref_image=ref_image_tensor if i == 0 else None,
                **stream_kwargs,
            )

        # 4. Reset streaming state (keeps compiled graphs in PyTorch's cache)
        self.pipeline.reset_all_caches()

        print("[warmup] Done. Compiled graphs cached, embeddings pre-computed.")

    @torch.no_grad()
    def set_prompt(self, prompt: str):
        """Pre-compute the text embedding used with ``use_precalculated_contexts=True``.

        No-op if *prompt* is already the pre-computed one.
        """
        if prompt == self._precomputed_prompt:
            return
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            self.pipeline.precomputed_text_emb = self.pipeline.text_encoder([prompt])
        self._precomputed_prompt = prompt

    def reset(self):
        """Reset caches between generation sessions."""
        self.pipeline.reset_all_caches()
