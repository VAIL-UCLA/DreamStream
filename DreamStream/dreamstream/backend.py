"""Diffusion backend wrapping stream_package.StreamingPipeline.

Block-based streaming: each block takes a fixed number of traffic-layout
frames (perspective-projected HD map + 3D boxes) and returns the same number
of rendered RGB frames, continuing from the pipeline's KV cache.
"""

import io
import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _layout_to_control_tensor(layout: np.ndarray, device: str, size: Tuple[int, int]):
    """Convert a uint8 layout image to a control video tensor.

    Args:
        layout: (H, W, 3), (H, W, 1) or (H, W) uint8 array.
        device: Target torch device string.
        size: (width, height) for resizing.

    Returns:
        torch.Tensor of shape (1, 1, 3, H, W), bfloat16, values in [-1, 1].
    """
    import torch

    if layout.dtype != np.uint8:
        raise ValueError(f"Layout frames must be uint8 images, got dtype {layout.dtype}")
    if layout.ndim == 2:
        layout = np.stack([layout] * 3, axis=-1)
    elif layout.ndim == 3 and layout.shape[2] == 1:
        layout = np.repeat(layout, 3, axis=2)

    target_w, target_h = size
    h, w = layout.shape[:2]
    if h != target_h or w != target_w:
        from PIL import Image
        img = Image.fromarray(layout)
        img = img.resize((target_w, target_h), Image.LANCZOS)
        layout = np.array(img)

    # (H, W, C) -> (1, 1, C, H, W) tensor, normalize to [-1, 1]
    tensor = torch.from_numpy(layout).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.bfloat16)
    tensor = tensor / 127.5 - 1.0

    return tensor


def _output_tensor_to_numpy(tensor) -> List[np.ndarray]:
    """Convert pipeline output (T, C, H, W) in [-1,1] to list of (H, W, 3) uint8 arrays."""
    import torch
    tensor = tensor.float().clamp(-1.0, 1.0)
    tensor = ((tensor + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
    return [tensor[t].permute(1, 2, 0).cpu().numpy() for t in range(tensor.shape[0])]


class DiffusionBackend:
    """Runs layout-conditioned streaming video diffusion via StreamingPipeline."""

    _DEFAULT_DENOISING_STEPS = [1000, 927, 771, 0]

    def __init__(
        self,
        wan_root: str,
        diffusion_ckpt: str,
        device: str = "cuda:0",
        prompt: str = "",
        size: Tuple[int, int] = (1280, 704),
        num_frame_per_block: int = 3,
        denoising_step_list: Optional[List[int]] = None,
        shift: float = 6.0,
        sink_size: int = 1,
        history_cutoff: int = 2,
        use_tvae_decoder: bool = False,
        use_tvae_encoder: bool = False,
        tvae_ckpt: Optional[str] = None,
        compile: bool = True,
        warmup: bool = True,
        seed: int = 42,
        profile: bool = False,
    ):
        self.wan_root = wan_root
        self.diffusion_ckpt = diffusion_ckpt
        self.device = device
        self.prompt = prompt
        self.size = size
        self.num_frame_per_block = num_frame_per_block
        self.denoising_step_list = denoising_step_list or self._DEFAULT_DENOISING_STEPS
        self.shift = shift
        self.sink_size = sink_size
        self.history_cutoff = history_cutoff
        self.use_tvae_decoder = use_tvae_decoder
        self.use_tvae_encoder = use_tvae_encoder
        self.tvae_ckpt = tvae_ckpt
        self.do_compile = compile
        self.do_warmup = warmup
        self.seed = seed
        self.profile = profile
        self.pipeline = None

        self._stream_idx = 0
        self._active_prompt: str = prompt
        self._first_frame = None

    def _load_first_frame(self, image_bytes: bytes):
        """Decode an encoded image to a (1, 3, H, W) bfloat16 tensor in [-1, 1]."""
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = self.size
        img = img.resize((w, h), Image.LANCZOS)
        arr = np.array(img)  # (H, W, 3) uint8
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        tensor = tensor.to(device=self.device, dtype=torch.bfloat16)
        tensor = tensor / 127.5 - 1.0  # [0,255] -> [-1,1]
        return tensor

    def _get_ref_image(self):
        """Client-provided first frame, or a black image for neutral conditioning."""
        if self._first_frame is not None:
            return self._first_frame
        import torch
        w, h = self.size
        return torch.full(
            (1, 3, h, w), -1.0, dtype=torch.bfloat16, device=self.device,
        )

    def setup(self) -> None:
        import torch
        from stream_package import StreamingPipeline

        logger.info("Loading diffusion pipeline on %s", self.device)
        self.pipeline = StreamingPipeline.from_pretrained(
            wan_root=self.wan_root,
            diffusion_ckpt=self.diffusion_ckpt,
            denoising_step_list=self.denoising_step_list,
            use_tvae_decoder=self.use_tvae_decoder,
            use_tvae_encoder=self.use_tvae_encoder,
            tvae_ckpt=self.tvae_ckpt,
            device=self.device,
        )
        logger.info("Diffusion pipeline loaded")

        if self.do_compile:
            logger.info("Compiling diffusion model (mode=default)...")
            self.pipeline.pipeline.diffusion.model = torch.compile(
                self.pipeline.pipeline.diffusion.model, mode="default",
            )
            if not self.use_tvae_encoder:
                logger.info("Compiling VAE encoder (mode=default)...")
                self.pipeline.pipeline.vae.model.encoder = torch.compile(
                    self.pipeline.pipeline.vae.model.encoder, mode="default",
                )
            logger.info("Compilation registered (tracing happens on first run)")

        # Warmup: pre-compute text embeddings and trigger compile tracing
        if self.do_warmup:
            logger.info("Running warmup (pre-compute embeddings + trigger compile)...")
            from stream_package.utils.streaming_utils import calculate_frame_counts
            w, h = self.size
            ref_image = torch.full(
                (1, 3, h, w), -1.0, dtype=torch.bfloat16, device=self.device,
            )
            # Block 0 needs 4n+1 frames
            _, block0_frames = calculate_frame_counts(0, self.num_frame_per_block)
            dummy_control = torch.zeros(
                1, block0_frames, 3, h, w, dtype=torch.bfloat16, device=self.device,
            )
            self.pipeline.warmup(
                prompt=self.prompt,
                ref_image=ref_image,
                control_block=dummy_control,
                size=self.size,
                num_dummy_blocks=2,
                shift=self.shift,
                num_frame_per_block=self.num_frame_per_block,
                sink_size=self.sink_size,
                history_cutoff=self.history_cutoff,
                seed=self.seed if self.seed >= 0 else 42,
            )
            logger.info("Warmup complete")

    def _streaming_kwargs(self) -> dict:
        kw = dict(
            size=self.size,
            shift=self.shift,
            num_frame_per_block=self.num_frame_per_block,
            sink_size=self.sink_size,
            history_cutoff=self.history_cutoff,
        )
        if self.seed >= 0:
            kw["seed"] = self.seed
        if self.do_warmup:
            kw["use_precalculated_contexts"] = True
        if self.profile:
            kw["enable_timing"] = True
        return kw

    def block_input_frames(self) -> int:
        """Number of input layout frames needed for the current block."""
        from stream_package.utils.streaming_utils import calculate_frame_counts
        start, end = calculate_frame_counts(self._stream_idx, self.num_frame_per_block)
        return end - start

    def begin_stream(self, prompt: Optional[str] = None, first_frame: Optional[bytes] = None) -> None:
        """Start a new session: reset caches, set prompt and first frame."""
        self.pipeline.reset()
        self._stream_idx = 0
        self._active_prompt = prompt or self.prompt
        if self.do_warmup:
            # Blocks run with use_precalculated_contexts=True after warmup, so
            # the embedding must match this session's prompt.
            self.pipeline.set_prompt(self._active_prompt)
        self._first_frame = self._load_first_frame(first_frame) if first_frame is not None else None

    def stream_block(self, layouts: List[np.ndarray]) -> List[np.ndarray]:
        """Render exactly one block; returns one RGB uint8 frame per input layout."""
        import torch

        needed = self.block_input_frames()
        if len(layouts) != needed:
            raise ValueError(
                f"Block {self._stream_idx} requires exactly {needed} "
                f"frames, got {len(layouts)}"
            )

        block_idx = self._stream_idx
        control_block = torch.cat(
            [_layout_to_control_tensor(x, self.device, self.size) for x in layouts], dim=1,
        )
        ref_image = self._get_ref_image() if block_idx == 0 else None
        output = self.pipeline.streaming_step(
            stream_idx=block_idx,
            prompt=self._active_prompt,
            control_block=control_block,
            ref_image=ref_image,
            **self._streaming_kwargs(),
        )
        self._stream_idx += 1
        return _output_tensor_to_numpy(output)

    def get_info(self) -> dict:
        return {
            "model": "diffusion",
            "backend": "stream_package.StreamingPipeline",
            "device": self.device,
            "size": list(self.size),
            "num_frame_per_block": self.num_frame_per_block,
            "prompt": self.prompt,
        }
