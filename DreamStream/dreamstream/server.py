"""HTTP batch inference server for DreamStream.

Lock-step block inference: send N traffic-layout frames for one block, get N
rendered frames back, repeat.

Endpoints:
    GET  /info   — backend metadata + frame count requirements
    POST /infer  — send layout frames, receive rendered frames
    POST /reset  — reset pipeline state (start a new session)

Run with ``python -m dreamstream.server`` (see ``--help``).
"""

import argparse
import base64
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from dreamstream.backend import DiffusionBackend
from dreamstream.checkpoint import (
    DEFAULT_DIFFUSION_CKPT,
    DEFAULT_HF_REPO,
    DEFAULT_TVAE_FILE,
    DEFAULT_TVAE_REPO,
    DEFAULT_WAN_REPO,
    fetch_repo_config,
    resolve_file,
    resolve_wan_root,
)

log = logging.getLogger(__name__)

# PIL releases the GIL while encoding, so JPEGs encode in parallel.
_jpeg_pool = ThreadPoolExecutor(max_workers=8)


def _encode_jpeg_b64(frame: np.ndarray, quality: int) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class _BatchState:
    """Mutable server state for block-based inference."""

    def __init__(
        self,
        backend: DiffusionBackend,
        block_0_input_frames: int,
        block_input_frames: int,
        default_prompt: str = "",
    ):
        self.backend = backend
        self.block_0_input_frames = block_0_input_frames
        self.block_input_frames = block_input_frames
        self.default_prompt = default_prompt
        self.block_idx = 0
        self.initialized = False


def create_app(
    backend: DiffusionBackend,
    block_0_input_frames: int,
    block_input_frames: int,
    default_prompt: str = "",
    jpeg_quality: int = 85,
) -> FastAPI:
    """Create a FastAPI app wrapping *backend* for batch inference."""

    state = _BatchState(
        backend, block_0_input_frames, block_input_frames, default_prompt,
    )

    app = FastAPI(title="DreamStream Batch API")

    @app.get("/info")
    def info():
        return {
            **state.backend.get_info(),
            "block_0_input_frames": state.block_0_input_frames,
            "block_input_frames": state.block_input_frames,
            "default_prompt": state.default_prompt,
        }

    @app.post("/infer")
    async def infer(
        frames: list[UploadFile] = File(...),
        prompt: Optional[str] = Form(None),
        first_frame: Optional[UploadFile] = File(None),
    ):
        # Each frame is an .npy-encoded (H, W, 3) uint8 layout image
        layouts: list[np.ndarray] = []
        for f in frames:
            layouts.append(np.load(io.BytesIO(await f.read())))

        expected = (
            state.block_0_input_frames if state.block_idx == 0
            else state.block_input_frames
        )
        if len(layouts) != expected:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Block {state.block_idx} requires exactly {expected} "
                    f"frames, got {len(layouts)}"
                ),
            )

        # Auto-initialize on first /infer call
        if not state.initialized:
            state.backend.begin_stream(
                prompt=prompt or state.default_prompt or None,
                first_frame=await first_frame.read() if first_frame is not None else None,
            )
            state.initialized = True

        try:
            results = state.backend.stream_block(layouts)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        encoded_frames = list(_jpeg_pool.map(lambda f: _encode_jpeg_b64(f, jpeg_quality), results))

        current_block = state.block_idx
        state.block_idx += 1

        return {
            "frames": encoded_frames,
            "block_idx": current_block,
            "num_output_frames": len(results),
        }

    @app.post("/reset")
    def reset():
        state.block_idx = 0
        state.initialized = False
        return {"status": "ok"}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="DreamStream HTTP Batch Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--prompt", default="a sunny highway",
                        help="Prompt used when the client sends none (also used for warmup)")
    parser.add_argument("--jpeg-quality", type=int, default=85)

    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                        help="Hugging Face repo holding the DreamStream checkpoint")
    parser.add_argument("--diffusion-ckpt", default=DEFAULT_DIFFUSION_CKPT,
                        help="Local path, or path inside --hf-repo")
    parser.add_argument("--wan-repo", default=DEFAULT_WAN_REPO,
                        help="Hugging Face repo of the Wan2.2-Fun-5B-Control base model")
    parser.add_argument("--wan-root", default=None,
                        help="Local Wan2.2-Fun-5B-Control directory (skips download)")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--full-vae-decoder", action="store_true",
                        help="Decode with the full Wan2.2 VAE instead of TinyVAE (sharper, slower, more memory)")
    parser.add_argument("--tvae-encoder", action="store_true",
                        help="Experimental: also encode layouts with TinyVAE")
    parser.add_argument("--tvae-ckpt", default=None,
                        help=f"TinyVAE checkpoint: local path, or path inside --hf-repo "
                             f"(default: {DEFAULT_TVAE_FILE} from {DEFAULT_TVAE_REPO})")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup blocks at startup")
    parser.add_argument("--profile", action="store_true",
                        help="Print a per-stage GPU timing breakdown for every block")

    parser.add_argument("--size", default="1280x704", help="WIDTHxHEIGHT")
    parser.add_argument("--num-frame-per-block", type=int, default=3,
                        help="Latent frames per block")
    parser.add_argument("--shift", type=float, default=6.0)
    parser.add_argument("--sink-size", type=int, default=1)
    parser.add_argument("--history-cutoff", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="-1 for random")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    use_tvae_decoder = not args.full_vae_decoder
    use_tvae = use_tvae_decoder or args.tvae_encoder
    tvae_ckpt = None
    if use_tvae:
        tvae_ckpt = (
            resolve_file(args.tvae_ckpt, args.hf_repo) if args.tvae_ckpt
            else resolve_file(DEFAULT_TVAE_FILE, DEFAULT_TVAE_REPO)
        )

    if not os.path.isfile(args.diffusion_ckpt):
        fetch_repo_config(args.hf_repo)

    w, h = args.size.split("x")
    backend = DiffusionBackend(
        wan_root=resolve_wan_root(args.wan_root, args.wan_repo),
        diffusion_ckpt=resolve_file(args.diffusion_ckpt, args.hf_repo),
        device=args.device,
        prompt=args.prompt,
        size=(int(w), int(h)),
        num_frame_per_block=args.num_frame_per_block,
        shift=args.shift,
        sink_size=args.sink_size,
        history_cutoff=args.history_cutoff,
        use_tvae_decoder=use_tvae_decoder,
        use_tvae_encoder=args.tvae_encoder,
        tvae_ckpt=tvae_ckpt,
        compile=not args.no_compile,
        warmup=not args.no_warmup,
        seed=args.seed,
        profile=args.profile,
    )
    backend.setup()

    # Wan VAE temporal compression is 4x; block 0 also carries the first frame.
    nfpb = args.num_frame_per_block
    block_0, block_n = 4 * (nfpb - 1) + 1, 4 * nfpb

    app = create_app(
        backend,
        block_0_input_frames=block_0,
        block_input_frames=block_n,
        default_prompt=args.prompt,
        jpeg_quality=args.jpeg_quality,
    )

    log.info("Starting batch server on %s:%d", args.host, args.port)
    log.info("Block 0: %d input frames, Blocks 1+: %d input frames", block_0, block_n)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
