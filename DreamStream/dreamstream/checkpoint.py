"""Checkpoint resolution with automatic download from the Hugging Face Hub.

Local paths are used as-is. Anything else is downloaded with
``huggingface_hub`` into its standard cache (override with ``HF_HOME``).
Authentication uses ``HF_TOKEN`` or the token saved by ``hf auth login``.
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "UCLA-VAIL/DreamStream"
DEFAULT_DIFFUSION_CKPT = "video_model/wan2.2-5b-nuplan-train-v1/ema_model.pt"
DEFAULT_WAN_REPO = "alibaba-pai/Wan2.2-Fun-5B-Control"
# Tiny autoencoder for the Wan2.2 latent space (taew2_2 from madebyollin/taehv).
DEFAULT_TVAE_REPO = "lightx2v/Autoencoders"
DEFAULT_TVAE_FILE = "taew2_2.safetensors"


def resolve_wan_root(wan_root: str | None = None, wan_repo: str = DEFAULT_WAN_REPO) -> str:
    """Return a local Wan2.2-Fun-5B-Control directory, downloading it if needed."""
    if wan_root:
        return wan_root
    from huggingface_hub import snapshot_download

    logger.info("Resolving base model %s from the Hugging Face Hub ...", wan_repo)
    return snapshot_download(repo_id=wan_repo)


def fetch_repo_config(hf_repo: str = DEFAULT_HF_REPO) -> None:
    """Fetch the repo's small ``config.json`` (the file the Hub counts downloads by)."""
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id=hf_repo, filename="config.json")
    except Exception as e:  # never block startup on this
        logger.debug("Could not fetch config.json from %s: %s", hf_repo, e)


def resolve_file(path: str, hf_repo: str = DEFAULT_HF_REPO) -> str:
    """Return *path* if it is a local file, else download it from *hf_repo*.

    Non-local values are treated as a path inside the Hub repo, e.g.
    ``video_model/wan2.2-5b-nuplan-train-v1/ema_model.pt``.
    """
    if os.path.isfile(path):
        return path
    from huggingface_hub import hf_hub_download

    logger.info("Resolving %s from %s on the Hugging Face Hub ...", path, hf_repo)
    return hf_hub_download(repo_id=hf_repo, filename=path)
