# DreamStream — Video Generation Model

Inference code for the DreamStream video generation model: a few-step,
autoregressive, layout-conditioned video diffusion model that renders front-camera video from traffic
layouts produced by the
[BridgeSim](../BridgeSim) simulator.

## Requirements

- One NVIDIA GPU with ~56 GB of memory per server (~68 GB with `--full-vae-decoder`).
- ~64 GB of CPU memory.
- Python 3.10+, PyTorch with CUDA. `flash-attn` is optional.

## Installation

```bash
conda create -n dreamstream python=3.10 -y
conda activate dreamstream
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA
cd DreamStream
pip install -e .
pip install flash-attn --no-build-isolation   # optional
```

## Model weights

| Weights | Source | Size |
|---|---|---|
| DreamStream | [`UCLA-VAIL/DreamStream`](https://huggingface.co/UCLA-VAIL/DreamStream): `video_model/wan2.2-5b-nuplan-train-v1/ema_model.pt` | 20 GB |
| Base model  | [`alibaba-pai/Wan2.2-Fun-5B-Control`](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control) | 24 GB |
| TinyVAE decoder | [`lightx2v/Autoencoders`](https://huggingface.co/lightx2v/Autoencoders): `taew2_2.safetensors` | 23 MB |

All three are downloaded automatically on first launch into the Hugging Face cache.

To use local copies instead, pass `--wan-root /path/to/Wan2.2-Fun-5B-Control`
and `--diffusion-ckpt /path/to/ema_model.pt`.

## Running the server

```bash
python serve.py --num-gpus 1
```

This starts one server per GPU on ports 8790, 8791, ... Arguments after `--`
are forwarded to each server, e.g.

```bash
python serve.py --num-gpus 2 -- --wan-root /data/Wan2.2-Fun-5B-Control --diffusion-ckpt /data/ema_model.pt
```

A single server can also be started directly with `python -m dreamstream.server`.

Main options (`python -m dreamstream.server --help` for all):

| Option | Default | Description |
|---|---|---|
| `--port` | 8790 | HTTP port |
| `--prompt` | `a sunny highway` | Prompt used when the client sends none |
| `--size` | `1280x704` | Output resolution (`WIDTHxHEIGHT`) |
| `--seed` | 42 | Noise seed (`-1` for random) |
| `--sink-size`, `--history-cutoff` | 1, 2 | KV-cache attention sink / history window, in blocks |
| `--no-compile`, `--no-warmup` | off | Disable `torch.compile` / startup warmup |
| `--full-vae-decoder` | off | Decode with the full Wan2.2 VAE instead of TinyVAE |
| `--tvae-encoder`, `--tvae-ckpt` | off | Experimental: also encode layouts with TinyVAE; custom TinyVAE checkpoint |
| `--profile` | off | Print a per-stage GPU timing breakdown for every block |

## License

See [LICENSE](../LICENSE) and [NOTICE](../NOTICE). Most code is Apache-2.0.
The model weights are for non-commercial
research use.
