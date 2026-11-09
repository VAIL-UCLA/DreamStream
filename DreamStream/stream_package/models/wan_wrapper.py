import os
import types
from typing import List

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from stream_package.models.wan_base.modules.t5 import umt5_xxl
from stream_package.models.wan_base.modules.tokenizers import HuggingfaceTokenizer
from stream_package.models.wan_base.modules.vae2_2 import count_conv3d, patchify, unpatchify


def clean_decoder_cache(self):
    self._conv_num = count_conv3d(self.decoder)
    self._conv_idx = [0]
    self._feat_map = [None] * self._conv_num


def clean_encoder_cache(self):
    self._enc_conv_num = count_conv3d(self.encoder)
    self._enc_conv_idx = [0]
    self._enc_feat_map = [None] * self._enc_conv_num


def streaming_decode(self, z, scale, start=True, end=True):
    if start:
        self.clean_decoder_cache()
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
    else:
        z = z / scale[1] + scale[0]
    iter_ = z.shape[2]
    x = self.conv2(z)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            if start:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True
                )
            else:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        else:
            out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)
    out = unpatchify(out, patch_size=2)
    if end:
        self.clean_decoder_cache()
    return out


def streaming_encode(self, x, scale, clear_cache=True):
    if clear_cache:
        self.clean_encoder_cache()
    x = patchify(x, patch_size=2)
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
            )
            out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    if isinstance(scale[0], torch.Tensor):
        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
    else:
        mu = (mu - scale[0]) * scale[1]
    if clear_cache:
        self.clean_encoder_cache()
    return mu


class WanTextEncoderWrapper(ModelMixin, ConfigMixin):
    def __init__(self, text_encoder, tokenizer):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.sample_neg_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    @classmethod
    def from_pretrained(cls, wan_root):
        text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        text_encoder.load_state_dict(
            torch.load(
                os.path.join(wan_root, "models_t5_umt5-xxl-enc-bf16.pth"),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        )
        tokenizer = HuggingfaceTokenizer(
            name=os.path.join(wan_root, "google/umt5-xxl/"),
            seq_len=512,
            clean="whitespace",
        )
        return cls(text_encoder, tokenizer)

    def forward(self, text_prompts: List[str], stacked=True):
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = torch.cat(
            [self.text_encoder(a.unsqueeze(0), b.unsqueeze(0)) for a, b in zip(ids, mask)],
            dim=0,
        )

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        if stacked:
            return context
        else:
            return [u[:v] for u, v in zip(context, seq_lens)]


class WanVAEWrapper(ModelMixin, ConfigMixin):
    def __init__(self, model):
        super().__init__()
        mean = [
            -0.2289,
            -0.0052,
            -0.1323,
            -0.2339,
            -0.2799,
            0.0174,
            0.1838,
            0.1557,
            -0.1382,
            0.0542,
            0.2813,
            0.0891,
            0.1570,
            -0.0098,
            0.0375,
            -0.1825,
            -0.2246,
            -0.1207,
            -0.0698,
            0.5109,
            0.2665,
            -0.2108,
            -0.2158,
            0.2502,
            -0.2055,
            -0.0322,
            0.1109,
            0.1567,
            -0.0729,
            0.0899,
            -0.2799,
            -0.1230,
            -0.0313,
            -0.1649,
            0.0117,
            0.0723,
            -0.2839,
            -0.2083,
            -0.0520,
            0.3748,
            0.0152,
            0.1957,
            0.1433,
            -0.2944,
            0.3573,
            -0.0548,
            -0.1681,
            -0.0667,
        ]
        std = [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.4990,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.0600,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = model
        self.model.clean_decoder_cache = types.MethodType(clean_decoder_cache, self.model)
        self.model.clean_encoder_cache = types.MethodType(clean_encoder_cache, self.model)
        self.model.decode = types.MethodType(streaming_decode, self.model)
        self.model.encode = types.MethodType(streaming_encode, self.model)

    @classmethod
    def from_pretrained(cls, wan_root: str, use_tvae_decoder: bool = False,
                        use_tvae_encoder: bool = False, tvae_ckpt: str = None):
        from stream_package.models.wan_base.modules.vae2_2 import _video_vae

        model = _video_vae(pretrained_path=os.path.join(wan_root, "Wan2.2_VAE.pth"))
        model = model.eval().requires_grad_(False)
        model = model.to("cpu")

        vae_wrapper = cls(model)
        vae_wrapper._use_tvae_encoder = False

        # Replace decoder (and optionally encoder) with TinyVAE if requested
        if (use_tvae_decoder or use_tvae_encoder) and tvae_ckpt:
            from stream_package.models.tiny_vae import TinyVAE

            tvae = TinyVAE.from_pretrained(checkpoint_path=tvae_ckpt)
            tvae = tvae.eval().requires_grad_(False).to("cpu")
            vae_wrapper.tvae = tvae

            if use_tvae_encoder:
                # Keep TinyVAE encoder, delete the heavy original encoder
                vae_wrapper._use_tvae_encoder = True
                del vae_wrapper.model.encoder
                vae_wrapper.model.encoder = None
                vae_wrapper.model.clean_encoder_cache = tvae.clean_encoder_cache
            else:
                del tvae.encoder
                tvae.encoder = None

            if use_tvae_decoder:
                vae_wrapper.decode = tvae.decode
                vae_wrapper.model.clean_decoder_cache = tvae.clean_decoder_cache
                del vae_wrapper.model.decoder
                vae_wrapper.model.decoder = None

        return vae_wrapper

    def encode(self, videos, stacked=True, clear_cache=True):
        if self._use_tvae_encoder:
            return self._tvae_encode(videos, stacked=stacked, clear_cache=clear_cache)
        # (list or batched) video of shape tchw -> cthw -> tchw
        self.scale = [i.to(self.device, dtype=videos[0].dtype) for i in self.scale]
        videos = [u.permute(1, 0, 2, 3) for u in videos]  # tchw -> cthw
        encoded = [self.model.encode(u.unsqueeze(0), self.scale, clear_cache=clear_cache).squeeze(0) for u in videos]
        encoded = [u.permute(1, 0, 2, 3) for u in encoded]  # cthw -> tchw
        if stacked:
            return torch.stack(encoded, dim=0)
        else:
            return encoded

    def _tvae_encode(self, videos, stacked=True, clear_cache=True):
        # TinyVAE encode: (N, T, C, H, W) in [-1, 1] -> (N, T', C', H', W')
        encoded = []
        for video in videos:
            # video: (T, C, H, W) -> (1, T, C, H, W)
            x = video.unsqueeze(0)
            z = self.tvae.encode(x, start=True, end=clear_cache, to_0_1=True)
            encoded.append(z.squeeze(0))  # (T', C', H', W')
        if stacked:
            return torch.stack(encoded, dim=0)
        return encoded

    def decode(self, zs, stacked=True, start=True, end=True):
        # (list or batched) latent of shape tchw -> cthw -> tchw
        self.scale = [i.to(self.device, dtype=zs[0].dtype) for i in self.scale]
        zs = [u.permute(1, 0, 2, 3) for u in zs]  # tchw -> cthw
        decoded = [
            self.model.decode(u.unsqueeze(0), self.scale, start=start, end=end)
            .clamp_(-1, 1)
            .squeeze(0)
            for u in zs
        ]
        decoded = [u.permute(1, 0, 2, 3) for u in decoded]  # cthw -> tchw
        if stacked:
            return torch.stack(decoded, dim=0)
        else:
            return decoded
