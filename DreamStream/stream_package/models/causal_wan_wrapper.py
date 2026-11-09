import math

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin


class CausalWanDiffusionWrapper(ModelMixin, ConfigMixin):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.frame_seq_len = None

    def configure_seq_len(self, size, frame_num=None):
        """Compute tokens-per-latent-frame from the active resolution.

        Must be called once by the pipeline before the first forward pass so
        non-default resolutions get the right per-block seq_len.
        ``frame_num`` is accepted for symmetry with the pipeline call site but
        unused -- only spatial dims matter for ``frame_seq_len``.
        """
        del frame_num
        vae_down = 16 if self.model.model_type == "ti2v" else 8
        h_lat = size[1] // vae_down
        w_lat = size[0] // vae_down
        ph, pw = self.model.patch_size[1], self.model.patch_size[2]
        self.frame_seq_len = math.ceil((h_lat * w_lat) / (ph * pw))

    @staticmethod
    def _convert_flow_pred_to_x0(
        scheduler, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def forward(
        self,
        x,
        t,
        context,
        scheduler,
        seq_len=None,
        clip_fea=None,
        y=None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=None,
        current_end=None,
        streaming_kwargs=None,
        num_frame_per_block=None,
        sink_size=None,
        history_cutoff=None,
    ):
        if seq_len is None:
            if self.frame_seq_len is None:
                raise RuntimeError(
                    "frame_seq_len is not configured. Call "
                    "diffusion.configure_seq_len(size=...) before forward(), "
                    "or pass seq_len explicitly."
                )
            seq_len = self.frame_seq_len * x[0].shape[0]
        x = torch.stack(x, dim=0) if isinstance(x, list) else x
        y = torch.stack(y, dim=0) if isinstance(y, list) and y is not None else y
        x_perm = x.permute(0, 2, 1, 3, 4)
        y_perm = y.permute(0, 2, 1, 3, 4) if y is not None else None
        input_timestep = t

        flow_pred = self.model(
            x=x_perm,
            t=input_timestep,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y_perm,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            current_end=current_end,
            streaming_kwargs=streaming_kwargs,
            num_frame_per_block=num_frame_per_block,
            sink_size=sink_size,
            history_cutoff=history_cutoff,
        ).permute(0, 2, 1, 3, 4)

        pred_x0 = CausalWanDiffusionWrapper._convert_flow_pred_to_x0(
            scheduler=scheduler,
            flow_pred=flow_pred.flatten(0, 1),
            xt=x.flatten(0, 1),
            timestep=input_timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0
