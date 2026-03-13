import torch
import torch.nn as nn


class ClassifierFreeSampleModel(nn.Module):
    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None, source_emb=None):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)

        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)

        if source_emb is not None:
            source_emb = torch.cat([source_emb, torch.zeros_like(source_emb)], dim=0)

        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)

        out = self.model(
            x_combined,
            timesteps_combined,
            cond=cond,
            mask=mask,
            source_emb=source_emb,
        )

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s * out_cond + (1 - self.s) * out_uncond
        return cfg_out
