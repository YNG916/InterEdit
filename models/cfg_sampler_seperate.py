import torch
import torch.nn as nn


class ClassifierFreeSampleModel(nn.Module):
    """
    Two-condition CFG with separate branches:
      - full:        (text, source)
      - motion-only: (0,    source)
      - uncond:      (0,    0)

    eps_hat = eps_u + s_m*(eps_m - eps_u) + s_t*(eps_f - eps_m)
    """
    def __init__(self, model, cfg_scale_text: float, cfg_scale_motion: float):
        super().__init__()
        self.model = model
        self.s_text = float(cfg_scale_text)
        self.s_motion = float(cfg_scale_motion)

    def forward(self, x, timesteps, cond=None, mask=None, source_emb=None):
        B = x.shape[0]

        # ---- pack 3 branches ----
        # [full, motion-only, uncond]
        x_cat = torch.cat([x, x, x], dim=0)
        t_cat = torch.cat([timesteps, timesteps, timesteps], dim=0)

        # text: full uses cond, others use 0
        if cond is not None:
            cond_cat = torch.cat([cond, torch.zeros_like(cond), torch.zeros_like(cond)], dim=0)
        else:
            cond_cat = None

        # source: full & motion-only use source_emb, uncond uses 0
        if source_emb is not None:
            source_cat = torch.cat([source_emb, source_emb, torch.zeros_like(source_emb)], dim=0)
        else:
            source_cat = None

        if mask is not None:
            mask_cat = torch.cat([mask, mask, mask], dim=0)
        else:
            mask_cat = None

        out = self.model(
            x_cat,
            t_cat,
            cond=cond_cat,
            mask=mask_cat,
            source_emb=source_cat,
        )

        eps_f = out[:B]
        eps_m = out[B:2*B]
        eps_u = out[2*B:]

        # ---- two-scale guidance ----
        eps_hat = (
            eps_u
            + self.s_motion * (eps_m - eps_u)
            + self.s_text * (eps_f - eps_m)
        )
        return eps_hat