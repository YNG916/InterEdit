from __future__ import annotations

import torch
import torch.nn as nn
from os.path import join as pjoin
from types import SimpleNamespace
import yaml


def set_requires_grad(model: nn.Module, flag: bool):
    for p in model.parameters():
        p.requires_grad = flag


def load_yaml_cfg(path: str):
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return SimpleNamespace(**d)


def build_interclip_teacher(cfg_or_path, ckpt_path: str | None = None) -> nn.Module:
    """
    Build InterCLIP teacher from cfg object OR yaml path.

    IMPORTANT: we import InterCLIP lazily here to avoid circular import:
      evaluator_models -> models -> nets -> interclip_teacher -> evaluator_models
    """
    from datasets.evaluator_models import InterCLIP

    if isinstance(cfg_or_path, str):
        cfg = load_yaml_cfg(cfg_or_path)
    else:
        cfg = cfg_or_path

    model = InterCLIP(cfg)

    if ckpt_path is None:
        ckpt_path = pjoin("eval_model", "interclip.ckpt")

    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = checkpoint.get("state_dict", checkpoint)

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k.replace("model.", "", 1)] = v
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=True)
    model.eval()
    set_requires_grad(model, False)
    return model


class InterCLIPTeacherWrapper(nn.Module):
    """
    Teacher wrapper that returns motion embedding (B,512) for target motion.
    Auto-adapts whether to drop foot-contact dims (last 4 dims per person) externally.
    """
    def __init__(self, teacher: nn.Module):
        super().__init__()
        self.teacher = teacher
        self._mode = None  # "raw" or "drop4"

    @staticmethod
    def _drop_contact(motions: torch.Tensor) -> torch.Tensor:
        B, T, DD = motions.shape
        x = motions.view(B, T, 2, -1)   # (B,T,2,D)
        if x.shape[-1] < 5:
            return motions
        x = x[..., :-4]
        return x.reshape(B, T, -1)

    @torch.no_grad()
    def motion_emb(self, motions: torch.Tensor, motion_lens: torch.Tensor) -> torch.Tensor:
        device = next(self.teacher.parameters()).device
        motions = motions.to(device)
        motion_lens = motion_lens.to(device)

        def _run(m: torch.Tensor) -> torch.Tensor:
            b = {"motions": m, "motion_lens": motion_lens}
            b = self.teacher.encode_motion(b)
            return b["motion_emb"]  # (B,512)

        if self._mode is None:
            try:
                out = _run(motions)
                self._mode = "raw"
                return out
            except Exception:
                out = _run(self._drop_contact(motions))
                self._mode = "drop4"
                return out
        else:
            if self._mode == "raw":
                return _run(motions)
            else:
                return _run(self._drop_contact(motions))
