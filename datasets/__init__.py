import pytorch_lightning as pl
import torch
from .interedit import InterEditDataset

from datasets.evaluator import (
    EvaluatorModelWrapper,
    EvaluationDataset,
    get_dataset_motion_loader,
    get_motion_loader
)

__all__ = [
    "InterEditDataset", "EvaluationDataset",
    "get_dataset_motion_loader", "get_motion_loader",
    "build_loader", "DataModule"
]


def _norm_name(x) -> str:
    return str(x).strip().lower()


def _build_dataset(data_cfg):
    name = _norm_name(getattr(data_cfg, "NAME", ""))

    if name in ["interedit", "interhuman"]:
        return InterEditDataset(data_cfg)

    raise NotImplementedError(f"Unknown dataset NAME={getattr(data_cfg, 'NAME', None)}")


def build_loader(cfg, data_cfg):
    train_dataset = _build_dataset(data_cfg)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=1,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    return loader


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = _build_dataset(self.cfg)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )