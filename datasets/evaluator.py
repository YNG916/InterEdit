from os.path import join as pjoin
import copy
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import InterEditDataset
from datasets.evaluator_models import InterCLIP
from utils.utils import MotionNormalizer



def build_eval_model(cfg):
    model = InterCLIP(cfg)
    ckpt_path = pjoin("eval_model", "interclip.ckpt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    sd = checkpoint.get("state_dict", checkpoint)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k.replace("model.", "", 1)] = v
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd, strict=True)
    return model


class EvaluatorModelWrapper(object):
    """
    Encode text/motion with InterCLIP.
    IMPORTANT: restore original order for motion embeddings (and text embeddings for get_co_embeddings).
    """
    def __init__(self, cfg, device):
        self.model = build_eval_model(cfg).to(device)
        self.model.eval()
        self.cfg = cfg
        self.device = device

    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()
            motion2 = motion2.detach().float()
            motions = torch.cat([motion1, motion2], dim=-1).to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens_sorted = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, int(m_len)) for m_len in motion_lens_sorted]).to(self.device)
            padded_len = int(cur_len.max().item())

            batch = {
                "text": text,
                "motions": motions.reshape(B, T, -1)[:, :padded_len],
                "motion_lens": motion_lens_sorted,
            }

            motion_embedding = self.model.encode_motion(batch)["motion_emb"]
            text_embedding = self.model.encode_text(batch)["text_emb"][align_idx]

            inv = np.argsort(align_idx)
            motion_embedding = motion_embedding[inv]
            text_embedding = text_embedding[inv]
        return text_embedding, motion_embedding

    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()
            motion2 = motion2.detach().float()
            motions = torch.cat([motion1, motion2], dim=-1).to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens_sorted = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, int(m_len)) for m_len in motion_lens_sorted]).to(self.device)
            padded_len = int(cur_len.max().item())

            batch = {
                "text": text,
                "motions": motions.reshape(B, T, -1)[:, :padded_len],
                "motion_lens": motion_lens_sorted,
            }

            motion_embedding = self.model.encode_motion(batch)["motion_emb"]

            inv = np.argsort(align_idx)
            motion_embedding = motion_embedding[inv]
        return motion_embedding



class EvaluationDataset(Dataset):
    """
    Cache generated motions:
    """
    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()

        self.max_length = getattr(dataset, "max_gt_length", getattr(dataset, "max_length", 300))

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = set(idxs[:mm_num_samples])

        self.generated_motions = []
        self.mm_generated_motions = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Caching generated motions"):
                R = mm_num_repeats if i in mm_idxs else 1

                if len(data) == 5:
                    # name, text, motion1, motion2, motion_lens
                    name, text, motion1, motion2, motion_lens = data
                    # NOTE: forward_test expects list[str]
                    batch = {
                        "text": list(text) * R,
                        "motion_lens": motion_lens.repeat(R),
                    }

                elif len(data) == 8:
                    # name, text, src1, src2, tgt1, tgt2, src_len, tgt_len
                    name, text, src1, src2, tgt1, tgt2, src_len, tgt_len = data

                    src1 = src1.detach().float().to(device)  # (1,300,262)
                    src2 = src2.detach().float().to(device)
                    src_len_i = int(src_len[0].item())
                    tgt_len_i = int(tgt_len[0].item())

                    T = max(1, tgt_len_i)
                    src1_T = src1[:, :T, :]
                    src2_T = src2[:, :T, :]
                    sources = torch.cat([src1_T, src2_T], dim=-1)  # (1,T,524)

                    batch = {
                        "text": list(text) * R,
                        "sources": sources.repeat(R, 1, 1),
                        "motion_lens": torch.LongTensor([T] * R).to(device),
                        "source_lens": torch.LongTensor([min(src_len_i, T)] * R).to(device),
                    }

                else:
                    raise RuntimeError(f"Unexpected dataset return length: {len(data)}")

                out = self.model.forward_test(batch)
                motions_output = out["output"]  # (B,T,2*D)

                motions_output = motions_output.reshape(motions_output.shape[0], motions_output.shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.detach().cpu().numpy())

                B, Tgen = motions_output.shape[0], motions_output.shape[1]
                if Tgen < self.max_length:
                    pad_len = self.max_length - Tgen
                    D = motions_output.shape[-1]
                    pad = np.zeros((B, pad_len, 2, D), dtype=motions_output.dtype)
                    motions_output = np.concatenate([motions_output, pad], axis=1)
                else:
                    motions_output = motions_output[:, :self.max_length]

                if "motion_lens" in out and isinstance(out["motion_lens"], torch.Tensor):
                    store_len = int(out["motion_lens"][0].item())
                else:
                    if len(data) == 8:
                        store_len = int(tgt_len[0].item())
                    else:
                        store_len = self.max_length

                self.generated_motions.append({
                    "motion1": motions_output[0, :, 0],
                    "motion2": motions_output[0, :, 1],
                    "motion_lens": store_len,
                    "text": str(text[0]),
                })

                if i in mm_idxs:
                    self.mm_generated_motions.append({
                        "mm_motions": motions_output,  # (R, max_len, 2, D)
                        "motion_lens": store_len,
                        "text": str(text[0]),
                    })

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, idx):
        d = self.generated_motions[idx]
        return "generated", d["text"], d["motion1"], d["motion2"], np.array(d["motion_lens"], dtype=np.int64)


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        mm = d["mm_motions"]  # (R, max_len, 2, D)
        mm1 = mm[:, :, 0]
        mm2 = mm[:, :, 1]
        L = np.array([d["motion_lens"]] * mm1.shape[0], dtype=np.int64)
        return "mm_generated", d["text"], mm1, mm2, L


def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats):
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=0, shuffle=False)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0, shuffle=False)

    print("Generated Dataset Loading Completed!!!")
    return motion_loader, mm_motion_loader



class EditEvaluationDataset(Dataset):
    """
    Cache aligned triples for editing:
      dataset return:
        name, text, src1, src2, tgt1, tgt2, src_len, tgt_len
    """
    def __init__(self, model, dataset, device):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()

        self.max_length = getattr(dataset, "max_gt_length", 300)
        self.generated_motions = []

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

        with torch.no_grad():
            for _, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Caching edit triples"):
                name, text, src1, src2, tgt1, tgt2, src_len, tgt_len = data

                src1 = src1.detach().float().to(device)  # (1,300,262)
                src2 = src2.detach().float().to(device)
                tgt1 = tgt1.detach().float().to(device)
                tgt2 = tgt2.detach().float().to(device)

                src_len_i = int(src_len[0].item())
                tgt_len_i = int(tgt_len[0].item())
                T = max(1, tgt_len_i)

                src1_T = src1[:, :T, :]
                src2_T = src2[:, :T, :]
                tgt1_T = tgt1[:, :T, :]
                tgt2_T = tgt2[:, :T, :]

                sources = torch.cat([src1_T, src2_T], dim=-1)  # (1,T,524)
                motions  = torch.cat([tgt1_T, tgt2_T], dim=-1) # (1,T,524)

                batch = {
                    "text": list(text),
                    "motions": motions,
                    "sources": sources,
                    "motion_lens": torch.LongTensor([T]).to(device),
                    "source_lens": torch.LongTensor([min(src_len_i, T)]).to(device),
                }

                out = self.model.forward_test(batch)
                gen = out["output"].reshape(out["output"].shape[0], out["output"].shape[1], 2, -1)

                gen_np = self.normalizer.backward(gen.detach().cpu().numpy())  # (1,T,2,D)

                # pad/crop gen to max_length for evaluator encoder
                B2, Tgen = gen_np.shape[0], gen_np.shape[1]
                if Tgen < self.max_length:
                    pad_len = self.max_length - Tgen
                    D2 = gen_np.shape[-1]
                    pad = np.zeros((B2, pad_len, 2, D2), dtype=gen_np.dtype)
                    gen_np = np.concatenate([gen_np, pad], axis=1)
                else:
                    gen_np = gen_np[:, :self.max_length]

                self.generated_motions.append({
                    "name": str(name[0]),
                    "text": str(text[0]),
                    "src1": src1.detach().cpu().numpy()[0],  # (300,262) padded
                    "src2": src2.detach().cpu().numpy()[0],
                    "tgt1": tgt1.detach().cpu().numpy()[0],
                    "tgt2": tgt2.detach().cpu().numpy()[0],
                    "gen1": gen_np[0, :, 0],                 # (300,262) padded
                    "gen2": gen_np[0, :, 1],
                    "src_len": src_len_i,
                    "tgt_len": tgt_len_i,
                })

    def __len__(self):
        return len(self.generated_motions)


class _GenDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        d = self.base.generated_motions[idx]
        return "gen", d["text"], d["gen1"], d["gen2"], np.array(d["tgt_len"], dtype=np.int64)


class _TgtDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        d = self.base.generated_motions[idx]
        return "tgt", d["text"], d["tgt1"], d["tgt2"], np.array(d["tgt_len"], dtype=np.int64)


class _SrcDataset(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        d = self.base.generated_motions[idx]
        return "src", d["text"], d["src1"], d["src2"], np.array(d["src_len"], dtype=np.int64)


def get_edit_motion_loader(batch_size, model, ground_truth_dataset, device):
    """
    Returns three aligned loaders: gen_loader, tgt_loader, src_loader
    """
    base = EditEvaluationDataset(model, ground_truth_dataset, device)

    gen_loader = DataLoader(_GenDataset(base), batch_size=batch_size, drop_last=False, num_workers=0, shuffle=False)
    tgt_loader = DataLoader(_TgtDataset(base), batch_size=batch_size, drop_last=False, num_workers=0, shuffle=False)
    src_loader = DataLoader(_SrcDataset(base), batch_size=batch_size, drop_last=False, num_workers=0, shuffle=False)

    print("Editing Generated Dataset Loading Completed!!!")
    return gen_loader, tgt_loader, src_loader



def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    if opt.NAME == "interedit":
        print(f"Loading dataset {opt.NAME} ...")
        dataset = InterEditDataset(opt)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False)
    else:
        raise KeyError("Dataset not Recognized !!")
    print("Ground Truth Dataset Loading Completed!!!")
    return dataloader, dataset
