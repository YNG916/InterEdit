import os
import numpy as np
import torch
import random

from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin

from utils.utils import *
from utils.plot_script import *
from utils.preprocess import *


class InterEditDataset(data.Dataset):
    """
    Editing dataset:
      - source motions from motions_source/person1|person2
      - target motions from motions_processed/person1|person2
      - text from annots/*.txt
    Returns:
      name, text, src1, src2, tgt1, tgt2, src_len, tgt_len
    """
    def __init__(self, opt):
        self.opt = opt
        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length - 1
        self.min_length = self.min_cond_length + self.min_gt_length - 1

        self.motion_rep = opt.MOTION_REP
        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE

        ignore_list = []
        try:
            ignore_list = open(os.path.join(opt.DATA_ROOT, "ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)

        data_list = []
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        random.shuffle(data_list)

        index = 0
        for root, dirs, files in os.walk(pjoin(opt.DATA_ROOT)):
            for file in tqdm(files):
                # use motions_processed/person1 as anchor to enumerate ids
                if file.endswith(".npy") and "motions_processed" in root and "person1" in root:
                    motion_name = file.split(".")[0]

                    if file.split(".")[0] + "\n" in ignore_list:
                        print("ignore: ", file)
                        continue
                    if file.split(".")[0] + "\n" not in data_list:
                        continue

                    tgt_p1 = pjoin(root, file)  # motions_processed/person1/{id}.npy
                    tgt_p2 = pjoin(root.replace("person1", "person2"), file)
                    src_p1 = tgt_p1.replace("motions_processed", "motions_source")
                    src_p2 = tgt_p2.replace("motions_processed", "motions_source")

                    # text path: data_root/annots/{id}.txt
                    text_path = tgt_p1.replace("motions_processed", "annots").replace("person1", "").replace("npy", "txt")
                    if not os.path.exists(text_path):
                        print(f"[WARN] missing text: {text_path}, skip {motion_name}")
                        continue

                    texts = [item.replace("\n", "") for item in open(text_path, "r").readlines()]
                    texts_swap = [
                        item.replace("\n", "")
                            .replace("left", "tmp").replace("right", "left").replace("tmp", "right")
                            .replace("clockwise", "tmp").replace("counterclockwise", "clockwise").replace("tmp", "counterclockwise")
                        for item in texts
                    ]

                    # optional cache: store motions in memory
                    if self.cache:
                        # load swap versions for augmentation
                        tgt1, tgt1_swap = load_motion(tgt_p1, self.min_length, swap=True)
                        tgt2, tgt2_swap = load_motion(tgt_p2, self.min_length, swap=True)
                        src1, src1_swap = load_motion(src_p1, self.min_length, swap=True)
                        src2, src2_swap = load_motion(src_p2, self.min_length, swap=True)

                        if tgt1 is None or src1 is None:
                            continue

                        self.motion_dict[index] = [src1, src2, tgt1, tgt2]
                        self.motion_dict[index + 1] = [src1_swap, src2_swap, tgt1_swap, tgt2_swap]
                    else:
                        # store paths; swap will be handled in __getitem__ by selecting swap version
                        self.motion_dict[index] = [src_p1, src_p2, tgt_p1, tgt_p2]
                        self.motion_dict[index + 1] = [src_p1, src_p2, tgt_p1, tgt_p2]

                    self.data_list.append({
                        "name": motion_name,
                        "motion_id": index,
                        "swap": False,
                        "texts": texts
                    })
                    if opt.MODE == "train":
                        self.data_list.append({
                            "name": motion_name + "_swap",
                            "motion_id": index + 1,
                            "swap": True,
                            "texts": texts_swap
                        })

                    index += 2

        print("total dataset: ", len(self.data_list))

    def real_len(self):
        return len(self.data_list)

    def __len__(self):
        return self.real_len() * 1

    @staticmethod
    def _pad_to_max(x: np.ndarray, max_len: int) -> np.ndarray:
        Lx = x.shape[0]
        if Lx < max_len:
            pad = np.zeros((max_len - Lx, x.shape[1]), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        return x

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        swap = data["swap"]
        text = random.choice(data["texts"]).strip()

        # 1) load src/tgt
        if self.cache:
            src1_full, src2_full, tgt1_full, tgt2_full = self.motion_dict[motion_id]
        else:
            src_p1, src_p2, tgt_p1, tgt_p2 = self.motion_dict[motion_id]

            # load with swap=True to get both; select by `swap`
            src1, src1_swap = load_motion(src_p1, self.min_length, swap=True)
            src2, src2_swap = load_motion(src_p2, self.min_length, swap=True)
            tgt1, tgt1_swap = load_motion(tgt_p1, self.min_length, swap=True)
            tgt2, tgt2_swap = load_motion(tgt_p2, self.min_length, swap=True)

            if src1 is None or tgt1 is None:
                ridx = random.randint(0, self.real_len() - 1)
                return self.__getitem__(ridx)

            if swap:
                src1_full, src2_full = src1_swap, src2_swap
                tgt1_full, tgt2_full = tgt1_swap, tgt2_swap
            else:
                src1_full, src2_full = src1, src2
                tgt1_full, tgt2_full = tgt1, tgt2

        # 2) synchronized crop (use min length to keep alignment)
        length = min(src1_full.shape[0], tgt1_full.shape[0])
        if length <= 0:
            ridx = random.randint(0, self.real_len() - 1)
            return self.__getitem__(ridx)

        if length > self.max_length:
            start = random.choice(list(range(0, max(1, length - self.max_gt_length), 1)))
            L = self.max_gt_length
        else:
            start = 0
            L = min(length - start, self.max_gt_length)

        src1 = src1_full[start:start + L]
        src2 = src2_full[start:start + L]
        tgt1 = tgt1_full[start:start + L]
        tgt2 = tgt2_full[start:start + L]

        # 3) random swap persons — must be synchronized between src & tgt
        if np.random.rand() > 0.5:
            src1, src2 = src2, src1
            tgt1, tgt2 = tgt2, tgt1

        # 4) preprocess (process_motion_np + rigid_transform) for tgt pair
        tgt1, rq1, rp1 = process_motion_np(tgt1, 0.001, 0, n_joints=22)
        tgt2, rq2, rp2 = process_motion_np(tgt2, 0.001, 0, n_joints=22)
        r_relative = qmul_np(rq2, qinv_np(rq1))
        angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])
        xz = qrot_np(rq1, rp2 - rp1)[:, [0, 2]]
        relative = np.concatenate([angle, xz], axis=-1)[0]
        tgt2 = rigid_transform(relative, tgt2)

        # 5) preprocess for src pair
        src1, srq1, srp1 = process_motion_np(src1, 0.001, 0, n_joints=22)
        src2, srq2, srp2 = process_motion_np(src2, 0.001, 0, n_joints=22)
        sr_relative = qmul_np(srq2, qinv_np(srq1))
        s_angle = np.arctan2(sr_relative[:, 2:3], sr_relative[:, 0:1])
        s_xz = qrot_np(srq1, srp2 - srp1)[:, [0, 2]]
        s_relative = np.concatenate([s_angle, s_xz], axis=-1)[0]
        src2 = rigid_transform(s_relative, src2)

        # 6) pad all to max_gt_length
        tgt1 = self._pad_to_max(tgt1, self.max_gt_length)
        tgt2 = self._pad_to_max(tgt2, self.max_gt_length)
        src1 = self._pad_to_max(src1, self.max_gt_length)
        src2 = self._pad_to_max(src2, self.max_gt_length)

        tgt_len = min(L, self.max_gt_length)
        src_len = min(L, self.max_gt_length)

        assert len(tgt1) == self.max_gt_length
        assert len(tgt2) == self.max_gt_length
        assert len(src1) == self.max_gt_length
        assert len(src2) == self.max_gt_length

        # 7) final optional swap — still synchronized
        if np.random.rand() > 0.5:
            src1, src2 = src2, src1
            tgt1, tgt2 = tgt2, tgt1

        return name, text, src1, src2, tgt1, tgt2, src_len, tgt_len

