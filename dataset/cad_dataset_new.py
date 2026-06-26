from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
import numpy as np
from cadlib.macro import *


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle
    dataset    = CADDataset(phase, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_shuffle,
        num_workers=config.num_workers,
        worker_init_fn=lambda _: np.random.seed(),
    )
    print(f"[Dataloader] Phase: {phase} | Batches: {len(dataloader)} | "
          f"Batch size: {config.batch_size} | Shuffle: {is_shuffle}")
    return dataloader


class CADDataset(Dataset):
    def __init__(self, phase, config):
        super().__init__()
        self.raw_data      = os.path.join(config.data_root, "cad_vec")
        self.phase         = phase
        self.aug           = getattr(config, 'augment',          False)
        self.jitter_aug    = getattr(config, 'jitter_aug',       False)
        self.jitter_str    = getattr(config, 'jitter_strength',  2)
        self.translate_aug = getattr(config, 'translate_aug',    False)
        self.translate_str = getattr(config, 'translate_strength', 15)
        self.max_total_len = config.max_total_len
        self.max_n_loops   = config.max_n_loops
        self.max_n_curves  = config.max_n_curves

        # ── Load split ────────────────────────────────────────────────────
        # split_path = os.path.join(config.data_root, "deepcad_data/train_val_test_split.json")
        split_path = "content/deepcad_data/train_val_test_split.json"
        with open(split_path) as f:
            self.all_data = json.load(f)[phase]

        # ── Dedup filter (train only) ─────────────────────────────────────
        # Uses HNC-CAD's geometrically deduplicated ID list intersected
        # with our train split. Val/test untouched — fair evaluation.
        dedup_path = getattr(config, 'dedup_ids_path', None)
        if phase == 'train' and dedup_path and os.path.exists(dedup_path):
            with open(dedup_path) as f:
                dedup_ids = set(json.load(f))
            before = len(self.all_data)
            self.all_data = [i for i in self.all_data if i in dedup_ids]
            print(f"[Dataset] Dedup filter: {before} → {len(self.all_data)} "
                  f"(removed {before - len(self.all_data)})")
        else:
            if phase == 'train' and dedup_path:
                print(f"[Dataset] WARNING: dedup_ids_path not found: {dedup_path}")

        print(f"[Dataset] Phase: {phase} | Samples: {len(self.all_data)}")
        print(f"[Dataset] data_root: {self.raw_data}")
        print(f"[Dataset] First 3 IDs: {self.all_data[:3]}")

    # ── Coordinate translation augmentation ───────────────────────────────
    # Attacks the x=128 center-bias found in EDA (37% of Line x values).
    # Shifts all sketch coordinate args by a random offset ±translate_str.
    # Only LINE/ARC/CIRCLE coordinate columns — never command col, never EXT.
    #
    # Column layout (from cadlib.macro):
    #   LINE:   col 0 = cmd, col 1 = x,  col 2 = y   (cols 3-16 = PAD)
    #   ARC:    col 0 = cmd, col 1 = x,  col 2 = y,
    #                        col 3 = sweep, col 4 = ?  (col 5+ = PAD)
    #   CIRCLE: col 0 = cmd, col 1 = cx, col 2 = cy, col 3 = ?,
    #                        col 4 = r               (col 5+ = PAD)
    #
    # We shift cols 1 and 2 (x, y) for all three curve types.
    # This is a global translation of the entire sketch — topologically
    # neutral but breaks the positional prior.
    # ─────────────────────────────────────────────────────────────────────

    def _apply_translate_aug(self, cad_vec):
        """
        cad_vec: (raw_len, 17) int array — before padding.
        Returns modified copy.
        """
        CURVE_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}
        is_curve   = np.array([int(c) in CURVE_CMDS for c in cad_vec[:, 0]])
        if not is_curve.any():
            return cad_vec

        # Single offset per sequence — global translation, not per-token
        dx = random.randint(-self.translate_str, self.translate_str)
        dy = random.randint(-self.translate_str, self.translate_str)

        cad_vec = cad_vec.copy().astype(np.int32)

        # Shift x (col 1) and y (col 2) for all curve rows
        # Clip to [0, 255] — stay within quantization range
        cad_vec[is_curve, 1] = np.clip(cad_vec[is_curve, 1] + dx, 0, 255)
        cad_vec[is_curve, 2] = np.clip(cad_vec[is_curve, 2] + dy, 0, 255)

        return cad_vec.astype(np.int64)

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:]   # (raw_len, 17) int64

        # ── Block-swap augmentation (off by default) ──────────────────────
        # Swaps random blocks between two sequences.
        # Controlled by --augment flag. Not used in clean JEPA training.
        if self.aug and self.phase == 'train':
            command1    = cad_vec[:, 0]
            ext_indices = np.where(command1 == EXT_IDX)[0]
            if len(ext_indices) > 1 and random.uniform(0, 1) > 0.5:
                ext_vec1 = np.split(cad_vec, ext_indices + 1, axis=0)[:-1]

                data_id2 = self.all_data[random.randint(0, len(self.all_data) - 1)]
                h5_path2 = os.path.join(self.raw_data, data_id2 + ".h5")
                with h5py.File(h5_path2, "r") as fp:
                    cad_vec2 = fp["vec"][:]

                command2     = cad_vec2[:, 0]
                ext_indices2 = np.where(command2 == EXT_IDX)[0]
                ext_vec2     = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]

                n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
                old_idx   = sorted(random.sample(range(len(ext_vec1)), n_replace))
                new_idx   = sorted(random.sample(range(len(ext_vec2)), n_replace))
                for i in range(len(old_idx)):
                    ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]

                new_vec, total = [], 0
                for chunk in ext_vec1:
                    total += len(chunk)
                    if total > self.max_total_len:
                        break
                    new_vec.append(chunk)
                cad_vec = np.concatenate(new_vec, axis=0)

        # ── Jitter augmentation ───────────────────────────────────────────
        # ±jitter_str quantization noise on curve arg values.
        # Prevents encoder from memorizing exact parameter values.
        # Applied before translation so both attacks are independent.
        if self.jitter_aug and self.phase == 'train':
            CURVE_CMDS = {LINE_IDX, ARC_IDX, CIRCLE_IDX}
            is_curve   = np.array([int(c) in CURVE_CMDS for c in cad_vec[:, 0]])
            if is_curve.any():
                s             = self.jitter_str
                jitter        = np.random.randint(-s, s + 1,
                                                  size=cad_vec[is_curve].shape)
                jitter[:, 0]  = 0   # never touch command column
                cad_vec[is_curve] = np.clip(
                    cad_vec[is_curve].astype(np.int32) + jitter, 0, 255
                ).astype(cad_vec.dtype)

        # ── Coordinate translation augmentation ───────────────────────────
        # Shifts all sketch x,y coordinates by a random global offset.
        # Attacks the x=128 center-bias (37% of Line x values in EDA).
        # Applied after jitter — both augmentations are independent.
        if self.translate_aug and self.phase == 'train':
            cad_vec = self._apply_translate_aug(cad_vec)

        # ── Padding to max_total_len ──────────────────────────────────────
        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate(
            [cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0
        )

        command = torch.tensor(cad_vec[:, 0],  dtype=torch.long)
        args    = torch.tensor(cad_vec[:, 1:], dtype=torch.long)

        return {"command": command, "args": args, "id": data_id}

    def __len__(self):
        return len(self.all_data)