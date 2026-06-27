# %%writefile /content/DeepCAD_prashant/eval/train_completion.py
"""
train_completion.py
====================
Freeze any encoder, train a standard DeepCAD decoder on top.
Task: encode first 30 tokens → decode → reconstruct full 60-token sequence.
Evaluated on multi-block sequences only (n_ops >= 2).

Usage:
    encoder_name = 'ae' | 'mae' | 'jepa' | 'random'
    See bottom of file for launch examples.
"""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '/content/DeepCAD_prashant')

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import h5py, json
from cadlib.macro import *


# ── Completion Dataset ─────────────────────────────────────────────────────────
class CompletionDataset(Dataset):
    """
    Returns (context_60, full_60) pairs.
    context_60: first ctx_len tokens, rest padded with EOS
    full_60:    original full sequence (ground truth)
    Multi-block only: n_ops >= 2
    """
    def __init__(self, phase, data_root, dedup_path=None, ctx_len=30, max_total_len=60):
        self.raw_data     = os.path.join(data_root, 'cad_vec')
        self.ctx_len      = ctx_len
        self.max_total_len= max_total_len

        split_path = os.path.join(data_root, 'train_val_test_split.json')
        with open(split_path) as f:
            all_ids = json.load(f)[phase]

        # Dedup filter (train only)
        if phase == 'train' and dedup_path and os.path.exists(dedup_path):
            with open(dedup_path) as f:
                dedup_ids = set(json.load(f))
            all_ids = [i for i in all_ids if i in dedup_ids]

        # Multi-block filter — completion only meaningful for n_ops >= 2
        self.data = []
        for data_id in all_ids:
            h5_path = os.path.join(self.raw_data, data_id + '.h5')
            try:
                with h5py.File(h5_path, 'r') as fp:
                    vec = fp['vec'][:]
                n_ops = (vec[:, 0] == EXT_IDX).sum()
                if n_ops >= 2:
                    self.data.append(data_id)
            except Exception:
                pass

        print(f"[CompletionDataset] {phase}: {len(self.data)} multi-block sequences")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h5_path = os.path.join(self.raw_data, self.data[idx] + '.h5')
        with h5py.File(h5_path, 'r') as fp:
            vec = fp['vec'][:]

        # Pad to max_total_len
        pad = self.max_total_len - vec.shape[0]
        vec = np.concatenate([vec, EOS_VEC[np.newaxis].repeat(pad, axis=0)], axis=0)

        full_cmd = torch.tensor(vec[:, 0],  dtype=torch.long)  # (60,)
        full_args= torch.tensor(vec[:, 1:], dtype=torch.long)  # (60, 16)

        # Context: first ctx_len tokens, rest = EOS
        ctx_vec = EOS_VEC[np.newaxis].repeat(self.max_total_len, axis=0).copy()
        ctx_vec[:self.ctx_len] = vec[:self.ctx_len]
        ctx_cmd  = torch.tensor(ctx_vec[:, 0],  dtype=torch.long)
        ctx_args = torch.tensor(ctx_vec[:, 1:], dtype=torch.long)

        return {
            'ctx_cmd':   ctx_cmd,    # (60,) — first 30 real, rest EOS
            'ctx_args':  ctx_args,   # (60, 16)
            'full_cmd':  full_cmd,   # (60,) ground truth
            'full_args': full_args,  # (60, 16)
            'id':        self.data[idx]
        }


# ── Completion Loss ────────────────────────────────────────────────────────────
def completion_loss(cmd_logits, args_logits, full_cmd, full_args, weights=(1.0, 2.0)):
    """Standard AE loss on full sequence reconstruction. Batch-first throughout."""
    from model.model_utils import _get_padding_mask, _get_visibility_mask
    from cadlib.macro import CMD_ARGS_MASK

    # full_cmd: (N, S) batch-first
    vis = _get_visibility_mask(full_cmd, seq_dim=-1)          # (N,)
    pad = (_get_padding_mask(full_cmd, seq_dim=-1, extended=True)  # (N, S)
           * vis.unsqueeze(-1))                                # (N, 1) → (N, S)

    # Decoder outputs seq-first (S, N, ...) — permute to batch-first
    cmd_bf  = cmd_logits.permute(1, 0, 2)          # (N, S, n_cmd)
    args_bf = args_logits.permute(1, 0, 2, 3)      # (N, S, n_args, args_dim)

    # Command loss
    loss_cmd = nn.functional.cross_entropy(
        cmd_bf[pad.bool()].reshape(-1, cmd_bf.shape[-1]),
        full_cmd[pad.bool()].reshape(-1).long()
    )

    # Args loss
    mask     = torch.tensor(CMD_ARGS_MASK).cuda()[full_cmd.long()]  # (N, S, n_args)
    args_dim = args_bf.shape[-1]
    loss_args = nn.functional.cross_entropy(
        args_bf[mask.bool()].reshape(-1, args_dim),
        full_args[mask.bool()].reshape(-1).long() + 1
    )
    return weights[0] * loss_cmd + weights[1] * loss_args

# ── Completion Trainer ─────────────────────────────────────────────────────────
class CompletionTrainer:
    def __init__(self, encoder, decoder, lr=1e-3, warmup=2000):
        self.encoder = encoder.cuda().eval()
        self.decoder = decoder.cuda()

        # Freeze encoder completely
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = None
        self.global_step = 0

    def _build_scheduler(self, total_steps):
        warmup = 2000
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            t = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * t))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, loader):
        self.decoder.train()
        losses = []
        for batch in tqdm(loader, leave=False):
            ctx_cmd  = batch['ctx_cmd'].cuda()
            ctx_args = batch['ctx_args'].cuda()
            full_cmd = batch['full_cmd'].cuda()
            full_args= batch['full_args'].cuda()

            with torch.no_grad():
                z = self.encoder.get_pooled_embedding(ctx_cmd, ctx_args)  # (N, 256)
            z = z.unsqueeze(0)   # (1, N, 256) — matches decoder expectation

            cmd_logits, args_logits = self.decoder(z)

            loss = completion_loss(cmd_logits, args_logits, full_cmd, full_args)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.decoder.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.global_step += 1
            losses.append(loss.item())

        return float(np.mean(losses))

    @torch.no_grad()
    def predict_and_save(self, loader, save_dir):
        """Run inference, save h5 files for evaluate_ae_acc.py."""
        os.makedirs(save_dir, exist_ok=True)
        self.decoder.eval()

        for batch in tqdm(loader, desc='Inference', leave=False):
            ctx_cmd  = batch['ctx_cmd'].cuda()
            ctx_args = batch['ctx_args'].cuda()
            full_cmd = batch['full_cmd']
            full_args= batch['full_args']

            z = self.encoder.get_pooled_embedding(ctx_cmd, ctx_args).unsqueeze(0)
            cmd_logits, args_logits = self.decoder(z)

            out_cmd  = cmd_logits.permute(1,0,2).argmax(-1).cpu().numpy()         # (N, 60)
            out_args = args_logits.permute(1,0,2,3).argmax(-1).cpu().numpy() - 1  # (N, 60, 16)

            for i, seq_id in enumerate(batch['id']):
                out_vec = torch.cat([out_cmd[i:i+1].T,
                                     out_args[i]], dim=-1).numpy()    # (60, 17)
                gt_vec  = torch.cat([full_cmd[i:i+1].T,
                                     full_args[i]], dim=-1).numpy()   # (60, 17)
                safe_id = seq_id.replace('/', '_')
                h5_path = os.path.join(save_dir, f'{safe_id}.h5')
                with h5py.File(h5_path, 'w') as fp:
                    fp.create_dataset('out_vec', data=out_vec.astype(np.int64))
                    fp.create_dataset('gt_vec',  data=gt_vec.astype(np.int64))


# ── ACC computation (inline, no file I/O needed) ──────────────────────────────
@torch.no_grad()
def compute_acc(encoder, decoder, loader, ctx_len=30):
    """
    Compute ACC_cmd and ACC_param directly without saving h5 files.
    Tolerance=3 on continuous params, exact on discrete (as per evaluate_ae_acc.py).
    """
    decoder.eval()
    TOLERANCE = 3
    all_cmd_acc, all_param_acc = [], []

    for batch in tqdm(loader, desc='ACC eval', leave=False):
        ctx_cmd  = batch['ctx_cmd'].cuda()
        ctx_args = batch['ctx_args'].cuda()
        full_cmd = batch['full_cmd'].numpy()
        full_args= batch['full_args'].numpy()

        z = encoder.get_pooled_embedding(ctx_cmd, ctx_args).unsqueeze(0)
        cmd_logits, args_logits = decoder(z)

        out_cmd  = cmd_logits.argmax(-1).cpu().numpy()       # (N, 60)
        out_args = args_logits.argmax(-1).cpu().numpy() - 1  # (N, 60, 16)

        for i in range(len(full_cmd)):
            gt_c  = full_cmd[i]   # (60,)
            gt_a  = full_args[i]  # (60, 16)
            pr_c  = out_cmd[i]
            pr_a  = out_args[i]

            cmd_acc  = (gt_c == pr_c).astype(float)
            all_cmd_acc.append(cmd_acc.mean())

            param_acc = []
            for j in range(len(gt_c)):
                if gt_c[j] in [SOL_IDX, EOS_IDX]:
                    continue
                if pr_c[j] != gt_c[j]:
                    continue
                tol = (np.abs(gt_a[j] - pr_a[j]) < TOLERANCE).astype(float)
                if gt_c[j] == EXT_IDX:
                    tol[-2:] = (gt_a[j][-2:] == pr_a[j][-2:]).astype(float)
                elif gt_c[j] == ARC_IDX:
                    tol[3] = float(gt_a[j][3] == pr_a[j][3])
                param_acc.extend(tol[CMD_ARGS_MASK[gt_c[j]].astype(bool)].tolist())

            if param_acc:
                all_param_acc.append(np.mean(param_acc))

    return np.mean(all_cmd_acc), np.mean(all_param_acc) if all_param_acc else 0.0