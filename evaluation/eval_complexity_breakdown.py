"""
eval_complexity_breakdown.py
────────────────────────────
Per-complexity frozen CAD completion evaluation.

For each encoder (AE / MAE / JEPA / ContrastCAD):
  1. Train a frozen decoder for FROZEN_EPOCHS (skips if checkpoint exists)
  2. Evaluate on multi-block test set (n_ops >= 2, 3377 sequences)
  3. Report ACC_cmd + ACC_param broken down by:
       Overall / 2-3 op / 4-5 op / 6+ op

Usage in Colab:
  %run /content/DeepCAD_prashant/evaluation/eval_complexity_breakdown.py

Outputs:
  {DRIVE}/complexity_eval/frozen_dec_{method}.pt   <- decoder checkpoints
  {DRIVE}/complexity_eval/complexity_breakdown.json <- full results
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '/content/DeepCAD_prashant')

# ── Paths ──────────────────────────────────────────────────────────────────────
DRIVE      = '/content/drive/MyDrive/cadjepa_data'
DATA_ROOT  = '/content/deepcad_data'
DEDUP_PATH = '/content/deepcad_data/train_dedup_ids.json'
OUT_DIR    = f'{DRIVE}/complexity_eval'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
FROZEN_EPOCHS = 100    # train frozen decoder for 100 epochs — matches original eval
DEC_LR        = 1e-3
BATCH_SIZE    = 256
WARMUP_STEPS  = 2000

# ── Imports that need sys.path set first ───────────────────────────────────────
from cadlib.macro import (
    EXT_IDX, SOL_IDX, EOS_IDX, ARC_IDX, LINE_IDX, CIRCLE_IDX,
    EOS_VEC, CMD_ARGS_MASK
)
from evaluation.train_completion import (
    CompletionDataset,
    completion_loss,
    decoder_outputs_to_batch_first,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_ae_encoder():
    from model.autoencoder import CADTransformer
    from config.configAE_paper import ConfigAEPaper
    sys.argv = ['eval']
    cfg = ConfigAEPaper('train')
    net = CADTransformer(cfg).cuda()
    ckpt = torch.load(f'{DRIVE}/ae_paper/latest.pt',
                      map_location='cuda', weights_only=False)
    net.load_state_dict(ckpt['net'])
    enc = net.encoder.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"  [AE] loaded ep200")
    return enc


def load_mae_encoder():
    from model.jepa_encoder import JEPAEncoder
    from config.configJEPA_new import ConfigJEPA
    sys.argv = ['eval', '--exp_name', 'eval']
    cfg = ConfigJEPA('train')
    enc = JEPAEncoder(cfg).cuda()
    ckpt = torch.load(f'{DRIVE}/mae_run1/latest.pt',
                      map_location='cuda', weights_only=False)
    enc.load_state_dict(ckpt['encoder'])
    enc = enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"  [MAE] loaded ep399")
    return enc


def load_jepa_encoder():
    from model.jepa_encoder import JEPAEncoder
    from config.configJEPA_new import ConfigJEPA
    sys.argv = ['eval', '--exp_name', 'eval']
    cfg = ConfigJEPA('train')
    enc = JEPAEncoder(cfg).cuda()
    ckpt = torch.load(f'{DRIVE}/jepa_run4/latest.pt',
                      map_location='cuda', weights_only=False)
    enc.load_state_dict(ckpt['ema_encoder'])
    enc = enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"  [JEPA] loaded ep399 (ema_encoder)")
    return enc


def load_contrastcad_encoder():
    """Run this only after ContrastCAD finishes (~ep400)."""
    from trainer.trainerContrastCAD import TrainerContrastCAD
    from config.configContrastCAD import ConfigContrastCAD
    sys.argv = ['eval']
    cfg = ConfigContrastCAD('train')
    trainer = TrainerContrastCAD(cfg)
    ckpt = torch.load(f'{DRIVE}/contrastcad_paper/latest.pt',
                      map_location='cuda', weights_only=False)
    trainer.net.load_state_dict(ckpt['net'])
    trainer.proj.load_state_dict(ckpt['proj'])
    enc = trainer.net.encoder.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    print(f"  [ContrastCAD] loaded (pre-projection encoder)")
    return enc


def build_fresh_decoder():
    """
    Decoder is always DeepCAD's Decoder class with AE config.
    Architecture is identical for all 4 encoders.
    """
    from model.autoencoder import CADTransformer
    from config.configAE_paper import ConfigAEPaper
    sys.argv = ['eval']
    cfg = ConfigAEPaper('train')
    decoder = CADTransformer(cfg).decoder.cuda()
    return decoder


# ═══════════════════════════════════════════════════════════════════════════════
# Frozen decoder training
# ═══════════════════════════════════════════════════════════════════════════════

def train_frozen_decoder(encoder, decoder, train_loader, ckpt_path):
    """
    Train decoder with encoder completely frozen.
    Saves checkpoint to ckpt_path. Skips training if checkpoint already exists.
    Returns trained decoder.
    """
    if os.path.exists(ckpt_path):
        print(f"  [skip training] loading existing: {os.path.basename(ckpt_path)}")
        saved = torch.load(ckpt_path, map_location='cuda', weights_only=False)
        decoder.load_state_dict(saved['decoder'])
        print(f"  [loaded] ep={saved.get('epoch', '?')}  "
              f"loss={saved.get('loss', '?'):.4f}")
        return decoder

    total_steps = FROZEN_EPOCHS * len(train_loader)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DEC_LR)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        t = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step = 0
    last_loss = None

    for ep in range(FROZEN_EPOCHS):
        decoder.train()
        losses = []

        for batch in tqdm(train_loader, desc=f'  ep={ep:03d}', leave=False):
            ctx_cmd  = batch['ctx_cmd'].cuda()
            ctx_args = batch['ctx_args'].cuda()
            full_cmd  = batch['full_cmd'].cuda()
            full_args = batch['full_args'].cuda()

            with torch.no_grad():
                z = encoder.get_pooled_embedding(ctx_cmd, ctx_args)  # (N, 256)
            z = z.unsqueeze(0)  # (1, N, 256)

            cmd_logits, args_logits = decoder(z)
            loss = completion_loss(cmd_logits, args_logits, full_cmd, full_args)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(loss.item())

        last_loss = float(np.mean(losses))
        if (ep + 1) % 10 == 0:
            print(f"  ep={ep:03d}  loss={last_loss:.4f}")

    torch.save({
        'decoder': decoder.state_dict(),
        'epoch': FROZEN_EPOCHS,
        'loss': last_loss,
    }, ckpt_path)
    print(f"  [saved] {ckpt_path}")
    return decoder


# ═══════════════════════════════════════════════════════════════════════════════
# Per-complexity evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_acc_by_complexity(encoder, decoder, loader):
    """
    Modified compute_acc that tracks n_ops per sample.

    Returns: list of dicts with keys:
      n_ops     (int)   — number of EXT tokens = operation count
      acc_cmd   (float) — command accuracy for this sample
      acc_param (float or None) — param accuracy (None if no valid tokens)
    """
    decoder.eval()
    TOLERANCE = 3
    results = []

    for batch in tqdm(loader, desc='  Eval', leave=False):
        ctx_cmd  = batch['ctx_cmd'].cuda()
        ctx_args = batch['ctx_args'].cuda()
        full_cmd  = batch['full_cmd'].numpy()   # (N, 60)
        full_args = batch['full_args'].numpy()  # (N, 60, 16)

        z = encoder.get_pooled_embedding(ctx_cmd, ctx_args).unsqueeze(0)
        cmd_logits, args_logits = decoder(z)

        cmd_bf, args_bf = decoder_outputs_to_batch_first(cmd_logits, args_logits)
        out_cmd  = cmd_bf.argmax(-1).cpu().numpy()          # (N, 60)
        out_args = args_bf.argmax(-1).cpu().numpy() - 1    # (N, 60, 16)

        N = full_cmd.shape[0]
        for i in range(N):
            gt_c = full_cmd[i]    # (60,)
            gt_a = full_args[i]   # (60, 16)
            pr_c = out_cmd[i]
            pr_a = out_args[i]

            # n_ops = number of extrusion tokens
            n_ops = int((gt_c == EXT_IDX).sum())

            # ACC_cmd
            acc_cmd = float((gt_c == pr_c).mean())

            # ACC_param — only for correctly predicted non-SOL/EOS commands
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
                param_acc.extend(
                    tol[CMD_ARGS_MASK[int(gt_c[j])].astype(bool)].tolist()
                )

            results.append({
                'n_ops':     n_ops,
                'acc_cmd':   acc_cmd,
                'acc_param': float(np.mean(param_acc)) if param_acc else None,
            })

    return results


def summarize_by_complexity(results):
    """
    Aggregate per-sample results into complexity groups.
    Groups: Overall / 2-3op / 4-5op / 6+op
    """
    GROUPS = [
        ('Overall', 0,   999),
        ('2-3op',   2,   3),
        ('4-5op',   4,   5),
        ('6+op',    6,   999),
    ]
    summary = {}
    for label, lo, hi in GROUPS:
        if label == 'Overall':
            subset = results
        else:
            subset = [r for r in results if lo <= r['n_ops'] <= hi]

        if not subset:
            summary[label] = {'n': 0, 'acc_cmd': 0.0, 'acc_param': 0.0}
            continue

        cmd_vals   = [r['acc_cmd'] for r in subset]
        param_vals = [r['acc_param'] for r in subset if r['acc_param'] is not None]
        summary[label] = {
            'n':         len(subset),
            'acc_cmd':   float(np.mean(cmd_vals)),
            'acc_param': float(np.mean(param_vals)) if param_vals else 0.0,
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Printing
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(all_summaries):
    GROUPS = ['Overall', '2-3op', '4-5op', '6+op']
    header = (f"\n{'Method':<16} {'Group':<10} {'N':>5} "
              f"{'ACC_cmd':>10} {'ACC_param':>10}")
    print(header)
    print('─' * 56)
    for method, summary in all_summaries.items():
        for g in GROUPS:
            row = summary[g]
            label = method if g == 'Overall' else ''
            print(f"{label:<16} {g:<10} {row['n']:>5} "
                  f"{row['acc_cmd']*100:>9.2f}% "
                  f"{row['acc_param']*100:>9.2f}%")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_encoder(name, load_fn, train_loader, test_loader):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    encoder = load_fn()
    decoder = build_fresh_decoder()

    ckpt_path = f'{OUT_DIR}/frozen_dec_{name.lower()}.pt'
    decoder = train_frozen_decoder(encoder, decoder, train_loader, ckpt_path)

    results = compute_acc_by_complexity(encoder, decoder, test_loader)
    summary = summarize_by_complexity(results)

    print(f"\n  {name} results:")
    for g, row in summary.items():
        print(f"    {g:<10}  n={row['n']:>4}  "
              f"ACC_cmd={row['acc_cmd']*100:.2f}%  "
              f"ACC_param={row['acc_param']*100:.2f}%")

    # Free GPU memory before next encoder
    del encoder, decoder
    torch.cuda.empty_cache()

    return summary, results


if __name__ == '__main__':
    # ── Datasets ───────────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds = CompletionDataset('train', DATA_ROOT, DEDUP_PATH, ctx_len=30)
    test_ds  = CompletionDataset('test',  DATA_ROOT, None,       ctx_len=30)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Train: {len(train_ds)} | Test: {len(test_ds)}")

    # ── Run each encoder ───────────────────────────────────────────────────────
    # Add 'ContrastCAD': load_contrastcad_encoder after it finishes training
    ENCODERS = {
        'AE':   load_ae_encoder,
        'MAE':  load_mae_encoder,
        'JEPA': load_jepa_encoder,
        # 'ContrastCAD': load_contrastcad_encoder,  # uncomment when ep400 done
    }

    all_summaries = {}
    all_raw       = {}

    for name, load_fn in ENCODERS.items():
        summary, raw = run_encoder(name, load_fn, train_loader, test_loader)
        all_summaries[name] = summary
        all_raw[name] = raw  # list of {n_ops, acc_cmd, acc_param}

    # ── Print comparison table ─────────────────────────────────────────────────
    print_table(all_summaries)

    # ── Save to Drive ──────────────────────────────────────────────────────────
    out_path = f'{OUT_DIR}/complexity_breakdown.json'
    with open(out_path, 'w') as f:
        # raw contains None values — replace with -1 for JSON serialization
        serializable_raw = {}
        for method, rlist in all_raw.items():
            serializable_raw[method] = [
                {k: (v if v is not None else -1) for k, v in r.items()}
                for r in rlist
            ]
        json.dump({
            'summaries': all_summaries,
            'raw':       serializable_raw,
            'config': {
                'frozen_epochs': FROZEN_EPOCHS,
                'dec_lr':        DEC_LR,
                'batch_size':    BATCH_SIZE,
            }
        }, f, indent=2)
    print(f"\n[saved] {out_path}")