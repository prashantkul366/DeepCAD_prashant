"""
train_decoder.py
================
Trains a CAD decoder on top of a FROZEN H-CAD-JEPA encoder.
Architecture: frozen JEPAEncoder → trainable Bottleneck → trainable Decoder
Metrics: ACC_cmd, ACC_param (standard in DeepCAD / SkexGen / ContrastCAD)

Usage (Colab):
  !cd /content/DeepCAD_prashant && python train_decoder.py \
      --jepa_ckpt /content/drive/MyDrive/jepa_experiments/hcadjepa_jitter/model/ckpt_ep0400.pt \
      --data_root /content/deepcad_data \
      --proj_dir  /content/drive/MyDrive/jepa_experiments \
      --exp_name  decoder_jitter \
      --nr_epochs 300 --lr 1e-4 --batch_size 256

To compare baselines, rerun with different --jepa_ckpt:
  hcadjepa_jitter      ep400  (best overall, with group emb)
  hcadjepa_hierarchical ep400 (best no-jitter, with group emb)
  hcadjepa_mae         latest (MAE baseline)
  contrastcad          (use their embeddings differently — skip for now)
"""

import os, sys, argparse, json, time
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.jepa_encoder import JEPAEncoder
from model.autoencoder import Decoder, Bottleneck
from dataset.cad_dataset import get_dataloader
from model.model_utils import _get_padding_mask, _get_group_mask
from trainer.loss import CADLoss
from cadlib.macro import (
    EOS_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX,
    CMD_ARGS_MASK,
    N_ARGS_EXT, N_ARGS_PLANE, N_ARGS_TRANS, N_ARGS_EXT_PARAM,
)

# ── Constants ─────────────────────────────────────────────────────────────────
PAD_VAL = -1  # padding sentinel for both commands and args


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    # Encoder
    p.add_argument('--jepa_ckpt',        type=str,   required=True,
                   help='Path to JEPA checkpoint (.pt)')
    p.add_argument('--jepa_key',         type=str,   default='ema_encoder',
                   help='Key inside checkpoint dict for encoder weights')
    p.add_argument('--use_group_emb',    action='store_true', default=False,
                   help='Use group (block-ID) embedding in encoder')
    # Decoder arch — match DeepCAD AE defaults
    p.add_argument('--dim_z',            type=int,   default=256,
                   help='Bottleneck dimension (DeepCAD default: 256)')
    p.add_argument('--n_layers_decode',  type=int,   default=4,
                   help='Number of decoder transformer layers')
    # Data
    p.add_argument('--data_root',        type=str,   default='/content/deepcad_data')
    p.add_argument('--proj_dir',         type=str,   default='/content/drive/MyDrive/jepa_experiments')
    p.add_argument('--exp_name',         type=str,   default='decoder_jepa')
    # Training
    p.add_argument('--nr_epochs',        type=int,   default=300)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--batch_size',       type=int,   default=256)
    p.add_argument('--num_workers',      type=int,   default=4)
    p.add_argument('--grad_clip',        type=float, default=1.0)
    p.add_argument('--warmup_epochs',    type=int,   default=10)
    p.add_argument('--save_every',       type=int,   default=20)
    p.add_argument('--val_every',        type=int,   default=5)
    return p.parse_args()


# ── Config namespaces ─────────────────────────────────────────────────────────
def make_encoder_cfg(args):
    return argparse.Namespace(
        d_model=256, n_layers=4, n_heads=8, dim_feedforward=512,
        dropout=0.0,
        use_group_emb=args.use_group_emb,
        max_num_groups=30, max_total_len=60, max_n_loops=6,
        max_n_curves=15, n_commands=6, n_args=16, args_dim=256,
        max_n_ext=10, augment=False, jitter_aug=False,
        batch_size=args.batch_size, num_workers=args.num_workers,
        data_root=args.data_root, use_cls=False,
    )


def make_decoder_cfg(args):
    return argparse.Namespace(
        d_model=256,
        dim_z=args.dim_z,
        n_heads=8,
        dim_feedforward=512,
        dropout=0.1,
        n_layers_decode=args.n_layers_decode,
        max_total_len=60,
        n_commands=6,
        n_args=16,
        args_dim=256,        # FCN uses args_dim+1 = 257 internally
    )


# ── Loss (exact match to DeepCAD's CADLoss) ───────────────────────────────────
def make_loss_fn(device):
    """Returns CADLoss instance configured identically to DeepCAD AE training."""
    loss_cfg = argparse.Namespace(
        n_commands=6,
        args_dim=256,
        loss_weights={"loss_cmd_weight": 1.0, "loss_args_weight": 2.0},
    )
    return CADLoss(loss_cfg).to(device)


def apply_loss(loss_fn, cmd_logits, arg_logits, gt_cmd, gt_args):
    """
    Wrap decoder outputs into CADLoss's expected dict format and compute loss.
    cmd_logits: (N, S, n_commands) — batch first
    arg_logits: (N, S, n_args, args_dim+1) — batch first
    gt_cmd:     (N, S)
    gt_args:    (N, S, n_args)
    """
    output = {
        "command_logits": cmd_logits,
        "args_logits":    arg_logits,
        "tgt_commands":   gt_cmd,
        "tgt_args":       gt_args,
    }
    loss_dict = loss_fn(output)
    return sum(loss_dict.values())


# ── Accuracy (exact match to DeepCAD's TrainerAE.evaluate()) ─────────────────
@torch.no_grad()
def compute_acc(cmd_logits, arg_logits, gt_cmd, gt_args):
    """
    Replicates DeepCAD TrainerAE.evaluate() protocol.
    Reports:
      acc_cmd   — command type accuracy at non-padding positions
      acc_param — average of per-command-type arg accuracies
                  (line, arc, circle, plane, trans, extent)
    These numbers are directly comparable to published DeepCAD/SkexGen tables.
    """
    # Decode predictions
    cmd_pred = cmd_logits.argmax(dim=-1)                    # (N, S)
    out_args = arg_logits.argmax(dim=-1) - 1               # (N, S, n_args)

    gt_cmd_np  = gt_cmd.long().detach().cpu().numpy()
    gt_args_np = gt_args.long().detach().cpu().numpy()
    out_args_np= out_args.detach().cpu().numpy()
    cmd_pred_np= cmd_pred.detach().cpu().numpy()

    # ACC_cmd: over all non-padding positions
    import numpy as np
    valid = (gt_cmd_np >= 0) & (gt_cmd_np <= EOS_IDX)
    acc_cmd = float((cmd_pred_np[valid] == gt_cmd_np[valid]).mean()) if valid.any() else 0.0

    # ACC_param: per-command-type, specific arg slots only (matches DeepCAD eval)
    args_comp = (gt_args_np == out_args_np).astype(int)

    ext_pos    = np.where(gt_cmd_np == EXT_IDX)
    line_pos   = np.where(gt_cmd_np == LINE_IDX)
    arc_pos    = np.where(gt_cmd_np == ARC_IDX)
    circle_pos = np.where(gt_cmd_np == CIRCLE_IDX)

    accs = []
    if len(ext_pos[0]) > 0:
        ext_comp = args_comp[ext_pos][:, -N_ARGS_EXT:]
        accs.append(('plane',  float(np.mean(ext_comp[:, :N_ARGS_PLANE]))))
        accs.append(('trans',  float(np.mean(ext_comp[:, N_ARGS_PLANE:N_ARGS_PLANE+N_ARGS_TRANS]))))
        accs.append(('extent', float(np.mean(ext_comp[:, -N_ARGS_EXT_PARAM:]))))
    if len(line_pos[0]) > 0:
        accs.append(('line',   float(np.mean(args_comp[line_pos][:, :2]))))
    if len(arc_pos[0]) > 0:
        accs.append(('arc',    float(np.mean(args_comp[arc_pos][:, :4]))))
    if len(circle_pos[0]) > 0:
        accs.append(('circle', float(np.mean(args_comp[circle_pos][:, [0, 1, 4]]))))

    acc_param = float(np.mean([v for _, v in accs])) if accs else 0.0
    per_type  = {k: v for k, v in accs}

    return acc_cmd, acc_param, per_type


# ── Encoder forward (frozen) ──────────────────────────────────────────────────
@torch.no_grad()
def encode(encoder, cmd, args_seq, use_group_emb=False):
    """
    Returns z: (1, N, d_model) — mean-pooled encoder output, keepdim.
    cmd:      (N, S)  batch-first
    args_seq: (N, S, n_args) batch-first
    """
    mem, _ = encoder(cmd, args_seq)                 # mem: (S, N, d_model) seq-first
    cmd_sf = cmd.permute(1, 0)                      # (S, N)
    pad    = _get_padding_mask(cmd_sf, seq_dim=0)   # (S, N, 1)  1=valid token
    z = (mem.float() * pad).sum(0, keepdim=True) / \
        pad.sum(0, keepdim=True).clamp(min=1)       # (1, N, d_model)
    return z


# ── Warmup LR scheduler ───────────────────────────────────────────────────────
def get_lr(optimizer, epoch, warmup_epochs, max_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    # cosine decay handled by scheduler — just return current
    return optimizer.param_groups[0]['lr']


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    exp_dir   = os.path.join(args.proj_dir, args.exp_name)
    model_dir = os.path.join(exp_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    print(f"Experiment: {exp_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Frozen encoder ────────────────────────────────────────────────────
    enc_cfg = make_encoder_cfg(args)
    encoder = JEPAEncoder(enc_cfg).to(device)
    raw = torch.load(args.jepa_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(raw[args.jepa_key])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"Frozen encoder: {enc_params/1e6:.2f}M params")
    print(f"  checkpoint: {args.jepa_ckpt}")
    print(f"  key:        {args.jepa_key}")
    print(f"  group_emb:  {args.use_group_emb}")

    # ── Trainable: Bottleneck + Decoder ──────────────────────────────────
    dec_cfg    = make_decoder_cfg(args)
    bottleneck = Bottleneck(dec_cfg).to(device)
    decoder    = Decoder(dec_cfg).to(device)
    loss_fn    = make_loss_fn(device)

    bn_params  = sum(p.numel() for p in bottleneck.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Trainable bottleneck: {bn_params/1e6:.3f}M params")
    print(f"Trainable decoder:    {dec_params/1e6:.2f}M params")
    print(f"dim_z={args.dim_z}, n_layers_decode={args.n_layers_decode}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader = get_dataloader('train', enc_cfg, shuffle=True)
    val_loader   = get_dataloader('test',  enc_cfg, shuffle=False)
    print(f"Train: {len(train_loader)} batches | Test: {len(val_loader)} batches")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    trainable_params = list(bottleneck.parameters()) + list(decoder.parameters())
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    # Cosine decay after warmup
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.nr_epochs - args.warmup_epochs,
        eta_min=1e-6,
    )

    # ── Training ─────────────────────────────────────────────────────────
    log = []
    best_cmd_acc = 0.0
    t0 = time.time()

    for epoch in range(args.nr_epochs):

        # LR warmup
        if epoch < args.warmup_epochs:
            lr_scale = (epoch + 1) / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale

        bottleneck.train()
        decoder.train()
        train_loss = 0.0

        for batch in train_loader:
            cmd      = batch['command'].to(device)   # (N, S)
            args_seq = batch['args'].to(device)      # (N, S, 16)

            # Frozen encoder forward
            with torch.no_grad():
                z_enc = encode(encoder, cmd, args_seq, args.use_group_emb)  # (1, N, 256)

            # Trainable bottleneck + decoder
            z   = bottleneck(z_enc)    # (1, N, dim_z)
            # Decoder returns (cmd_logits, arg_logits) both seq-first
            cmd_logits, arg_logits = decoder(z)
            # Permute to batch-first for loss
            cmd_logits = cmd_logits.permute(1, 0, 2)      # (N, S, 6)
            arg_logits = arg_logits.permute(1, 0, 2, 3)   # (N, S, 16, 257)

            loss = apply_loss(loss_fn, cmd_logits, arg_logits, cmd, args_seq)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Step scheduler after warmup
        if epoch >= args.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # ── Validation ────────────────────────────────────────────────────
        if epoch % args.val_every == 0 or epoch == args.nr_epochs - 1:
            bottleneck.eval()
            decoder.eval()
            cmd_accs, param_accs = [], []

            with torch.no_grad():
                for batch in val_loader:
                    cmd      = batch['command'].to(device)
                    args_seq = batch['args'].to(device)

                    z_enc      = encode(encoder, cmd, args_seq, args.use_group_emb)
                    z          = bottleneck(z_enc)
                    cmd_lg, arg_lg = decoder(z)
                    cmd_lg = cmd_lg.permute(1, 0, 2)
                    arg_lg = arg_lg.permute(1, 0, 2, 3)

                    ca, pa, _ = compute_acc(cmd_lg, arg_lg, cmd, args_seq)
                    cmd_accs.append(ca)
                    param_accs.append(pa)

            acc_cmd   = np.mean(cmd_accs)   * 100
            acc_param = np.mean(param_accs) * 100
            elapsed   = (time.time() - t0) / 60

            print(f"ep={epoch:04d}  loss={train_loss:.4f}  "
                  f"ACC_cmd={acc_cmd:.2f}%  ACC_param={acc_param:.2f}%  "
                  f"lr={current_lr:.2e}  [{elapsed:.1f}min]")

            row = dict(epoch=epoch, loss=round(train_loss, 5),
                       acc_cmd=round(acc_cmd, 3), acc_param=round(acc_param, 3))
            log.append(row)

            if acc_cmd > best_cmd_acc:
                best_cmd_acc = acc_cmd
                torch.save({
                    'epoch':      epoch,
                    'bottleneck': bottleneck.state_dict(),
                    'decoder':    decoder.state_dict(),
                    'ACC_cmd':    acc_cmd,
                    'ACC_param':  acc_param,
                    'jepa_ckpt':  args.jepa_ckpt,
                }, f'{model_dir}/best.pt')
                print(f"  ★ new best ACC_cmd={acc_cmd:.2f}%")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch':      epoch,
                'bottleneck': bottleneck.state_dict(),
                'decoder':    decoder.state_dict(),
            }, f'{model_dir}/ckpt_ep{epoch:04d}.pt')

    # ── Final save ────────────────────────────────────────────────────────
    torch.save({
        'epoch':      args.nr_epochs - 1,
        'bottleneck': bottleneck.state_dict(),
        'decoder':    decoder.state_dict(),
    }, f'{model_dir}/latest.pt')

    with open(f'{exp_dir}/train_log.json', 'w') as f:
        json.dump(log, f, indent=2)

    total_min = (time.time() - t0) / 60
    print(f"\nTraining complete in {total_min:.1f} min")
    print(f"Best ACC_cmd: {best_cmd_acc:.2f}%")
    print(f"Checkpoints: {model_dir}")
    print(f"Log:         {exp_dir}/train_log.json")


if __name__ == '__main__':
    main()