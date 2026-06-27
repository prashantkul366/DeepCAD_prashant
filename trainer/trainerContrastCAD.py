# %%writefile /content/DeepCAD_prashant/trainer/trainerContrastCAD.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import shutil

from model.autoencoder import CADTransformer, Encoder
from trainer.cl_loss import CADContrastiveLoss
from trainer.scheduler import GradualWarmupScheduler
from dataset.cad_dataset_new import get_dataloader
from cadlib.macro import *


class ProjectionHead(nn.Module):
    """Multi-layer projection head — ported from ContrastCAD."""
    def __init__(self, d_model, n_layers=2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(d_model, d_model))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
        self.proj = nn.Sequential(*layers)

    def forward(self, z):
        return self.proj(z)


class TrainerContrastCAD:
    """
    ContrastCAD: same encoder as AE + projection head + dropout views + InfoNCE.
    Reconstruction loss retained alongside contrastive loss (as per their paper).
    Eval embedding: pre-projection encoder output (_z), NOT the projected z.
    """
    def __init__(self, cfg):
        self.cfg         = cfg
        self.global_step = 0
        self._build_models()
        self._build_optimizer()

    def _build_models(self):
        cfg = self.cfg

        # Full CADTransformer — encoder + bottleneck + decoder
        # Same as DeepCAD AE, identical architecture
        self.net      = CADTransformer(cfg).cuda()
        self.proj     = ProjectionHead(cfg.dim_z, n_layers=cfg.n_phead_layers).cuda()
        self.dropout  = nn.Dropout(p=cfg.latent_dropout)
        self.tanh     = nn.Tanh()

        n_net  = sum(p.numel() for p in self.net.parameters())
        n_proj = sum(p.numel() for p in self.proj.parameters())
        print(f"[ContrastCAD] Net    : {n_net/1e6:.2f}M")
        print(f"[ContrastCAD] ProjHead: {n_proj/1e6:.2f}M")

    def _build_optimizer(self):
        params = (list(self.net.parameters()) +
                  list(self.proj.parameters()))
        self.optimizer = optim.Adam(params, lr=self.cfg.lr)
        self.scheduler = GradualWarmupScheduler(
            self.optimizer, 1.0, self.cfg.warmup_step
        )

    def _forward(self, data):
        commands = data['command'].cuda()
        args     = data['args'].cuda()

        from model.model_utils import _make_seq_first, _make_batch_first
        commands_sf, args_sf = _make_seq_first(commands, args)

        # Encode
        _z = self.net.encoder(commands_sf, args_sf)   # (1, N, d_model) pre-bottleneck

        # Project
        z       = self.proj(_z)                        # (1, N, d_model)
        proj_z1 = self.dropout(z)                      # view 1
        proj_z2 = self.dropout(z)                      # view 2
        z_tanh  = self.tanh(z)                         # for decoder

        # Decode
        out_logits = self.net.decoder(z_tanh)
        out_logits = _make_batch_first(*out_logits)

        return {
            "command_logits": out_logits[0],
            "args_logits":    out_logits[1],
            "proj_z1":        proj_z1,
            "proj_z2":        proj_z2,
            "tgt_commands":   commands,
            "tgt_args":       args,
            "_z":             _z,   # pre-projection, for eval
        }

    @torch.no_grad()
    def get_encoder(self):
        """Returns encoder for downstream eval."""
        return self.net.encoder

    def _train_step(self, data, loss_func):
        self.optimizer.zero_grad()
        outputs   = self._forward(data)
        loss_dict = loss_func(outputs)
        loss      = sum(loss_dict.values())
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.net.parameters()) +
            list(self.proj.parameters()),
            self.cfg.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1
        return loss_dict

    def _backup(self, path, name):
        drive_dir = getattr(self.cfg, 'drive_backup_dir', None)
        if drive_dir and os.path.exists(drive_dir):
            shutil.copy(path, os.path.join(drive_dir, name))

    def save_ckpt(self, epoch, tag=None):
        name = tag if tag is not None else f'ckpt_ep{epoch:04d}'
        path = os.path.join(self.cfg.model_dir, f'{name}.pt')
        torch.save({
            'epoch':       epoch,
            'global_step': self.global_step,
            'net':         self.net.state_dict(),
            'proj':        self.proj.state_dict(),
            'optimizer':   self.optimizer.state_dict(),
        }, path)
        self._backup(path, f'{name}.pt')
        return path

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location='cuda', weights_only=False)
        self.net.load_state_dict(ckpt['net'])
        self.proj.load_state_dict(ckpt['proj'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt.get('global_step', 0)
        return ckpt['epoch']

    def train(self):
        cfg          = self.cfg
        train_loader = get_dataloader('train', cfg)
        val_loader   = get_dataloader('validation', cfg)
        loss_func    = CADContrastiveLoss(
            cfg, device='cuda',
            batch_size=cfg.batch_size,
            temperature=cfg.temperature
        ).cuda()

        log_path    = os.path.join(cfg.log_dir, 'losses.txt')
        start_epoch = 0

        if cfg.cont:
            ckpt_path   = os.path.join(cfg.model_dir, f'{cfg.ckpt}.pt')
            start_epoch = self.load_ckpt(ckpt_path) + 1
            print(f'[ContrastCAD] Resumed from epoch {start_epoch}')

        for epoch in range(start_epoch, cfg.nr_epochs):
            self.net.train()
            self.proj.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f'Ep{epoch:04d}', leave=False)

            for data in pbar:
                loss_dict = self._train_step(data, loss_func)
                total     = sum(v.item() for v in loss_dict.values())
                epoch_losses.append(total)
                pbar.set_postfix({k: f'{v.item():.4f}'
                                  for k, v in loss_dict.items()})

            mean_loss = np.mean(epoch_losses)
            lr_now    = self.optimizer.param_groups[0]['lr']
            msg       = (f'ep={epoch:04d}  train={mean_loss:.5f}  '
                         f'lr={lr_now:.2e}  step={self.global_step}')
            print(msg)
            with open(log_path, 'a') as f:
                f.write(msg + '\n')

            if epoch % cfg.save_frequency == 0 or epoch == cfg.nr_epochs - 1:
                self.save_ckpt(epoch)
            self.save_ckpt(epoch, tag='latest')

        print('[ContrastCAD] Training complete.')