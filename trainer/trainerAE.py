# %%writefile /content/DeepCAD_prashant/trainer/trainerAE.py
import torch
import torch.optim as optim
from tqdm import tqdm
from model import CADTransformer
from .base import BaseTrainer
from .loss import CADLoss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *
import os
import shutil


class TrainerAE(BaseTrainer):
    def build_net(self, cfg):
        self.net = CADTransformer(cfg).cuda()

    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        self.loss_func = CADLoss(self.cfg).cuda()

    def forward(self, data):
        commands = data['command'].cuda()
        args     = data['args'].cuda()
        self.net._debug = (self.clock.step < 3)
        if self.clock.step == 0:
            print(f"[Forward] commands: {commands.shape} | args: {args.shape}")
        outputs   = self.net(commands, args)
        loss_dict = self.loss_func(outputs)
        return outputs, loss_dict

    def encode(self, data, is_batch=False):
        commands = data['command'].cuda()
        args     = data['args'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args     = args.unsqueeze(0)
        z = self.net(commands, args, encode_mode=True)
        return z

    def decode(self, z):
        outputs = self.net(None, None, z=z, return_tgt=False)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)
        out_args    = torch.argmax(torch.softmax(outputs['args_logits'],    dim=-1), dim=-1) - 1
        if refill_pad:
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1
        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def save_ckpt(self, epoch, tag=None):
        """Save checkpoint. epoch must be an int. tag overrides filename."""
        name = tag if tag is not None else f'ckpt_ep{epoch:04d}'
        path = os.path.join(self.cfg.model_dir, f'{name}.pt')
        torch.save({
            'epoch':     epoch,
            'net':       self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        # Drive backup
        drive_dir = getattr(self.cfg, 'drive_backup_dir', None)
        if drive_dir and os.path.exists(drive_dir):
            shutil.copy(path, os.path.join(drive_dir, f'{name}.pt'))
        return path

    def load_ckpt(self, tag='latest'):
        path = os.path.join(self.cfg.model_dir, f'{tag}.pt')
        ckpt = torch.load(path, map_location='cuda', weights_only=False)
        self.net.load_state_dict(ckpt['net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt['epoch']

    def evaluate(self, test_loader):
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))

        all_ext_args_comp    = []
        all_line_args_comp   = []
        all_arc_args_comp    = []
        all_circle_args_comp = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                commands = data['command'].cuda()
                args     = data['args'].cuda()
                outputs  = self.net(commands, args)
                out_args = torch.argmax(
                    torch.softmax(outputs['args_logits'], dim=-1), dim=-1
                ) - 1
                out_args = out_args.long().detach().cpu().numpy()

            gt_commands = commands.long().detach().cpu().numpy()
            gt_args     = args.long().detach().cpu().numpy()

            ext_pos    = np.where(gt_commands == EXT_IDX)
            line_pos   = np.where(gt_commands == LINE_IDX)
            arc_pos    = np.where(gt_commands == ARC_IDX)
            circle_pos = np.where(gt_commands == CIRCLE_IDX)

            args_comp = (gt_args == out_args).astype(int)
            all_ext_args_comp.append(args_comp[ext_pos][:, -N_ARGS_EXT:])
            all_line_args_comp.append(args_comp[line_pos][:, :2])
            all_arc_args_comp.append(args_comp[arc_pos][:, :4])
            all_circle_args_comp.append(args_comp[circle_pos][:, [0, 1, 4]])

        all_ext_args_comp = np.concatenate(all_ext_args_comp, axis=0)
        sket_plane_acc = np.mean(all_ext_args_comp[:, :N_ARGS_PLANE])
        sket_trans_acc = np.mean(
            all_ext_args_comp[:, N_ARGS_PLANE:N_ARGS_PLANE + N_ARGS_TRANS]
        )
        extent_one_acc = np.mean(all_ext_args_comp[:, -N_ARGS_EXT_PARAM])
        line_acc       = np.mean(np.concatenate(all_line_args_comp,   axis=0))
        arc_acc        = np.mean(np.concatenate(all_arc_args_comp,    axis=0))
        circle_acc     = np.mean(np.concatenate(all_circle_args_comp, axis=0))

        print(f"[Eval Epoch {self.clock.epoch}] "
              f"line: {line_acc:.4f} | arc: {arc_acc:.4f} | "
              f"circle: {circle_acc:.4f} | plane: {sket_plane_acc:.4f} | "
              f"trans: {sket_trans_acc:.4f} | extent: {extent_one_acc:.4f}")