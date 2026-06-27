# %%writefile /content/DeepCAD_prashant/trainer/cl_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import _get_padding_mask, _get_visibility_mask
from cadlib.macro import CMD_ARGS_MASK


class CADContrastiveLoss(nn.Module):
    def __init__(self, cfg, device, batch_size, temperature=0.07):
        super().__init__()
        self.n_commands   = cfg.n_commands
        self.args_dim     = cfg.args_dim + 1
        self.weights      = cfg.loss_weights
        self.device       = device
        self.temperature  = temperature
        self.batch_size   = batch_size
        self.cl_loss_type = getattr(cfg, 'cl_loss', 'infonce')
        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK))

    def forward(self, output):
        # tgt_commands: (N, S) batch-first — use seq_dim=-1
        tgt_commands = output["tgt_commands"]   # (N, S)
        tgt_args     = output["tgt_args"]       # (N, S, n_args)

        visibility_mask = _get_visibility_mask(tgt_commands, seq_dim=-1)  # (N, 1)
        padding_mask    = (
            _get_padding_mask(tgt_commands, seq_dim=-1, extended=True)    # (N, S)
            * visibility_mask                                              # (N, 1) → broadcasts
        )   # (N, S)

        command_logits = output["command_logits"]   # (N, S, n_commands)
        args_logits    = output["args_logits"]      # (N, S, n_args, args_dim)
        mask           = self.cmd_args_mask[tgt_commands.long()]  # (N, S, n_args)

        loss_cmd = F.cross_entropy(
            command_logits[padding_mask.bool()].reshape(-1, self.n_commands),
            tgt_commands[padding_mask.bool()].reshape(-1).long()
        )
        loss_args = F.cross_entropy(
            args_logits[mask.bool()].reshape(-1, self.args_dim),
            tgt_args[mask.bool()].reshape(-1).long() + 1
        )
        loss_cmd  = self.weights["loss_cmd_weight"]  * loss_cmd
        loss_args = self.weights["loss_args_weight"] * loss_args

        if self.cl_loss_type == 'infonce':
            logit, label     = self._info_nce_loss(output["proj_z1"], output["proj_z2"])
            loss_contrastive = self.weights["loss_cl_weight"] * \
                               nn.CrossEntropyLoss()(logit, label)
        else:
            loss_contrastive = self.weights["loss_cl_weight"] * \
                               self._contrastive_loss(output["proj_z1"], output["proj_z2"])

        return {
            "loss_cmd":         loss_cmd,
            "loss_args":        loss_args,
            "loss_contrastive": loss_contrastive,
        }

    def _info_nce_loss(self, f1, f2):
        f1 = F.normalize(f1.squeeze(0), dim=1)   # (N, d)
        f2 = F.normalize(f2.squeeze(0), dim=1)   # (N, d)
        N  = f1.size(0)

        features = torch.cat([f1, f2], dim=0)    # (2N, d)
        sim      = torch.matmul(features, features.T) / self.temperature  # (2N, 2N)

        labels = torch.zeros(2*N, 2*N, device=self.device)
        for i in range(N):
            labels[i,   N+i] = 1.0
            labels[N+i, i  ] = 1.0

        mask_eye = torch.eye(2*N, dtype=torch.bool, device=self.device)
        labels   = labels[~mask_eye].view(2*N, -1)
        sim      = sim[~mask_eye].view(2*N, -1)

        positives = sim[labels.bool()].view(2*N, -1)
        negatives = sim[~labels.bool()].view(2*N, -1)
        logits    = torch.cat([positives, negatives], dim=1)
        labels_out = torch.zeros(2*N, dtype=torch.long, device=self.device)
        return logits, labels_out

    def _contrastive_loss(self, z1, z2):
        z1 = F.normalize(z1.squeeze(0), dim=1)
        z2 = F.normalize(z2.squeeze(0), dim=1)
        N  = z1.size(0)

        labels = F.one_hot(torch.arange(N), N*2).float().cuda()
        masks  = F.one_hot(torch.arange(N), N).cuda()

        logits_aa = torch.matmul(z1, z1.T) / self.temperature - masks * 1e9
        logits_bb = torch.matmul(z2, z2.T) / self.temperature - masks * 1e9
        logits_ab = torch.matmul(z1, z2.T) / self.temperature
        logits_ba = torch.matmul(z2, z1.T) / self.temperature

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        return (loss_a + loss_b).mean()