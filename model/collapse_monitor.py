import torch
import torch.nn.functional as F
from cadlib.macro import EOS_IDX


class CollapseMonitor:
    """
    Monitors effective rank of EMA encoder representations.
    Computed once per epoch on a fixed validation sample.
    Triggers VICReg regularization warning if rank drops below threshold.
    """

    def __init__(self, d_model, rank_threshold=0.70):
        self._d_model    = d_model
        self._threshold  = rank_threshold
        self._last_rank  = 1.0

    def compute_rank(self, ema_encoder, monitor_cmds, monitor_args):
        """
        ema_encoder  : JEPAEncoder (EMA copy) — eval mode, no grad
        monitor_cmds : (N, 60) long — fixed val sample, on CPU
        monitor_args : (N, 60, 16) long — fixed val sample, on CPU
        Returns rank fraction (float in [0, 1])
        """
        ema_encoder.eval()
        with torch.no_grad():
            cmds = monitor_cmds.cuda()
            args = monitor_args.cuda()
            # Use get_pooled_embedding — same path as downstream eval
            z = ema_encoder.get_pooled_embedding(cmds, args)  # (N, d_model)
            z = z.float()

        # PCA-based effective rank: dims needed to explain 99% of variance
        # q = min of (N-1, d_model) — required by pca_lowrank
        q = min(z.shape[0] - 1, self._d_model)
        _, s, _ = torch.pca_lowrank(z, q=q, niter=4)
        s2       = s ** 2
        cumvar   = torch.cumsum(s2 / s2.sum(), dim=0)
        rank     = int((cumvar < 0.99).sum().item()) + 1
        self._last_rank = rank / self._d_model
        return self._last_rank

    def is_collapsing(self):
        return self._last_rank < self._threshold

    @property
    def last_rank(self):
        return self._last_rank