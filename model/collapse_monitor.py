# %%writefile /content/DeepCAD_prashant/model/collapse_monitor.py
import torch
from cadlib.macro import EOS_IDX


class CollapseMonitor:
    """
    Monitors effective rank of EMA encoder representations.
    Computed once per epoch on a fixed validation sample.
    """

    def __init__(self, d_model, rank_threshold=0.70):
        self._d_model   = d_model
        self._threshold = rank_threshold
        self._last_rank = 1.0

    def compute_rank(self, ema_encoder, monitor_cmds, monitor_args):
        """
        ema_encoder  : JEPAEncoder (EMA copy)
        monitor_cmds : (N, 60) long — on CPU
        monitor_args : (N, 60, 16) long — on CPU
        Returns rank fraction float in [0, 1]
        """
        ema_encoder.eval()
        with torch.no_grad():
            z = ema_encoder.get_pooled_embedding(
                monitor_cmds.cuda(), monitor_args.cuda()
            ).float()   # (N, d_model)

        q        = min(z.shape[0] - 1, self._d_model)
        _, s, _  = torch.pca_lowrank(z, q=q, niter=4)
        s2       = s ** 2
        cumvar   = torch.cumsum(s2 / s2.sum(), dim=0)
        rank     = int((cumvar < 0.99).sum().item()) + 1
        # self._last_rank = rank / self._d_model
        self._last_rank = rank / q   # fraction of measurable components
        return self._last_rank

    def is_collapsing(self):
        return self._last_rank < self._threshold

    @property
    def last_rank(self):
        return self._last_rank