import random
import numpy as np
import torch
from config.configJEPA_new import ConfigJEPA
from trainer.trainerJEPA_new import TrainerJEPA


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Not setting cudnn.deterministic=True — 20-30% slowdown not worth it
    # Masking randomness is intentionally stochastic per step — correct JEPA behavior


if __name__ == '__main__':
    cfg = ConfigJEPA('train')
    set_seed(cfg.seed)
    print(f"Seed: {cfg.seed}")
    trainer = TrainerJEPA(cfg)
    trainer.train()