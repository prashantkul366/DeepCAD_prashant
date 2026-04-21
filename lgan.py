import os
import numpy as np
import h5py
from utils import ensure_dir
from config import ConfigLGAN
from trainer import TrainerLatentWGAN
from dataset.lgan_dataset import get_dataloader


cfg = ConfigLGAN()
print("data path:", cfg.data_root)

print(f"[LGAN] exp_name: {cfg.exp_name} | ae_ckpt: {cfg.ae_ckpt}")
print(f"[LGAN] Mode: {'TEST/Generate' if cfg.test else 'TRAIN'}")

agent = TrainerLatentWGAN(cfg)

if not cfg.test:
    # load from checkpoint if provided
    if cfg.cont:
        agent.load_ckpt(cfg.ckpt)

    # create dataloader
    train_loader = get_dataloader(cfg)

    print(f"[LGAN Train] Latent vectors loaded | Batches: {len(train_loader)}")
    # print(f"[LGAN Train] Training for {cfg.n_epochs} epochs | Saving to: {cfg.exp_dir}")
    print(f"[LGAN Train] Training for {cfg.n_iters} iters | Saving to: {cfg.exp_dir}")


    agent.train(train_loader)
    print(f"[LGAN Train] Training complete.")
else:
    # load trained weights
    agent.load_ckpt(cfg.ckpt)
    print(f"[LGAN Test] Loaded GAN ckpt: {cfg.ckpt}")

    # run generator
    generated_shape_codes = agent.generate(cfg.n_samples)
    print(f"[LGAN Test] Generated shape: {generated_shape_codes.shape}")  # (n_samples, 256)
    print(f"[LGAN Test] mean: {generated_shape_codes.mean():.4f} | std: {generated_shape_codes.std():.4f}")


    # save generated z
    save_path = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}.h5".format(cfg.ckpt, cfg.n_samples))

    ensure_dir(os.path.dirname(save_path))
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("zs", shape=generated_shape_codes.shape, data=generated_shape_codes)
    
    print(f"[LGAN Test] Saved generated z to: {save_path}")
