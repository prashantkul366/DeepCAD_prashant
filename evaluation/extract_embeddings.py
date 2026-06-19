import os, sys, argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.configJEPA import ConfigJEPA
from model.jepa_encoder import JEPAEncoder
from dataset.cad_dataset import get_dataloader


def extract(ckpt_path, data_root, phase='test', batch_size=256, save_prefix='emb'):
    cfg           = ConfigJEPA('test')
    cfg.data_root = data_root
    cfg.batch_size = batch_size
    cfg.num_workers = 2

    encoder = JEPAEncoder(cfg).cuda()
    ckpt    = torch.load(ckpt_path, map_location='cuda')

    # Use EMA encoder — better representations for downstream
    encoder.load_state_dict(ckpt['ema_encoder'])
    encoder.eval()

    loader = get_dataloader(phase, cfg, shuffle=False)
    embs, ids = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Extracting'):
            commands = batch['command'].cuda()
            args     = batch['args'].cuda()
            z = encoder.get_pooled_embedding(commands, args)  # (N, d_model)
            embs.append(z.cpu().float().numpy())
            ids.extend(batch['id'])

    embs = np.concatenate(embs, axis=0)   # (total_samples, d_model)
    ids  = np.array(ids)

    np.save(f'{save_prefix}_embeddings.npy', embs)
    np.save(f'{save_prefix}_ids.npy', ids)
    print(f'Saved {len(embs)} embeddings → {save_prefix}_embeddings.npy')
    return embs, ids


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',       required=True)
    ap.add_argument('--data_root',  default='data')
    ap.add_argument('--phase',      default='test')
    ap.add_argument('--save_prefix',default='emb')
    args = ap.parse_args()
    extract(args.ckpt, args.data_root, args.phase, save_prefix=args.save_prefix)