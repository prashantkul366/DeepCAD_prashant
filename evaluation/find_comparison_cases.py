"""
find_comparison_cases.py  — Run on Colab
==========================================
Finds queries where JEPA is geometrically correct
but ContrastCAD is geometrically wrong.

Uses CD matrix to define "geometric correctness":
  Good retrieval: top-1 CD < low_threshold
  Bad retrieval:  top-1 CD > high_threshold

Saves comparison_cases.json to Drive for rendering on Acer.
"""

import os, sys, json, h5py, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import argparse
from sklearn.preprocessing import normalize

sys.path.insert(0, '/content/DeepCAD_prashant')
from model.jepa_encoder import JEPAEncoder
from dataset.cad_dataset import get_dataloader

PROJ      = '/content/drive/MyDrive/jepa_experiments'
DATA_ROOT = '/content/deepcad_data'
OUT_DIR   = f'{PROJ}/eval_results'

# ── Load everything ───────────────────────────────────────────

def load_all():
    # Labels and test IDs
    with open(f'{DATA_ROOT}/train_val_test_split.json') as f:
        test_ids = json.load(f)['test']
    labels = np.load('/content/test_labels.npy')

    # CD matrix (class-3 only, 480×480)
    cd_matrix = np.load(f'{OUT_DIR}/cd_matrix.npy')
    valid_ids = np.load(f'{OUT_DIR}/valid_ids.npy', allow_pickle=True)

    test_id_to_idx = {id_: i for i, id_ in enumerate(test_ids)}
    valid_idxs = np.array([test_id_to_idx[id_] for id_ in valid_ids])

    # JEPA+Jitter embeddings
    cfg = argparse.Namespace(
        d_model=256, n_layers=4, n_heads=8, dim_feedforward=512,
        dropout=0.0, use_group_emb=True,
        max_num_groups=30, max_total_len=60, max_n_loops=6,
        max_n_curves=15, n_commands=6, n_args=16, args_dim=256,
        max_n_ext=10, augment=False, jitter_aug=False,
        batch_size=256, num_workers=4, data_root=DATA_ROOT, use_cls=False,
    )
    enc = JEPAEncoder(cfg).cuda(); enc.eval()
    ckpt = torch.load(f'{PROJ}/hcadjepa_jitter/model/ckpt_ep0400.pt',
                      map_location='cuda', weights_only=False)
    enc.load_state_dict(ckpt['ema_encoder'])

    loader = get_dataloader('test', cfg, shuffle=False)
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            z = enc.get_pooled_embedding(batch['command'].cuda(), batch['args'].cuda())
            all_embs.append(z.cpu().numpy())
    all_embs = normalize(np.concatenate(all_embs))

    # ContrastCAD embeddings
    with h5py.File(f'{PROJ}/contrastcad_rre_embeddings.h5', 'r') as f:
        cc_embs = normalize(f['test_zs'][:].astype(np.float32))

    return (test_ids, labels, cd_matrix, valid_ids,
            valid_idxs, all_embs, cc_embs)


def find_comparison_cases(test_ids, labels, cd_matrix, valid_ids,
                           valid_idxs, jepa_embs, cc_embs,
                           n_cases=6, k=3):
    """
    Find queries where:
      JEPA top-1 has LOW CD (correct geometric retrieval)
      ContrastCAD top-1 has HIGH CD (wrong geometric retrieval)
    """
    N_cd = len(cd_matrix)

    # For each valid class-3 query:
    # 1. Compute JEPA top-3 in full test set
    # 2. Compute CC top-3 in full test set
    # 3. Check which top-retrieved shapes are in our CD-valid set

    cases = []
    scores = []

    for qi in range(N_cd):
        global_qi = valid_idxs[qi]  # index into full test set

        # JEPA top-K
        sims_jepa = jepa_embs @ jepa_embs[global_qi]
        sims_jepa[global_qi] = -1
        jepa_topk_global = np.argsort(sims_jepa)[::-1][:20]  # top-20 pool

        # CC top-K
        sims_cc = cc_embs @ cc_embs[global_qi]
        sims_cc[global_qi] = -1
        cc_topk_global = np.argsort(sims_cc)[::-1][:20]

        # Filter to those in valid_ids (have CD values)
        global_to_local = {int(v): i for i, v in enumerate(valid_idxs)}
        jepa_local = [global_to_local[int(g)] for g in jepa_topk_global
                      if int(g) in global_to_local][:k]
        cc_local   = [global_to_local[int(g)] for g in cc_topk_global
                      if int(g) in global_to_local][:k]

        if len(jepa_local) < k or len(cc_local) < k:
            continue

        # CD of top-1 for each method
        cd_jepa = cd_matrix[qi, jepa_local[0]] if jepa_local else 999
        cd_cc   = cd_matrix[qi, cc_local[0]]   if cc_local   else 0

        # Score: high CD difference favors our method
        advantage = cd_cc - cd_jepa
        scores.append((advantage, qi, jepa_local, cc_local, cd_jepa, cd_cc))

    # Sort by advantage (ours better = high score)
    scores.sort(reverse=True)

    for advantage, qi, jepa_local, cc_local, cd_jepa, cd_cc in scores[:n_cases]:
        global_qi = valid_idxs[qi]
        case = {
            'query_id':   test_ids[global_qi],
            'query_label': int(labels[global_qi]),
            'jepa_top3':  [test_ids[valid_idxs[j]] for j in jepa_local[:k]],
            'cc_top3':    [test_ids[valid_idxs[j]] for j in cc_local[:k]],
            'jepa_cd':    float(cd_jepa),
            'cc_cd':      float(cd_cc),
            'advantage':  float(advantage),
        }
        cases.append(case)
        print(f"  Case: {case['query_id'][:15]}  "
              f"JEPA_CD={cd_jepa:.3f}  CC_CD={cd_cc:.3f}  "
              f"Advantage={advantage:.3f}")

    return cases


def main():
    print("Finding comparison cases...")
    (test_ids, labels, cd_matrix, valid_ids,
     valid_idxs, jepa_embs, cc_embs) = load_all()

    print(f"  CD matrix: {cd_matrix.shape}")
    print(f"  Valid class-3: {len(valid_ids)}")

    cases = find_comparison_cases(
        test_ids, labels, cd_matrix, valid_ids,
        valid_idxs, jepa_embs, cc_embs,
        n_cases=6, k=3
    )

    out_path = f'{PROJ}/eval_results/comparison_cases.json'
    with open(out_path, 'w') as f:
        json.dump(cases, f, indent=2)
    print(f"\n✓ Saved {len(cases)} cases → {out_path}")

    print("\nAll shape IDs needed on Acer:")
    needed = set()
    for c in cases:
        needed.add(c['query_id'])
        needed.update(c['jepa_top3'])
        needed.update(c['cc_top3'])
    for sid in sorted(needed):
        print(f"  {sid}")

    print(f"\nTotal shapes needed: {len(needed)}")
    print("Download comparison_cases.json + these h5 files to Acer")
    print("  (h5 files from /content/deepcad_data/cad_vec/)")

    # Save list of needed IDs for easy download
    needed_list = sorted(needed)
    with open(f'{PROJ}/eval_results/needed_h5_ids.json', 'w') as f:
        json.dump(needed_list, f, indent=2)
    print(f"✓ Saved needed IDs → {PROJ}/eval_results/needed_h5_ids.json")

    # Helper: zip the needed h5 files for download
    print("\nCreating zip of needed h5 files...")
    import zipfile
    zip_path = f'{PROJ}/eval_results/comparison_h5.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for sid in needed_list:
            h5_path = f'{DATA_ROOT}/cad_vec/{sid}.h5'
            if os.path.exists(h5_path):
                zf.write(h5_path, sid + '.h5')
                print(f"  ✓ {sid}")
            else:
                print(f"  ✗ {sid} (not found)")
    print(f"✓ Zip saved → {zip_path}")
    print("Download this zip to Acer and extract into cd_eval_data/cad_vec/")


if __name__ == '__main__':
    main()