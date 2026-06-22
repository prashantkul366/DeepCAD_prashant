"""
extract_baseline_embeddings.py  — Run on Colab (A100)
======================================================
Extracts retrieval embeddings from SkexGen and HNC-CAD.

SkexGen:  cmd + param + ext encoders, pre-VQ continuous embedding
          (4 code tokens × 128d) × 3 encoders = 1536d → L2 normalize

HNC-CAD:  solid VQ codebook lookup — no re-encoding needed
          solid.pkl has pre-computed VQ code index per shape
          codebook embedding: (10000, 256) → look up by index → 256d

Both are aligned to our 8052 test IDs. Missing samples → zero vector.

Usage:
  python extract_baseline_embeddings.py
  Saves to {PROJ}/eval_results/skexgen_embs.npy
           {PROJ}/eval_results/hnccad_embs.npy
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import normalize

# ── PATHS ─────────────────────────────────────────────────────
PROJ      = '/content/drive/MyDrive/jepa_experiments'
SKEX_ROOT = '/content/drive/MyDrive/Prashant/SkexGen_prashant'
HNC_ROOT  = '/content/drive/MyDrive/Prashant/hnc-cad_prashant'
SPLIT_JSON= '/content/deepcad_data/train_val_test_split.json'
OUT_DIR   = f'{PROJ}/eval_results'
os.makedirs(OUT_DIR, exist_ok=True)

# SkexGen checkpoint paths
SKEX_CKPT = {
    'cmd':   f'{SKEX_ROOT}/proj_log/exp_sketch/cmdenc_epoch_300.pt',
    'param': f'{SKEX_ROOT}/proj_log/exp_sketch/paramenc_epoch_300.pt',
    'ext':   f'{SKEX_ROOT}/proj_log/exp_extrude/extenc_epoch_200.pt',
}
SKEX_DATA = f'{SKEX_ROOT}/data/cad_data/test.pkl'

# HNC-CAD paths
HNC_SOLID_ENC  = f'{HNC_ROOT}/proj_log/solid/enc_epoch_250.pt'
HNC_SOLID_PKL  = f'{HNC_ROOT}/proj_log/solid/solid.pkl'
# ─────────────────────────────────────────────────────────────

# SkexGen constants (from dataset.py)
EXTRA_PAD = 1
COORD_PAD = 4
PIX_PAD   = 4
EXT_PAD   = 1
CMD_PAD   = 3
EXT_FLAGS_PER_SE = [1,1,2,2,2,3,3,3,3,3,3,3,3,3,4,5,6,6,7]  # 19 per extrude


# ══════════════════════════════════════════════════════════════
#  SkexGen Encoder definitions (copied from their model/encoder.py)
#  Minimal version for inference only
# ══════════════════════════════════════════════════════════════

sys.path.insert(0, SKEX_ROOT)
from model.encoder import CMDEncoder, PARAMEncoder, EXTEncoder


def build_skexgen_config(cmd_ckpt, param_ckpt, ext_ckpt):
    """Infer model config from checkpoint shapes."""
    # All encoders share same transformer config
    in_proj = cmd_ckpt['encoder.layers.0.self_attn.in_proj_weight']
    embed_dim = in_proj.shape[1]      # 256
    n_layers  = sum(1 for k in cmd_ckpt if 'encoder.layers' in k and 'in_proj_weight' in k)

    config = {
        'embed_dim':    embed_dim,
        'dropout_rate': 0.0,           # inference: no dropout
        'num_heads':    8,             # 256 / 32 per head
        'hidden_dim':   embed_dim * 2,
        'num_layers':   n_layers,
    }

    cmd_num_code   = cmd_ckpt['vq_vae._embedding.weight'].shape[0]
    param_num_code = param_ckpt['vq_vae._embedding.weight'].shape[0]
    ext_num_code   = ext_ckpt['vq_vae._embedding.weight'].shape[0]

    print(f"  embed_dim={embed_dim}, n_layers={n_layers}")
    print(f"  cmd_num_code={cmd_num_code}, param_num_code={param_num_code}, ext_num_code={ext_num_code}")

    return config, cmd_num_code, param_num_code, ext_num_code


# def load_skexgen_encoders():
#     """Load all three SkexGen encoders."""
#     cmd_ckpt   = torch.load(SKEX_CKPT['cmd'],   map_location='cpu', weights_only=False)
#     param_ckpt = torch.load(SKEX_CKPT['param'], map_location='cpu', weights_only=False)
#     ext_ckpt   = torch.load(SKEX_CKPT['ext'],   map_location='cpu', weights_only=False)

#     config, cmd_nc, param_nc, ext_nc = build_skexgen_config(cmd_ckpt, param_ckpt, ext_ckpt)

#     cmd_enc = CMDEncoder(config, code_len=4, max_len=80, num_code=cmd_nc).cuda()
#     cmd_enc.load_state_dict(cmd_ckpt)
#     cmd_enc.eval()

#     param_enc = PARAMEncoder(config, quantization_bits=6,
#                               num_code=param_nc, code_len=4, max_len=80).cuda()
#     param_enc.load_state_dict(param_ckpt)
#     param_enc.eval()

#     ext_enc = EXTEncoder(config, quantization_bits=6,
#                           num_code=ext_nc, code_len=4, max_len=80).cuda()
#     ext_enc.load_state_dict(ext_ckpt)
#     ext_enc.eval()

#     print("  ✓ All SkexGen encoders loaded")
#     return cmd_enc, param_enc, ext_enc

def load_skexgen_encoders():
    cmd_ckpt   = torch.load(SKEX_CKPT['cmd'],   map_location='cpu', weights_only=False)
    param_ckpt = torch.load(SKEX_CKPT['param'], map_location='cpu', weights_only=False)
    ext_ckpt   = torch.load(SKEX_CKPT['ext'],   map_location='cpu', weights_only=False)

    config, cmd_nc, param_nc, ext_nc = build_skexgen_config(cmd_ckpt, param_ckpt, ext_ckpt)

    # Infer max_len from checkpoint pos_embed shape
    # pos_embed size = max_len + code_len (4)
    CODE_LEN = 4
    cmd_maxlen   = int(cmd_ckpt['pos_embed.position'].shape[0])   - CODE_LEN
    param_maxlen = int(param_ckpt['pos_embed.position'].shape[0]) - CODE_LEN
    ext_maxlen   = int(ext_ckpt['pos_embed.position'].shape[0])   - CODE_LEN
    print(f"  Inferred max_len: cmd={cmd_maxlen}, param={param_maxlen}, ext={ext_maxlen}")

    cmd_enc = CMDEncoder(config, code_len=CODE_LEN,
                         max_len=cmd_maxlen, num_code=cmd_nc).cuda()
    cmd_enc.load_state_dict(cmd_ckpt)
    cmd_enc.eval()

    param_enc = PARAMEncoder(config, quantization_bits=6,
                              num_code=param_nc, code_len=CODE_LEN,
                              max_len=param_maxlen).cuda()
    param_enc.load_state_dict(param_ckpt)
    param_enc.eval()

    ext_enc = EXTEncoder(config, quantization_bits=6,
                          num_code=ext_nc, code_len=CODE_LEN,
                          max_len=ext_maxlen).cuda()
    ext_enc.load_state_dict(ext_ckpt)
    ext_enc.eval()

    print("  ✓ All SkexGen encoders loaded")
    return cmd_enc, param_enc, ext_enc

# ── Pre-VQ embedding extractors ──────────────────────────────

@torch.no_grad()
def get_cmd_embedding(model, cmd_seq_np):
    """
    cmd_seq_np: 1D numpy array of command tokens (already padded with EXTRA_PAD)
    Returns: (code_len * codebook_dim,) numpy array
    """
    bs = 1
    seq_len = len(cmd_seq_np)
    code_len = model.code_len

    cmd_t = torch.tensor(cmd_seq_np).unsqueeze(0).cuda()  # (1, seq_len)
    mask  = torch.zeros(bs, seq_len, dtype=torch.bool).cuda()

    c_embeds = model.c_embed(cmd_t.flatten()).view(bs, seq_len, -1)
    embeddings = c_embeds.transpose(0, 1)
    z_embed = model.const_embed(
        torch.arange(0, code_len).long().cuda()
    ).unsqueeze(1).repeat(1, bs, 1)
    embed_input = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = model.pos_embed(embed_input)

    mask_full = torch.cat([torch.zeros(bs, code_len, dtype=torch.bool).cuda(), mask], dim=1)
    outputs   = model.encoder(src=encoder_input, src_key_padding_mask=mask_full)
    z_encoded = outputs[0:code_len]                        # (code_len, 1, embed_dim)

    z_pre_vq = model.down(z_encoded)                       # (code_len, 1, codebook_dim)
    return z_pre_vq.squeeze(1).flatten().cpu().numpy()      # (code_len * codebook_dim,)


@torch.no_grad()
def get_param_embedding(model, pix_seq_np, xy_seq_np):
    """
    pix_seq_np: 1D numpy array of pixel tokens
    xy_seq_np:  2D numpy array of xy coords (seq_len, 2)
    Returns: (code_len * codebook_dim,) numpy array
    """
    bs = 1
    seq_len = len(pix_seq_np)
    code_len = model.code_len

    pix_t = torch.tensor(pix_seq_np).unsqueeze(0).cuda()   # (1, seq_len)
    xy_t  = torch.tensor(xy_seq_np).unsqueeze(0).cuda()    # (1, seq_len, 2)
    mask  = torch.zeros(bs, seq_len, dtype=torch.bool).cuda()

    coord_embed = model.coord_embed_x(xy_t[..., 0]) + model.coord_embed_y(xy_t[..., 1])
    pixel_embed = model.pixel_embed(pix_t)
    embeddings  = (coord_embed + pixel_embed).transpose(0, 1)

    z_embed = model.const_embed(
        torch.arange(0, code_len).long().cuda()
    ).unsqueeze(1).repeat(1, bs, 1)
    embed_input   = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = model.pos_embed(embed_input)

    mask_full = torch.cat([torch.zeros(bs, code_len, dtype=torch.bool).cuda(), mask], dim=1)
    outputs   = model.encoder(src=encoder_input, src_key_padding_mask=mask_full)
    z_encoded = outputs[0:code_len]

    z_pre_vq = model.down(z_encoded)
    return z_pre_vq.squeeze(1).flatten().cpu().numpy()


@torch.no_grad()
def get_ext_embedding(model, ext_seq_np, flag_seq_np):
    """
    ext_seq_np:  1D numpy array of extrude tokens
    flag_seq_np: 1D numpy array of flag tokens (hardcoded per extrude)
    Returns: (code_len * codebook_dim,) numpy array
    """
    bs = 1
    seq_len = len(ext_seq_np)
    code_len = model.code_len

    ext_t  = torch.tensor(ext_seq_np).unsqueeze(0).cuda()   # (1, seq_len)
    flag_t = torch.tensor(flag_seq_np).unsqueeze(0).cuda()  # (1, seq_len)
    mask   = torch.zeros(bs, seq_len, dtype=torch.bool).cuda()

    ext_embeds  = model.ext_embed(ext_t)
    flag_embeds = model.flag_embed(flag_t)
    embeddings  = (ext_embeds + flag_embeds).transpose(0, 1)

    z_embed = model.const_embed(
        torch.arange(0, code_len).long().cuda()
    ).unsqueeze(1).repeat(1, bs, 1)
    embed_input   = torch.cat([z_embed, embeddings], dim=0)
    encoder_input = model.pos_embed(embed_input)

    mask_full = torch.cat([torch.zeros(bs, code_len, dtype=torch.bool).cuda(), mask], dim=1)
    outputs   = model.encoder(src=encoder_input, src_key_padding_mask=mask_full)
    z_encoded = outputs[0:code_len]

    z_pre_vq = model.down(z_encoded)
    return z_pre_vq.squeeze(1).flatten().cpu().numpy()


def prepare_sample(sample):
    """Prepare SkexGen sample into encoder inputs."""
    pix_tokens = sample['se_pix']
    xy_tokens  = sample['se_xy']
    cmd_tokens = sample['se_cmd']
    ext_tokens = sample['se_ext']
    n_se       = sample['num_se']

    # CMD
    cmds = np.hstack(cmd_tokens) + EXTRA_PAD
    cmds = np.concatenate([cmds, np.zeros(1, dtype=int)])

    # PARAM
    pixs = np.hstack(pix_tokens) + EXTRA_PAD
    pixs = np.concatenate([pixs, np.zeros(1, dtype=int)])
    xys  = np.vstack(xy_tokens) + EXTRA_PAD
    xys  = np.concatenate([xys,  np.zeros((1, 2), dtype=int)])

    # EXT
    exts  = np.hstack(ext_tokens) + EXTRA_PAD
    exts  = np.concatenate([exts, np.zeros(1, dtype=int)])
    flags = np.hstack(EXT_FLAGS_PER_SE * n_se)
    flags = np.concatenate([flags, np.zeros(1, dtype=int)])

    return cmds, pixs, xys, exts, flags


# ══════════════════════════════════════════════════════════════
#  Extract SkexGen embeddings
# ══════════════════════════════════════════════════════════════

def extract_skexgen(test_ids):
    print("\n── SkexGen Embedding Extraction ──────────────────────")

    # Load test data → {name: sample}
    with open(SKEX_DATA, 'rb') as f:
        skex_test = pickle.load(f)
    skex_dict = {s['name']: s for s in skex_test}
    print(f"  SkexGen test samples: {len(skex_dict)}/8052")

    cmd_enc, param_enc, ext_enc = load_skexgen_encoders()

    # Infer embedding dim
    dummy_code_dim = cmd_enc.codebook_dim
    total_dim      = 3 * 4 * dummy_code_dim   # 3 encoders × 4 codes × codebook_dim
    print(f"  Embedding dim per encoder: {4 * dummy_code_dim}  |  Total: {total_dim}")

    embeddings = np.zeros((len(test_ids), total_dim), dtype=np.float32)
    missing    = 0

    for i, test_id in enumerate(tqdm(test_ids, desc='  SkexGen')):
        if test_id not in skex_dict:
            missing += 1
            continue

        sample = skex_dict[test_id]
        try:
            cmds, pixs, xys, exts, flags = prepare_sample(sample)
            e_cmd   = get_cmd_embedding(cmd_enc,     cmds)
            e_param = get_param_embedding(param_enc, pixs, xys)
            e_ext   = get_ext_embedding(ext_enc,     exts, flags)
            embeddings[i] = np.concatenate([e_cmd, e_param, e_ext])
        except Exception as ex:
            missing += 1

    print(f"  Missing/failed: {missing}/8052")
    embs_norm = normalize(embeddings)
    return embs_norm


# ══════════════════════════════════════════════════════════════
#  Extract HNC-CAD embeddings (codebook lookup)
# ══════════════════════════════════════════════════════════════

def extract_hnccad(test_ids):
    print("\n── HNC-CAD Embedding Extraction (codebook lookup) ───")

    # Load solid VQ codes per shape
    with open(HNC_SOLID_PKL, 'rb') as f:
        solid_data = pickle.load(f)
    content = solid_data['content']   # {8-digit-id: int code_idx}
    print(f"  HNC solid content entries: {len(content)}")

    # Load codebook weights (no model instantiation needed)
    enc_ckpt       = torch.load(HNC_SOLID_ENC, map_location='cpu', weights_only=False)
    codebook_w     = enc_ckpt['codebook._embedding.weight'].numpy()  # (10000, 256)
    print(f"  Codebook shape: {codebook_w.shape}")

    embeddings = np.zeros((len(test_ids), 256), dtype=np.float32)
    missing    = 0

    for i, test_id in enumerate(test_ids):
        digit_id = test_id.split('/')[-1]   # '0000/00009254' → '00009254'
        if digit_id in content:
            code_idx       = int(content[digit_id])
            embeddings[i]  = codebook_w[code_idx]
        else:
            missing += 1

    print(f"  Missing/not in HNC data: {missing}/8052")
    embs_norm = normalize(embeddings)
    return embs_norm


# ══════════════════════════════════════════════════════════════
#  Quick retrieval eval
# ══════════════════════════════════════════════════════════════

def quick_eval(embs, labels, name, k=10):
    from sklearn.metrics import average_precision_score
    N = len(embs)
    aps, r1s = [], []
    for i in range(N):
        sims = embs @ embs[i]; sims[i] = -1
        ranked = np.argsort(sims)[::-1]
        r1s.append(int(labels[ranked[0]] == labels[i]))
        top_k = ranked[:k]
        rel   = (labels[top_k] == labels[i]).astype(int)
        if rel.sum() > 0:
            aps.append(average_precision_score(rel, sims[top_k]))
    mAP = float(np.mean(aps)) * 100
    R1  = float(np.mean(r1s)) * 100
    print(f"  {name}: mAP@10={mAP:.2f}%  R@1={R1:.2f}%")
    return mAP, R1


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    import numpy as np

    print("=" * 60)
    print("Baseline Embedding Extraction: SkexGen + HNC-CAD")
    print("=" * 60)

    with open(SPLIT_JSON) as f:
        test_ids = json.load(f)['test']
    labels = np.load('/content/test_labels.npy')
    print(f"Test sequences: {len(test_ids)}")

    results = {}

    # ── SkexGen ───────────────────────────────────────────────
    skex_embs = extract_skexgen(test_ids)
    skex_path = f'{OUT_DIR}/skexgen_embs.npy'
    np.save(skex_path, skex_embs)
    print(f"  Saved → {skex_path}")
    mAP, R1 = quick_eval(skex_embs, labels, 'SkexGen')
    results['SkexGen'] = {'mAP@10': round(mAP, 2), 'R@1': round(R1, 2),
                           'emb_dim': skex_embs.shape[1]}

    # ── HNC-CAD ───────────────────────────────────────────────
    hnc_embs  = extract_hnccad(test_ids)
    hnc_path  = f'{OUT_DIR}/hnccad_embs.npy'
    np.save(hnc_path, hnc_embs)
    print(f"  Saved → {hnc_path}")
    mAP, R1 = quick_eval(hnc_embs, labels, 'HNC-CAD')
    results['HNC-CAD'] = {'mAP@10': round(mAP, 2), 'R@1': round(R1, 2),
                           'emb_dim': hnc_embs.shape[1]}

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("── RESULTS ─────────────────────────────────────────────")
    print(f"{'Method':<15} {'mAP@10':>10} {'R@1':>10} {'dim':>8}")
    print("─" * 48)
    for name, r in results.items():
        print(f"{name:<15} {r['mAP@10']:>9.2f}%  {r['R@1']:>9.2f}%  {r['emb_dim']:>7}")

    # Save results
    import json as json_mod
    with open(f'{OUT_DIR}/baseline_results.json', 'w') as f:
        json_mod.dump(results, f, indent=2)
    print(f"\nSaved → {OUT_DIR}/baseline_results.json")
    print("\nTo include in eval_all_metrics.py, add to MODELS dict:")
    print("  'SkexGen':  (None, True, 'skexgen')   # uses pre-saved .npy")
    print("  'HNC-CAD':  (None, True, 'hnccad')    # uses pre-saved .npy")


if __name__ == '__main__':
    main()