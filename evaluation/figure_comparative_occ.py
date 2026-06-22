"""
figure_comparative_occ.py  — Run on Acer Windows
==================================================
Comparative qualitative figure: H-CAD-JEPA+Jitter vs ContrastCAD+RRE

For each query:
  Col 1:      Query shape (orange)
  Cols 2-4:   H-CAD-JEPA+Jitter top-3 (blue)  — geometrically correct
  Cols 5-7:   ContrastCAD+RRE top-3 (red)      — same class but wrong shape

Selection criteria (computed by find_comparison_cases.py on Colab):
  - Query from class-3 (6+ operations)
  - JEPA top-1 CD to query < threshold (correct geometric retrieval)
  - ContrastCAD top-1 CD to query >> JEPA top-1 (wrong geometric retrieval)

Step 1: Run find_comparison_cases.py on Colab → saves comparison_cases.json to Drive
Step 2: Download comparison_cases.json + needed h5 files to Acer
Step 3: Run this script on Acer to render solid CAD

Requires: pip install numpy-stl
"""

import os, sys, json, h5py, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

# ── CONFIGURE ─────────────────────────────────────────────────
REPO_ROOT   = r"C:\Users\prash\OneDrive\Desktop\CAD\DeepCAD_prashant"
DATA_ROOT   = r"C:\Users\prash\OneDrive\Desktop\CAD\cd_eval_data"
CASES_JSON  = r"C:\Users\prash\OneDrive\Desktop\CAD\cd_eval_output\comparison_cases.json"
STL_DIR     = r"C:\Users\prash\OneDrive\Desktop\CAD\cd_eval_output\stl_cache"
FIGS_DIR    = r"C:\Users\prash\OneDrive\Desktop\CAD\cd_eval_output\figures"
ELEV, AZIM  = 20, 225
DPI         = 250
N_QUERIES   = 3   # rows in figure
TOP_K       = 3   # columns per method
# ─────────────────────────────────────────────────────────────

os.makedirs(STL_DIR,  exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)
sys.path.insert(0, REPO_ROOT)
from cadlib.visualize import vec2CADsolid


def shape_to_stl(seq_id):
    """Reconstruct OCC → STL. Returns path or None."""
    safe     = seq_id.replace('/', '_')
    stl_path = os.path.join(STL_DIR, safe + '.stl')
    if os.path.exists(stl_path):
        return stl_path
    h5_path = os.path.join(DATA_ROOT, 'cad_vec',
                            seq_id.replace('/', os.sep) + '.h5')
    if not os.path.exists(h5_path):
        return None
    try:
        with h5py.File(h5_path, 'r') as f:
            vec = f['vec'][:].astype(float)
        shape = vec2CADsolid(vec)
        from OCC.Extend.DataExchange import write_stl_file
        write_stl_file(shape, stl_path)
        return stl_path
    except Exception as e:
        print(f"  STL failed {seq_id}: {e}")
        return None


def render_stl(ax, stl_path, base_color, title=''):
    """Render STL as shaded solid, normalized to unit cube."""
    from stl import mesh as stl_mesh
    m     = stl_mesh.Mesh.from_file(stl_path)
    verts = m.vectors.copy().astype(float)

    # Normalize to unit cube
    pts    = verts.reshape(-1, 3)
    center = pts.mean(axis=0)
    verts -= center
    scale  = np.abs(verts).max()
    if scale > 0:
        verts /= scale

    # Lighting
    v0, v1, v2 = verts[:, 0, :], verts[:, 1, :], verts[:, 2, :]
    normals = np.cross(v1 - v0, v2 - v0).astype(float)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms
    light      = np.array([0.4, 0.3, 1.0])
    light     /= np.linalg.norm(light)
    brightness = np.clip(normals @ light, 0.20, 1.0)

    base           = np.array(mcolors.to_rgb(base_color))
    face_colors    = np.clip(np.outer(brightness, base), 0, 1)
    face_rgba      = np.hstack([face_colors,
                                 np.full((len(face_colors), 1), 0.95)])

    poly = Poly3DCollection(verts)
    poly.set_facecolor(face_rgba)
    poly.set_edgecolor('none')
    ax.add_collection3d(poly)

    lim = 1.15
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_title(title, fontsize=8, pad=2,
                 fontweight='bold' if 'Query' in title else 'normal')
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_facecolor('white')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')


def render_placeholder(ax, title=''):
    """Placeholder when STL not available."""
    ax.text(0.5, 0.5, 'N/A\n(not in\nlocal data)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=8, color='#AAAAAA')
    ax.set_title(title, fontsize=8, pad=2)
    ax.set_axis_off()


def make_comparative_figure(cases):
    """
    Rows = queries, Cols = [Query | JEPA#1 #2 #3 | ContrastCAD#1 #2 #3]
    """
    n_rows = len(cases)
    n_cols = 1 + TOP_K + TOP_K

    fig = plt.figure(figsize=(n_cols * 2.6, n_rows * 2.8), facecolor='white')

    QUERY_COLOR   = '#C0431A'
    JEPA_COLORS   = ['#1560A8', '#2278C8', '#3A94E0']
    CC_COLORS     = ['#922B21', '#C0392B', '#E74C3C']

    col_titles = (['Query'] +
                  [f'JEPA #{j+1}' for j in range(TOP_K)] +
                  [f'ContrCAD #{j+1}' for j in range(TOP_K)])
    col_colors = ([QUERY_COLOR] + JEPA_COLORS[:TOP_K] + CC_COLORS[:TOP_K])

    for row, case in enumerate(cases):
        seq_ids = ([case['query_id']] +
                   case['jepa_top3'] +
                   case['cc_top3'])

        for col, (seq_id, color) in enumerate(zip(seq_ids, col_colors)):
            ax    = fig.add_subplot(n_rows, n_cols,
                                    row * n_cols + col + 1,
                                    projection='3d')
            title = col_titles[col] if row == 0 else ''

            stl_path = shape_to_stl(seq_id)
            if stl_path:
                render_stl(ax, stl_path, color, title=title)
            else:
                render_placeholder(ax, title=title)

        # Row annotation
        cd_jepa = case.get('jepa_cd', '—')
        cd_cc   = case.get('cc_cd', '—')
        fig.text(0.005,
                 1.0 - (row + 0.85) / n_rows,
                 f'CD(JEPA)={cd_jepa:.3f}\nCD(CC)={cd_cc:.3f}',
                 fontsize=7, va='top', ha='left',
                 color='#666666',
                 transform=fig.transFigure)

    # Column group labels
    fig.text(1.5 / n_cols, 1.01,
             '← H-CAD-JEPA+Jitter (ours) →',
             ha='center', fontsize=9.5, fontweight='bold', color='#1560A8')
    fig.text((1 + TOP_K + TOP_K/2) / n_cols, 1.01,
             '← ContrastCAD+RRE →',
             ha='center', fontsize=9.5, fontweight='bold', color='#922B21')

    plt.suptitle(
        'Qualitative Comparison: H-CAD-JEPA+Jitter vs ContrastCAD+RRE\n'
        'Queries from class-3 (6+ operations). '
        'Our method retrieves geometrically similar shapes; '
        'ContrastCAD retrieves wrong shapes despite same class label.',
        fontsize=9.5, fontweight='bold', y=1.05
    )
    plt.tight_layout(pad=0.3, h_pad=0.1, w_pad=0.0)

    out = os.path.join(FIGS_DIR, 'fig_comparative_ours_vs_contrastcad.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved → {out}")


def main():
    print("Comparative Retrieval Figure: Ours vs ContrastCAD")
    print("=" * 50)

    if not os.path.exists(CASES_JSON):
        print(f"ERROR: {CASES_JSON} not found.")
        print("Run find_comparison_cases.py on Colab first.")
        print("Then download comparison_cases.json to Acer.")
        return

    with open(CASES_JSON) as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} comparison cases")
    cases = cases[:N_QUERIES]

    # Reconstruct all needed STLs
    needed_ids = set()
    for case in cases:
        needed_ids.add(case['query_id'])
        needed_ids.update(case['jepa_top3'])
        needed_ids.update(case['cc_top3'])
    print(f"Reconstructing {len(needed_ids)} shapes...")

    for seq_id in needed_ids:
        p = shape_to_stl(seq_id)
        status = '✓' if p else '✗ (not in local data)'
        print(f"  {seq_id}: {status}")

    make_comparative_figure(cases)


if __name__ == '__main__':
    main()