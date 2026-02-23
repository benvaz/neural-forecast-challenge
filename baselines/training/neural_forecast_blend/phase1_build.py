"""
BEI D1 CODABENCH VALIDATOR - PHASE 1: Component Isolation (FIXED)
=================================================================
Generates 10 submissions, each routing 100% of blend_t to ONE component.
All other components (affi, priv) stay frozen at V70 best.

bei_d1_CB = 5 * MSR_CB - K
K = affi_d1 + affi_priv + bei_priv1 + bei_priv2 = 144,773

KEY INSIGHT: model.py always runs all 4 components (nb, mb, raw, koop).
d1_blend_t (10, 4) controls per-timestep mixing. Columns = [nb, mb, raw, koop].
To isolate component X: set blend_t[:, X] = 1.0, all others = 0.

To test a DIFFERENT config of a component (e.g. MB with α=5M):
replace that component's weight keys, keep blend_t routing 100% to it.
"""
import numpy as np
import os
import shutil
import subprocess
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

# ================================================================
# CONFIGURE THESE PATHS
# ================================================================
SUB_DIR = os.path.expanduser("~/hackathon_2026/submissions/neural_forecast_blend_v1")
DATA_DIR = os.path.expanduser("~/hackathon_2026/data")
DEST = "/mnt/c/Users/benva/Downloads"
BEST_BEI = os.path.join(SUB_DIR, "weights_beignet.npz")

FROZEN_FILES = ["model.py", "weights_affi.npz", "requirements.txt"]
AFFI_MLP_FILES = [f for f in os.listdir(SUB_DIR)
                   if f.startswith("mlp_affi_d1") and f.endswith(".npz")]

# ================================================================
# BLEND_T COLUMN ORDER (from model.py predict() call order)
# Components: ridge(None), nb(0), mb(1), raw(2), mlp(None), koop(3)
# Ridge and MLP return None for bei d1 → not in preds
# ================================================================
COL_NB   = 0
COL_MB   = 1
COL_RAW  = 2
COL_KOOP = 3
N_COMPONENTS = 4

def blend_t_single(col):
    """Create blend_t that routes 100% to one component at all timesteps."""
    bt = np.zeros((10, N_COMPONENTS), np.float32)
    bt[:, col] = 1.0
    return bt

# ================================================================
# LOAD TRAINING DATA
# ================================================================
print("Loading beignet training data...")
data = np.load(os.path.join(DATA_DIR, "train_data_beignet.npz"))["arr_0"].astype(np.float32)
C = 89
F = 9
x_all = data[:, :10, :, :]
y_all = data[:, 10:, :, 0]
x0 = x_all[:, :, :, 0]

corr = np.corrcoef(x0[:, 5, :].T)
np.fill_diagonal(corr, 0)
print(f"  Data: {len(data)} trials, {C} channels, {F} bands")

# ================================================================
# _stats_basic — MUST MATCH model.py EXACTLY
# ================================================================
def _stats_basic(x10):
    """6 features per (channel, band): mean, std, slope, last, range, curvature."""
    mu = x10.mean(axis=1, keepdims=True)
    sd = x10.std(axis=1, keepdims=True) + 1e-6
    slope = (x10[:, -1:] - x10[:, :1]) / 9.0
    last = x10[:, -1:]
    rng = x10.max(axis=1, keepdims=True) - x10.min(axis=1, keepdims=True)
    mid = x10.shape[1] // 2
    curv = x10[:, mid+1:mid+2] - 2.0 * x10[:, mid:mid+1] + x10[:, mid-1:mid]
    return np.concatenate([mu, sd, slope, last, rng, curv], axis=1).astype(np.float32)

# ================================================================
# TRAINING FUNCTIONS
# ================================================================
def train_nb(K_nb, alpha_nb):
    """Train NeighborRidge (NB). Uses band 0 raw timesteps as features."""
    def _fit(c):
        nbs = np.argsort(np.abs(corr[c]))[::-1][:K_nb]
        Xc = np.concatenate([x0[:, :, c]] + [x0[:, :, nb] for nb in nbs], axis=1)
        reg = Ridge(alpha=alpha_nb)
        reg.fit(Xc, y_all[:, :, c])
        return c, nbs, reg.coef_, reg.intercept_

    results = Parallel(n_jobs=-1)(delayed(_fit)(c) for c in range(C))
    idx, coefs, inters = [], [], []
    for c, nbs, coef, inter in sorted(results):
        idx.append(nbs)
        coefs.append(coef)
        inters.append(inter)
    return np.array(idx, np.int32), np.array(coefs, np.float32), np.array(inters, np.float32)


def train_mb(K_mb, alpha_mb):
    """Train MultiBand (MB). Uses _stats_basic features — 6 per channel per band."""
    def _fit(c):
        nbs = np.argsort(np.abs(corr[c]))[::-1][:K_mb]
        chans = [c] + list(nbs)
        feats = []
        for ch in chans:
            for b in range(F):
                feats.append(_stats_basic(x_all[:, :, ch, b]))
        X = np.concatenate(feats, axis=1)
        reg = Ridge(alpha=alpha_mb)
        reg.fit(X, y_all[:, :, c])
        return c, nbs, reg.coef_, reg.intercept_

    results = Parallel(n_jobs=-1)(delayed(_fit)(c) for c in range(C))
    idx, coefs, inters = [], [], []
    for c, nbs, coef, inter in sorted(results):
        idx.append(nbs)
        coefs.append(coef)
        inters.append(inter)
    return np.array(idx, np.int32), np.array(coefs, np.float32), np.array(inters, np.float32)


def train_raw(K_raw, alpha_raw):
    """Train RAW. Uses raw timesteps for all bands (no _stats_basic)."""
    def _fit(c):
        nbs = np.argsort(np.abs(corr[c]))[::-1][:K_raw]
        chans = [c] + list(nbs)
        feats = [x_all[:, :10, ch, b] for ch in chans for b in range(F)]
        X = np.concatenate(feats, axis=1)
        reg = Ridge(alpha=alpha_raw)
        reg.fit(X, y_all[:, :, c])
        return c, nbs, reg.coef_, reg.intercept_

    results = Parallel(n_jobs=-1)(delayed(_fit)(c) for c in range(C))
    idx, coefs, inters = [], [], []
    for c, nbs, coef, inter in sorted(results):
        idx.append(nbs)
        coefs.append(coef)
        inters.append(inter)
    return np.array(idx, np.int32), np.array(coefs, np.float32), np.array(inters, np.float32)


# ================================================================
# HELPER: Build a submission zip
# ================================================================
def make_zip(name, bei_weights_path, desc):
    tmp = f"/tmp/sub_{name}"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for fn in FROZEN_FILES:
        src = os.path.join(SUB_DIR, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tmp, fn))
    for mf in AFFI_MLP_FILES:
        src = os.path.join(SUB_DIR, mf)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tmp, mf))
    shutil.copy(bei_weights_path, os.path.join(tmp, "weights_beignet.npz"))
    zip_path = os.path.join(DEST, f"val_{name}.zip")
    subprocess.run(
        ["zip", "-q", "-j", zip_path] +
        [os.path.join(tmp, f) for f in os.listdir(tmp)],
        check=True
    )
    sz = os.path.getsize(zip_path) / 1024 / 1024
    print(f"  {name}: {sz:.1f} MB - {desc}")
    shutil.rmtree(tmp)


def build_submission(name, desc, blend_t, extra_keys=None):
    """Build a submission using base weights + custom blend_t + optional key overrides."""
    w = dict(np.load(BEST_BEI, allow_pickle=False))
    w["d1_blend_t"] = blend_t
    if extra_keys:
        # Remove conflicting NB keys before adding new ones
        if any("nb3" in k or "nb5" in k or "nb10" in k for k in extra_keys):
            for old_k in list(w.keys()):
                if any(tag in old_k for tag in ["nb3_", "nb5_", "nb10_"]) and old_k.startswith("d1_"):
                    del w[old_k]
        w.update(extra_keys)
    fn = f"/tmp/weights_bei_{name}.npz"
    np.savez_compressed(fn, **w)
    make_zip(name, fn, desc)


# ================================================================
# BUILD PHASE 1 SUBMISSIONS
# ================================================================
print("\n=== TRAINING & BUILDING PHASE 1 ===\n")

# --- S01: NB only, K=3, α=1 ---
print("S01: NB K=3 α=1...")
nb3_idx, nb3_coef, nb3_inter = train_nb(3, 1.0)
build_submission("p1_s01_nb_K3_a1", "NB only K=3 α=1",
    blend_t_single(COL_NB),
    {"d1_nb3_idx": nb3_idx, "d1_nb3_coefs": nb3_coef, "d1_nb3_intercepts": nb3_inter})

# --- S02: NB only, K=5, α=10 (current config) ---
print("S02: NB K=5 α=10 (current)...")
nb5_idx, nb5_coef, nb5_inter = train_nb(5, 10.0)
build_submission("p1_s02_nb_K5_a10", "NB only K=5 α=10",
    blend_t_single(COL_NB),
    {"d1_nb5_idx": nb5_idx, "d1_nb5_coefs": nb5_coef, "d1_nb5_intercepts": nb5_inter})

# --- S03: MB only, K=24, α=500k (current) ---
print("S03: MB K=24 α=500k (current)...")
mb24_500k_idx, mb24_500k_coef, mb24_500k_inter = train_mb(24, 500_000)
build_submission("p1_s03_mb_K24_a500k", "MB only K=24 α=500k",
    blend_t_single(COL_MB),
    {"d1_mb_idx": mb24_500k_idx, "d1_mb_coefs": mb24_500k_coef, "d1_mb_intercepts": mb24_500k_inter})

# --- S04: MB only, K=24, α=5M ---
print("S04: MB K=24 α=5M...")
mb24_5M_idx, mb24_5M_coef, mb24_5M_inter = train_mb(24, 5_000_000)
build_submission("p1_s04_mb_K24_a5M", "MB only K=24 α=5M",
    blend_t_single(COL_MB),
    {"d1_mb_idx": mb24_5M_idx, "d1_mb_coefs": mb24_5M_coef, "d1_mb_intercepts": mb24_5M_inter})

# --- S05: MB only, K=32, α=5M ---
print("S05: MB K=32 α=5M...")
mb32_5M_idx, mb32_5M_coef, mb32_5M_inter = train_mb(32, 5_000_000)
build_submission("p1_s05_mb_K32_a5M", "MB only K=32 α=5M",
    blend_t_single(COL_MB),
    {"d1_mb_idx": mb32_5M_idx, "d1_mb_coefs": mb32_5M_coef, "d1_mb_intercepts": mb32_5M_inter})

# --- S06: RAW only, K=10, α=5M ---
print("S06: RAW K=10 α=5M...")
raw10_5M_idx, raw10_5M_coef, raw10_5M_inter = train_raw(10, 5_000_000)
build_submission("p1_s06_raw_K10_a5M", "RAW only K=10 α=5M",
    blend_t_single(COL_RAW),
    {"d1_raw_idx": raw10_5M_idx, "d1_raw_coefs": raw10_5M_coef, "d1_raw_intercepts": raw10_5M_inter})

# --- S07: RAW only, K=8, α=3M ---
print("S07: RAW K=8 α=3M...")
raw8_3M_idx, raw8_3M_coef, raw8_3M_inter = train_raw(8, 3_000_000)
build_submission("p1_s07_raw_K8_a3M", "RAW only K=8 α=3M",
    blend_t_single(COL_RAW),
    {"d1_raw_idx": raw8_3M_idx, "d1_raw_coefs": raw8_3M_coef, "d1_raw_intercepts": raw8_3M_inter})

# --- S08: Koopman only (base config) ---
print("S08: Koopman only...")
build_submission("p1_s08_koop_only", "Koopman only (base config)",
    blend_t_single(COL_KOOP))

# --- S09: V70 baseline (all 4 components, original blend_t) ---
# This verifies K is correct — should reproduce MSR ≈ 40,301
print("S09: V70 baseline (verification)...")
w_base = dict(np.load(BEST_BEI, allow_pickle=False))
fn_v70 = "/tmp/weights_bei_p1_s09.npz"
np.savez_compressed(fn_v70, **w_base)
make_zip("p1_s09_v70_verify", fn_v70, "V70 baseline verification")

# --- S10: NB only, K=10, α=10 (wider) ---
print("S10: NB K=10 α=10...")
nb10_idx, nb10_coef, nb10_inter = train_nb(10, 10.0)
build_submission("p1_s10_nb_K10_a10", "NB only K=10 α=10",
    blend_t_single(COL_NB),
    {"d1_nb10_idx": nb10_idx, "d1_nb10_coefs": nb10_coef, "d1_nb10_intercepts": nb10_inter})


# ================================================================
# VERIFY: Check a sample submission's keys match model.py expectations
# ================================================================
print("\n=== VERIFICATION ===")
w_check = dict(np.load("/tmp/weights_bei_p1_s01_nb_K3_a1.npz", allow_pickle=False))
print("S01 d1 keys:")
for k in sorted(w_check.keys()):
    if k.startswith("d1_"):
        print(f"  {k}: shape={w_check[k].shape}")

print(f"\nblend_t check (should be 1.0 in col {COL_NB} only):")
print(w_check["d1_blend_t"])


print("""
╔══════════════════════════════════════════════════════════════════════╗
║                 PHASE 1: 10 SUBMISSIONS READY (FIXED)               ║
╠══════════════════════════════════════════════════════════════════════╣
║  Sub  │ Component      │ Config              │ blend_t col          ║
║───────┼────────────────┼─────────────────────┼──────────────────────║
║  S01  │ NB only        │ K=3, α=1            │ col 0 (nb)          ║
║  S02  │ NB only        │ K=5, α=10 (current) │ col 0 (nb)          ║
║  S03  │ MB only        │ K=24, α=500k (curr) │ col 1 (mb)          ║
║  S04  │ MB only        │ K=24, α=5M          │ col 1 (mb)          ║
║  S05  │ MB only        │ K=32, α=5M          │ col 1 (mb)          ║
║  S06  │ RAW only       │ K=10, α=5M          │ col 2 (raw)         ║
║  S07  │ RAW only       │ K=8, α=3M           │ col 2 (raw)         ║
║  S08  │ Koopman only   │ (base config)       │ col 3 (koop)        ║
║  S09  │ V70 baseline   │ (verification)      │ original blend_t    ║
║  S10  │ NB only        │ K=10, α=10          │ col 0 (nb)          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  S09 MUST return MSR ≈ 40,301. If not, K is wrong.                 ║
║  bei_d1 = 5 × MSR - 144,773                                        ║
║                                                                      ║
║  SUBMIT ORDER: S09 first (verify K), then S01-S08, S10             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
