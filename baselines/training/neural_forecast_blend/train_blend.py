import numpy as np, os
from sklearn.linear_model import Ridge

DATA = os.path.expanduser("~/hackathon_2026/data")

def load(name):
    return np.load(os.path.join(DATA, name))["arr_0"].astype(np.float32)

def train_blend(monkey, files, C, K, lamB, w_ridge, ridge_alpha=30.0):
    all_data = np.concatenate([load(f) for f in files], axis=0)
    raw = all_data[:, :, :, 0].astype(np.float32)
    x0 = raw[:, :10, :]
    y  = raw[:, 10:, :]

    # -----------------
    # Ridge per channel
    # -----------------
    ridge_coefs = np.empty((C, 10, 10), dtype=np.float32)
    ridge_intercepts = np.empty((C, 10), dtype=np.float32)
    for c in range(C):
        reg = Ridge(alpha=ridge_alpha)
        reg.fit(x0[:, :, c], y[:, :, c])
        ridge_coefs[c] = reg.coef_.astype(np.float32)
        ridge_intercepts[c] = reg.intercept_.astype(np.float32)

    # -----------------
    # Koopman residual model
    # -----------------
    naive = x0[:, 9:10, :].repeat(10, axis=1)
    r = y - naive

    mu  = x0.mean(axis=1, keepdims=True)
    sig = x0.std(axis=1, keepdims=True) + 1e-6
    x0n = (x0 - mu) / sig
    rn  = r / sig

    Xmat = x0n.reshape(-1, C)
    cov = (Xmat.T @ Xmat) / Xmat.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1][:K]
    W = evecs[:, idx].astype(np.float32)   # (C,K)

    Z = (Xmat @ W).reshape(-1, 10, K)
    F = Z.reshape(Z.shape[0], -1)          # (N, 10K)
    Y = rn.reshape(rn.shape[0], -1)        # (N, 10C)

    FtF = F.T @ F + lamB * np.eye(F.shape[1], dtype=np.float32)
    B = np.linalg.solve(FtF, F.T @ Y).astype(np.float32)  # (10K, 10C)

    np.savez_compressed(
        f"weights_{monkey}.npz",
        ridge_alpha=np.array([ridge_alpha], np.float32),
        ridge_coefs=ridge_coefs,
        ridge_intercepts=ridge_intercepts,
        koopman_W=W,
        koopman_B=B,
        K=np.array([K], np.int32),
        lamB=np.array([lamB], np.float32),
        w_ridge=np.array([w_ridge], np.float32),
        C=np.array([C], np.int32),
    )
    print(f"[OK] {monkey}: N={len(all_data)} C={C} ridge_alpha={ridge_alpha} K={K} lamB={lamB} w_ridge={w_ridge}")

def main():
    # From your sweep summary (best blend weights)
    train_blend(
        "affi",
        ["train_data_affi.npz", "train_data_affi_2024-03-20_private.npz"],
        C=239, K=10, lamB=100.0, w_ridge=0.90, ridge_alpha=30.0,
    )
    train_blend(
        "beignet",
        ["train_data_beignet.npz",
         "train_data_beignet_2022-06-01_private.npz",
         "train_data_beignet_2022-06-02_private.npz"],
        C=89, K=4, lamB=100.0, w_ridge=0.60, ridge_alpha=30.0,
    )

if __name__ == "__main__":
    main()
