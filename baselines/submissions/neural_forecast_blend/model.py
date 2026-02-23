import os
import numpy as np


# ---- config ----
VAR_THRESHOLD = 300.0


def _stats_basic(x10):
    # x10: (N, 10) -> (N, 6) summary features per channel per band
    mu = x10.mean(axis=1, keepdims=True)
    sd = x10.std(axis=1, keepdims=True) + 1e-6
    slope = (x10[:, -1:] - x10[:, :1]) / 9.0
    last = x10[:, -1:]
    rng = x10.max(axis=1, keepdims=True) - x10.min(axis=1, keepdims=True)
    mid = x10.shape[1] // 2
    curv = x10[:, mid+1:mid+2] - 2.0 * x10[:, mid:mid+1] + x10[:, mid-1:mid]
    return np.concatenate([mu, sd, slope, last, rng, curv], axis=1).astype(np.float32)


class Model:
    def __init__(self, monkey_name=""):
        self.monkey_name = monkey_name
        self.input_size = 239 if monkey_name == "affi" else (89 if monkey_name == "beignet" else None)
        self.weights = None
        self.mlp_weights = []
        self.bei_mlp_weights = []

    def load(self):
        if self.monkey_name not in ("affi", "beignet"):
            return
        self.weights = self._load_npz(self.monkey_name)
        here = os.path.dirname(__file__)
        if self.monkey_name == "affi":
            for fn in sorted(os.listdir(here)):
                if fn.startswith("mlp_affi_d1") and fn.endswith(".npz"):
                    self.mlp_weights.append(dict(np.load(os.path.join(here, fn), allow_pickle=False)))
        elif self.monkey_name == "beignet":
            for fn in sorted(os.listdir(here)):
                if fn.startswith("mlp_bei_d1") and fn.endswith(".npz"):
                    self.bei_mlp_weights.append(dict(np.load(os.path.join(here, fn), allow_pickle=False)))

    def _load_npz(self, name):
        here = os.path.dirname(__file__)
        fp = os.path.join(here, f"weights_{name}.npz")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing weights file: {fp}")
        return dict(np.load(fp, allow_pickle=False))

    def _ensure_loaded(self, C):
        if self.weights is not None:
            return
        name = "affi" if C == 239 else "beignet"
        self.monkey_name = name
        self.input_size = C
        self.weights = self._load_npz(name)
        here = os.path.dirname(__file__)
        if name == "affi":
            for fn in sorted(os.listdir(here)):
                if fn.startswith("mlp_affi_d1") and fn.endswith(".npz"):
                    self.mlp_weights.append(dict(np.load(os.path.join(here, fn), allow_pickle=False)))
        elif name == "beignet":
            for fn in sorted(os.listdir(here)):
                if fn.startswith("mlp_bei_d1") and fn.endswith(".npz"):
                    self.bei_mlp_weights.append(dict(np.load(os.path.join(here, fn), allow_pickle=False)))

    def _wt(self, w, pfx, name):
        # weight lookup: tries prefixed key first, falls back to global
        route = pfx.rstrip("_") if pfx else ""
        if route:
            k = f"{route}_w_{name}"
            if k in w:
                return float(w[k][0])
        return float(w.get(f"w_{name}", np.array([0.0]))[0])

    # ---- component runners ----

    def _run_ridge(self, w, pfx, x0, N, C):
        # per-channel ridge on raw band0 timesteps
        key = pfx + "ridge_coefs"
        if key not in w:
            return None, 0.0
        coefs = w[key].astype(np.float32)
        bias = w[pfx + "ridge_intercepts"].astype(np.float32)
        p = np.empty((N, 10, C), np.float32)
        for c in range(C):
            p[:, :, c] = x0[:, :, c] @ coefs[c].T + bias[c]
        return p, self._wt(w, pfx, "ridge")

    def _run_nb(self, w, pfx, x0, N, C):
        # neighbor-ridge: predict each channel using K most-correlated neighbors
        for tag in ["nb3", "nb5", "nb10"]:
            key = pfx + tag + "_coefs"
            if key not in w:
                continue
            idx = w[pfx + tag + "_idx"].astype(np.int32)
            coefs = w[key].astype(np.float32)
            bias = w[pfx + tag + "_intercepts"].astype(np.float32)
            K = idx.shape[1]
            p = np.empty((N, 10, C), np.float32)
            for c in range(C):
                blocks = [x0[:, :, c]]
                for j in range(K):
                    blocks.append(x0[:, :, int(idx[c, j])])
                Xc = np.concatenate(blocks, axis=1)
                p[:, :, c] = Xc @ coefs[c].T + bias[c]
            return p, self._wt(w, pfx, "nb")
        return None, 0.0

    def _run_mb(self, w, pfx, X, N, C, F):
        # moment-based: 6 summary stats per channel per band as features
        key = pfx + "mb_coefs"
        if key not in w:
            return None, 0.0
        idx = w[pfx + "mb_idx"].astype(np.int32)
        coefs = w[key].astype(np.float32)
        bias = w[pfx + "mb_intercepts"].astype(np.float32)
        p = np.empty((N, 10, C), np.float32)
        for c in range(C):
            chans = [c] + [int(j) for j in idx[c]]
            feats = []
            for ch in chans:
                for b in range(F):
                    feats.append(_stats_basic(X[:, :10, ch, b]))
            Xc = np.concatenate(feats, axis=1)
            p[:, :, c] = Xc @ coefs[c].T + bias[c]
        return p, self._wt(w, pfx, "mb")

    def _run_raw(self, w, pfx, X, N, C, F):
        # raw temporal values across all bands as features
        key = pfx + "raw_coefs"
        if key not in w:
            return None, 0.0
        idx = w[pfx + "raw_idx"].astype(np.int32)
        coefs = w[key].astype(np.float32)
        bias = w[pfx + "raw_intercepts"].astype(np.float32)
        p = np.empty((N, 10, C), np.float32)
        for c in range(C):
            chans = [c] + [int(j) for j in idx[c]]
            feats = []
            for ch in chans:
                for b in range(F):
                    feats.append(X[:, :10, ch, b])
            Xc = np.concatenate(feats, axis=1)
            p[:, :, c] = Xc @ coefs[c].T + bias[c]
        if pfx + "raw_residual" in w:
            naive = X[:, 9:10, :, 0].repeat(10, axis=1)
            p = p + naive
        return p, self._wt(w, pfx, "raw")

    def _run_koopman(self, w, pfx, x0, naive, N, C):
        # koopman correction in channel-PCA latent space
        key = pfx + "koopman_W"
        if key not in w:
            return None, 0.0
        Wk = w[key].astype(np.float32)
        Bk = w[pfx + "koopman_B"].astype(np.float32)
        Kk = int(w[pfx + "K"][0]) if (pfx + "K") in w else int(w["K"][0])
        mu = x0.mean(axis=1, keepdims=True)
        sig = x0.std(axis=1, keepdims=True) + 1e-6
        xn = (x0 - mu) / sig
        z = (xn.reshape(-1, C) @ Wk).reshape(N, 10, Kk)
        p = naive + (z.reshape(N, -1) @ Bk).reshape(N, 10, C) * sig
        return p, self._wt(w, pfx, "koop")

    def _run_mlp(self, w, pfx, X, N, C, F):
        # MLP ensemble: average predictions across seeds
        mw_list = self.mlp_weights if C == 239 else self.bei_mlp_weights
        if not mw_list or pfx != "d1_":
            return None, 0.0
        accum = np.zeros((N, 10, C), np.float32)
        for mw in mw_list:
            n_layers = int(mw["mlp_n_layers"][0]) + 1
            idx = mw["mlp_mb_idx"].astype(np.int32)
            for c in range(C):
                chans = [c] + [int(j) for j in idx[c]]
                feats = []
                for ch in chans:
                    for b in range(F):
                        feats.append(_stats_basic(X[:, :10, ch, b]))
                x = np.concatenate(feats, axis=1)
                for li in range(n_layers):
                    x = x @ mw[f"w{li}_c{c:03d}"].astype(np.float32) + mw[f"b{li}_c{c:03d}"].astype(np.float32)
                    if li < n_layers - 1:
                        x = np.maximum(x, 0)
                accum[:, :, c] += x
        p = accum / len(mw_list)
        return p, self._wt(w, pfx, "mlp")

    # ---- predict ----

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        N, T, C, F = X.shape
        self._ensure_loaded(C)
        w = self.weights

        x0 = X[:, :10, :, 0]
        naive = x0[:, 9:10, :].repeat(10, axis=1)

        # route to day-specific weights based on input variance
        pfx = ""
        if any(k.startswith("d1_") for k in w):
            thr = float(w.get("var_threshold", np.array([VAR_THRESHOLD]))[0])
            batch_std = float(x0.std(axis=(1, 2)).mean())
            pfx = "d1_" if batch_std > thr else "priv_"

        # ---- run components ----
        preds, wts = [], []
        p, wt = self._run_ridge(w, pfx, x0, N, C)
        if p is not None:
            preds.append(p); wts.append(wt)
        p, wt = self._run_nb(w, pfx, x0, N, C)
        if p is not None:
            preds.append(p); wts.append(wt)
        p, wt = self._run_mb(w, pfx, X, N, C, F)
        if p is not None:
            preds.append(p); wts.append(wt)
        p, wt = self._run_raw(w, pfx, X, N, C, F)
        if p is not None:
            preds.append(p); wts.append(wt)
        p, wt = self._run_mlp(w, pfx, X, N, C, F)
        if p is not None:
            preds.append(p); wts.append(wt)
        p, wt = self._run_koopman(w, pfx, x0, naive, N, C)
        if p is not None:
            preds.append(p); wts.append(wt)

        if not preds:
            raise RuntimeError("No recognized model components found in weights_*.npz")

        # ---- blend: per-timestep if available, else scalar weights ----
        tkey = pfx + "blend_t"
        if tkey in w and w[tkey].shape[1] == len(preds):
            blend_t = w[tkey].astype(np.float32)
            forecast = np.zeros((N, 10, C), np.float32)
            for t in range(10):
                tw = blend_t[t]; tw = tw / tw.sum()
                for i in range(len(preds)):
                    forecast[:, t, :] += tw[i] * preds[i][:, t, :]
        else:
            wts = np.array(wts, dtype=np.float32)
            wts /= wts.sum()
            forecast = sum(wt * p for wt, p in zip(wts, preds))

        out = np.zeros((N, 20, C), np.float32)
        out[:, :10, :] = x0
        out[:, 10:, :] = forecast
        return out
