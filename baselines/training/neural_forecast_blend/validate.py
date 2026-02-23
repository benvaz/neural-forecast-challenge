import os
import numpy as np
from model import Model


# ---- config ----
DATA = os.path.expanduser("~/hackathon_2026/data")


# ---- validation ----

def _check(monkey, fname):
    X = np.load(os.path.join(DATA, fname))["arr_0"]
    N = X.shape[0]
    split = int(0.8 * N)
    test = X[split:]

    # mask future timesteps with persistence
    Xm = test.copy()
    Xm[:, 10:, :, :] = Xm[:, 9:10, :, :]

    m = Model(monkey)
    m.load()
    pred = m.predict(Xm)

    actual = test[:, 10:, :, 0]
    forecast = pred[:, 10:, :]
    mse = np.mean((forecast - actual) ** 2)
    naive_mse = np.mean((test[:, 9:10, :, 0].repeat(10, axis=1) - actual) ** 2)
    r2 = 1 - np.sum((actual - forecast) ** 2) / np.sum(actual ** 2)
    print(f"{monkey}: MSE={mse:.0f}  naive={naive_mse:.0f}  "
          f"ratio={naive_mse / mse:.3f}x  R2={r2:.4f}")


# ---- main ----

if __name__ == "__main__":
    _check("affi", "train_data_affi.npz")
    _check("beignet", "train_data_beignet.npz")
