"""Training pipeline: split machines, train on healthy data, calibrate the alarm threshold, save and load."""
import json
import os

import numpy as np
import torch
from sklearn.decomposition import PCA

from .data import Scaler, healthy_part, make_windows, split_units
from .detect import Detector
from .model import Autoencoder, reconstruct_fn, train_autoencoder

DEFAULTS = dict(window=3, hidden=[32, 16], latent=3, epochs=200, lr=2e-3, noise=0.05, healthy_fraction=0.4,
                alpha=0.15, median_window=5, warmup=20, quantile=0.995, margin=1.0, patience=3, critical_mult=3.0, seed=0)


def fit_pipeline(df, features, unit_col="unit", cycle_col="cycle", method="autoencoder", run_to_failure=True,
                 verbose=True, **overrides):
    """Train a detector on the healthy early life of the training machines.

    method: 'autoencoder' (the real model) or 'pca' (a linear baseline for comparison).
    Returns (detector, info) where info holds the split of machine ids and the training history.
    """
    cfg = {**DEFAULTS, **overrides, "features": list(features), "unit_col": unit_col, "cycle_col": cycle_col, "method": method}
    units = sorted(df[unit_col].unique())
    train_u, val_u, test_u = split_units(units, cfg["seed"])
    groups = {u: g.sort_values(cycle_col) for u, g in df.groupby(unit_col)}
    frac, w = cfg["healthy_fraction"], cfg["window"]

    healthy_rows = np.vstack([g[features].to_numpy(float)[healthy_part(len(g), frac)] for u, g in groups.items() if u in train_u])
    scaler = Scaler.fit(healthy_rows)

    def healthy_windows(unit_ids):
        parts = []
        for u in unit_ids:
            g = groups[u]
            X = make_windows(scaler.transform(g[features].to_numpy(float)), w)
            parts.append(X[healthy_part(len(g), frac)])
        return np.vstack(parts)

    X_train, X_val = healthy_windows(train_u), healthy_windows(val_u)
    if verbose:
        print(f"Machines: {len(train_u)} train / {len(val_u)} validation / {len(test_u)} test. "
              f"Healthy training rows: {len(X_train):,}, input size: {X_train.shape[1]}")

    model, history = None, {}
    if method == "autoencoder":
        model, history = train_autoencoder(X_train, X_val, tuple(cfg["hidden"]), cfg["latent"], cfg["epochs"], lr=cfg["lr"],
                                           noise=cfg["noise"], seed=cfg["seed"], verbose=verbose)
        recon = reconstruct_fn(model)
    elif method == "pca":
        pca = PCA(n_components=cfg["latent"]).fit(X_train)
        recon = lambda X: pca.inverse_transform(pca.transform(X))
    else:
        raise ValueError("method must be 'autoencoder' or 'pca'")

    det = Detector(recon, scaler, {**cfg, "threshold": 1.0, "sensor_baseline": None, "failure_level": None}, model)

    # calibrate on healthy validation data: alarm threshold and normal error per sensor
    health_vals, sensor_err = [], []
    for u in val_u:
        g = groups[u]
        err, per_sensor = det.raw(g)
        mask = healthy_part(len(g), frac)
        health_vals.append(det.health(err)[mask])
        sensor_err.append(per_sensor[mask])
    det.cfg["sensor_baseline"] = np.vstack(sensor_err).mean(axis=0).tolist()
    health_vals = []                                            # recompute with every sensor weighted by its normal error
    for u in val_u:
        g = groups[u]
        health_vals.append(det.health(det.raw(g)[0])[healthy_part(len(g), frac)])
    det.cfg["threshold"] = float(np.quantile(np.concatenate(health_vals), cfg["quantile"]) * cfg["margin"])

    # level of the health index at the moment of failure, from the training machines (for the RUL estimate)
    if run_to_failure:
        finals = [det.health(det.raw(groups[u])[0])[-1] for u in train_u]
        det.cfg["failure_level"] = float(np.median(finals))
    info = {"train_units": train_u, "val_units": val_u, "test_units": test_u, "history": history}
    return det, info


def save_detector(det, directory, metrics=None):
    """Write autoencoder.pt (weights) and config.json (settings, scaler, threshold, metrics)."""
    os.makedirs(directory, exist_ok=True)
    if det.model is None:
        raise ValueError("Only autoencoder detectors can be saved.")
    m = det.model
    torch.save({"state": m.state_dict(), "n_in": m.n_in, "hidden": list(m.hidden), "latent": m.latent}, os.path.join(directory, "autoencoder.pt"))
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump({"config": det.cfg, "scaler": det.scaler.to_dict(), "metrics": metrics or {}}, f, indent=2)


def load_detector(directory):
    """Load a detector saved by save_detector. Returns (detector, metrics)."""
    with open(os.path.join(directory, "config.json")) as f:
        saved = json.load(f)
    ckpt = torch.load(os.path.join(directory, "autoencoder.pt"), map_location="cpu", weights_only=True)
    model = Autoencoder(ckpt["n_in"], tuple(ckpt["hidden"]), ckpt["latent"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return Detector(reconstruct_fn(model), Scaler.from_dict(saved["scaler"]), saved["config"], model), saved.get("metrics", {})
