"""Data handling: scaling, healthy-period selection, time windows, and splitting machines into train/val/test."""
import numpy as np
import pandas as pd


class Scaler:
    """Standardise each sensor with the mean and spread of healthy data."""

    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)

    @classmethod
    def fit(cls, X):
        std = X.std(axis=0)
        return cls(X.mean(axis=0), np.where(std < 1e-9, 1.0, std))

    def transform(self, X):
        return (np.asarray(X, float) - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


def load_csv(path, unit_col="unit", cycle_col="cycle", features=None):
    """Read a sensor CSV. Returns the DataFrame sorted by unit and cycle, and the list of feature columns."""
    df = pd.read_csv(path)
    for col in (unit_col, cycle_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. Available columns: {list(df.columns)}")
    if features is None:
        features = [c for c in df.columns if c not in (unit_col, cycle_col) and pd.api.types.is_numeric_dtype(df[c])]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")
    if not features:
        raise ValueError("No numeric feature columns found.")
    df = df.sort_values([unit_col, cycle_col]).reset_index(drop=True)
    if df[features].isna().any().any():
        df[features] = df.groupby(unit_col)[features].transform(lambda s: s.ffill().bfill())
        if df[features].isna().any().any():
            raise ValueError("Some feature columns are entirely empty.")
    return df, list(features)


def make_windows(X, window):
    """Stack the last `window` cycles of each cycle into one row. The start of the series is padded with its first row."""
    if window <= 1:
        return np.asarray(X, float)
    pad = np.vstack([np.repeat(X[:1], window - 1, axis=0), X])
    return np.hstack([pad[k:k + len(X)] for k in range(window)])


def split_units(units, seed=0, val=0.15, test=0.25):
    """Split machine ids into train / validation / test, so the model is tested on machines it has never seen."""
    units = list(units)
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_test, n_val = max(1, round(len(units) * test)), max(1, round(len(units) * val))
    if len(units) < 3:
        raise ValueError("Need at least 3 machines to split into train, validation and test sets.")
    return sorted(units[n_test + n_val:]), sorted(units[n_test:n_test + n_val]), sorted(units[:n_test])


def healthy_part(n_cycles, fraction):
    """Boolean mask of the first `fraction` of a machine's life: the part assumed to be healthy."""
    return np.arange(1, n_cycles + 1) <= fraction * n_cycles
