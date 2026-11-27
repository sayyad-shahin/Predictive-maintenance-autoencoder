"""From reconstruction error to maintenance decisions.

  error   : how badly the autoencoder rebuilt this cycle's readings (average over sensors of the squared error,
            each sensor measured in multiples of its normal error, so 1.0 = typical healthy behaviour)
  health  : the error smoothed over time (rolling median, then EWMA). One noisy cycle should not raise an alarm.
  alarm   : raised when health stays above the threshold for `patience` cycles in a row, and stays on after that
  status  : normal -> watch (close to the threshold) -> warning (alarm) -> critical (far above the threshold)
  RUL     : a rough estimate of remaining useful life, found by extending the recent trend of the health index
"""
import numpy as np

from .data import make_windows

STATUS_ORDER = ["normal", "watch", "warning", "critical"]
ADVICE = {
    "normal": "Operating normally. No action needed.",
    "watch": "Readings are drifting toward the alarm level. Keep an eye on this machine.",
    "warning": "Abnormal behaviour detected. Schedule an inspection soon.",
    "critical": "Severe abnormal behaviour. Plan maintenance immediately.",
}


def ewma(x, alpha):
    x = np.asarray(x, float)
    out = np.empty(len(x))
    acc = 0.0
    for i, v in enumerate(x):
        acc = v if i == 0 else alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def rolling_median(x, k):
    """Causal rolling median over the last k values: ignores isolated sensor glitches but follows real trends."""
    x = np.asarray(x, float)
    if k <= 1:
        return x
    return np.array([np.median(x[max(0, i - k + 1):i + 1]) for i in range(len(x))])


def first_alarm(health, threshold, patience):
    """Index of the cycle where `health` has been above `threshold` for `patience` cycles in a row, or None."""
    run = 0
    for i, v in enumerate(health):
        run = run + 1 if v > threshold else 0
        if run >= patience:
            return i
    return None


def status_series(health, threshold, patience, critical_mult, watch_ratio=0.75):
    out, run, latched = [], 0, False
    for v in health:
        run = run + 1 if v > threshold else 0
        latched = latched or run >= patience
        ratio = v / threshold
        if latched:
            out.append("critical" if ratio >= critical_mult else "warning")
        else:
            out.append("watch" if ratio >= watch_ratio else "normal")
    return out


def estimate_rul(cycles, health, alarm_index, failure_level, span=25, max_rul=400):
    """For every cycle after the alarm, extend the recent trend of sqrt(health) until it reaches the failure level.
    Returns a list with None where no estimate is possible. A rough guide, not a guarantee."""
    out = [None] * len(health)
    if alarm_index is None or failure_level is None or failure_level <= 0:
        return out
    root = np.sqrt(np.asarray(health, float))
    target = np.sqrt(failure_level)
    for i in range(alarm_index, len(health)):
        lo = max(0, i - span + 1)
        if i - lo < 4:
            continue
        slope, intercept = np.polyfit(np.asarray(cycles[lo:i + 1], float), root[lo:i + 1], 1)
        if slope <= 1e-9:
            continue
        now = slope * cycles[i] + intercept
        out[i] = float(np.clip((target - now) / slope, 0, max_rul))
    return out


class Detector:
    """Turns raw sensor rows into health, status and alarms. `reconstruct` maps standardised windows to their rebuild."""

    def __init__(self, reconstruct, scaler, cfg, model=None):
        self.reconstruct, self.scaler, self.cfg, self.model = reconstruct, scaler, dict(cfg), model

    @property
    def features(self):
        return self.cfg["features"]

    def raw(self, frame):
        """Per-cycle error (n,) and per-sensor squared error (n, n_sensors) for one machine's rows.

        Warm-up correction: every machine is a little different, so the average rebuild error over its first `warmup`
        cycles (assumed healthy) is treated as that machine's own 'zero point' and subtracted."""
        c = self.cfg
        w, d = c["window"], len(self.features)
        X = make_windows(self.scaler.transform(frame[self.features].to_numpy(float)), w)
        resid = (X - self.reconstruct(X)).reshape(len(X), w, d)               # signed rebuild error per sensor
        k = int(c.get("warmup", 0))
        if k > 0:
            resid = resid - np.median(resid[:k].mean(axis=1), axis=0)
        per_sensor = (resid ** 2).mean(axis=1)
        base = c.get("sensor_baseline")
        if base is None:                                                       # before calibration: plain mean squared error
            return per_sensor.mean(axis=1), per_sensor
        return (per_sensor / np.maximum(np.asarray(base), 1e-9)).mean(axis=1), per_sensor   # every sensor counts equally

    def health(self, err):
        """Health index: glitch-resistant (rolling median) and smoothed (EWMA) version of the error."""
        return ewma(rolling_median(err, int(self.cfg.get("median_window", 1))), self.cfg["alpha"])

    def analyze(self, cycles, err, per_sensor):
        c = self.cfg
        health = self.health(err)
        thr = c["threshold"]
        idx = first_alarm(health, thr, c["patience"])
        status = status_series(health, thr, c["patience"], c["critical_mult"])
        base = np.asarray(c.get("sensor_baseline") or np.ones(per_sensor.shape[1]))
        rel = per_sensor / np.maximum(base, 1e-9)                      # error per sensor, in multiples of normal
        smooth_rel = np.vstack([ewma(rel[:, j], 0.3) for j in range(rel.shape[1])]).T
        return {"cycles": [int(x) for x in cycles], "error": err, "health": health, "ratio": health / thr, "status": status,
                "sensor_ratio": smooth_rel, "alarm_index": idx,
                "alarm_cycle": int(cycles[idx]) if idx is not None else None,
                "rul": estimate_rul(cycles, health, idx, c.get("failure_level"))}

    def score_unit(self, frame, cycle_col="cycle"):
        frame = frame.sort_values(cycle_col)
        err, per_sensor = self.raw(frame)
        return self.analyze(frame[cycle_col].to_numpy(), err, per_sensor)

    def score_frame(self, df, unit_col="unit", cycle_col="cycle"):
        return {u: self.score_unit(g, cycle_col) for u, g in df.groupby(unit_col)}

    def top_sensors(self, result, index, k=3):
        """The sensors that look most abnormal at a given cycle, as (name, times-normal) pairs."""
        row = result["sensor_ratio"][index]
        order = np.argsort(-row)[:k]
        return [(self.features[j], float(row[j])) for j in order]
