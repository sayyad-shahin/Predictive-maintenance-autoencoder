"""Measure how well the detector predicts failures on machines it has never seen."""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_detector(det, df, units, truth=None, unit_col="unit", cycle_col="cycle", rul_window=50):
    """Metrics on the given machines. `truth` (optional DataFrame with unit, degrade_start) says when wear really began.
    Without it, the first `healthy_fraction` of life is assumed healthy."""
    frac = det.cfg["healthy_fraction"]
    starts = {} if truth is None else dict(zip(truth["unit"], truth["degrade_start"]))
    rows, labels, scores = [], [], []
    healthy_exceed, healthy_cycles = 0, 0
    rul_err_all, rul_err_late = [], []
    for u in units:
        g = df[df[unit_col] == u].sort_values(cycle_col)
        life = len(g)
        start = int(starts.get(u, frac * life))
        res = det.score_unit(g, cycle_col)
        cycles = np.array(res["cycles"])
        h = res["health"]
        alarm = res["alarm_cycle"]
        false_alarm = alarm is not None and alarm <= start
        healthy = cycles <= start
        degraded_late = (cycles > life - rul_window) & (cycles > start)
        healthy_exceed += int((h[healthy] > det.cfg["threshold"]).sum())
        healthy_cycles += int(healthy.sum())
        scores += list(h[healthy | degraded_late])
        labels += list(degraded_late[healthy | degraded_late].astype(int))
        for i, est in enumerate(res["rul"]):
            if est is not None and res["status"][i] in ("warning", "critical"):
                err = abs(est - (life - cycles[i]))
                rul_err_all.append(err)
                if degraded_late[i]:
                    rul_err_late.append(err)
        rows.append({"unit": int(u), "life": life, "degrade_start": start, "alarm_cycle": alarm,
                     "lead_time": (life - alarm) if alarm is not None and not false_alarm else None,
                     "false_alarm": bool(false_alarm), "detected": alarm is not None and not false_alarm})
    n = len(rows)
    leads = [r["lead_time"] for r in rows if r["lead_time"] is not None]
    metrics = {
        "machines_tested": n,
        "detection_rate": round(sum(r["detected"] for r in rows) / n, 3),
        "false_alarm_machines": round(sum(r["false_alarm"] for r in rows) / n, 3),
        "false_alarm_cycle_rate": round(healthy_exceed / max(healthy_cycles, 1), 4),
        "mean_lead_time_cycles": round(float(np.mean(leads)), 1) if leads else None,
        "median_lead_time_cycles": round(float(np.median(leads)), 1) if leads else None,
        "min_lead_time_cycles": int(min(leads)) if leads else None,
        "auroc_last_%d_cycles" % rul_window: round(float(roc_auc_score(labels, scores)), 4) if len(set(labels)) == 2 else None,
        "rul_mae_after_alarm": round(float(np.mean(rul_err_all)), 1) if rul_err_all else None,
        "rul_mae_last_%d_cycles" % rul_window: round(float(np.mean(rul_err_late)), 1) if rul_err_late else None,
    }
    return metrics, rows
