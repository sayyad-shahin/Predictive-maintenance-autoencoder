"""Figures for the README (matplotlib)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"autoencoder": "#4f46e5", "pca": "#9aa3b2"}


def training_curve(history, path):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(history["train"], label="training loss (noisy input)", color="#4f46e5")
    ax.plot(history["val"], label="validation loss", color="#e8590c")
    ax.set(xlabel="Epoch", ylabel="Mean squared error", title="Autoencoder training on healthy data", yscale="log")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, path)


def health_examples(det, df, rows, truth, path, unit_col="unit", cycle_col="cycle", n=3):
    """Health index of a few test machines, with the alarm, start of wear and failure marked."""
    modes = dict(zip(truth["unit"], truth["failure_mode"])) if truth is not None else {}
    chosen, seen = [], set()
    for r in rows:                                            # try to show one machine per failure mode
        if r["detected"] and modes.get(r["unit"], r["unit"]) not in seen:
            chosen.append(r); seen.add(modes.get(r["unit"], r["unit"]))
        if len(chosen) == n:
            break
    chosen = (chosen + [r for r in rows if r not in chosen])[:n]
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.6 * len(chosen), 3.6), squeeze=False)
    for ax, r in zip(axes[0], chosen):
        res = det.score_unit(df[df[unit_col] == r["unit"]], cycle_col)
        ax.plot(res["cycles"], res["ratio"], color="#4f46e5", lw=1.8, label="health index")
        ax.axhline(1.0, color="#d93a3a", ls="--", lw=1.2, label="alarm threshold")
        ax.axvline(r["degrade_start"], color="#999", ls=":", lw=1.2, label="wear begins (truth)")
        if r["alarm_cycle"]:
            ax.axvline(r["alarm_cycle"], color="#e8590c", lw=1.5, label="alarm raised")
        ax.axvline(r["life"], color="black", lw=1.5, label="failure")
        ax.set(xlabel="Cycle", yscale="log", title=f"Machine {r['unit']} ({modes.get(r['unit'], '')})")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Health index (1 = alarm level)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    _save(fig, path)


def comparison(results, path):
    """Bar chart of the autoencoder against the PCA baseline."""
    names = list(results)
    metrics = [("detection_rate", "Failures caught"), ("false_alarm_machines", "Machines with a false alarm"), ("mean_lead_time_cycles", "Mean warning time (cycles)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (key, title) in zip(axes, metrics):
        vals = [results[n][key] or 0 for n in names]
        ax.bar(names, vals, color=[COLORS.get(n, "#888") for n in names])
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
