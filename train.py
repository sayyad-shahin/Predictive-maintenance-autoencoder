"""Train the autoencoder, test it on machines it has never seen, and save it.

   python train.py                                   # uses data/sensor_data.csv (created if missing)
   python train.py --data my_machines.csv --unit-col machine --cycle-col hour --features temp vibration current
   python train.py --window 1 --latent 2 --epochs 100

Your CSV needs one row per machine and time step, a column that identifies the machine, a column with the time/cycle
number, and numeric sensor columns. Machines should be run-to-failure (they end when the machine failed or was
replaced). The first part of each machine's life (--healthy-fraction) is assumed to be healthy.
"""
import argparse
import json
import os
import sys

import pandas as pd

from pdm import fit_pipeline, save_detector
from pdm.data import load_csv
from pdm.evaluate import evaluate_detector
from pdm.pipeline import DEFAULTS
from pdm.simulate import simulate


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predictive maintenance with an autoencoder", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sensor_data.csv")
    ap.add_argument("--truth", default=None, help="optional CSV with columns unit, degrade_start (used only for evaluation)")
    ap.add_argument("--unit-col", default="unit")
    ap.add_argument("--cycle-col", default="cycle")
    ap.add_argument("--features", nargs="*", default=None, help="sensor columns (default: all numeric columns)")
    ap.add_argument("--window", type=int, default=DEFAULTS["window"], help="how many recent cycles the model sees at once")
    ap.add_argument("--latent", type=int, default=DEFAULTS["latent"], help="size of the autoencoder bottleneck")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--healthy-fraction", type=float, default=DEFAULTS["healthy_fraction"], help="first part of each machine's life assumed healthy")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--out", default="models", help="folder for the saved model")
    ap.add_argument("--plots", default="assets", help="folder for figures ('' to skip)")
    a = ap.parse_args(argv)

    if not os.path.exists(a.data):
        if a.data != "data/sensor_data.csv":
            print(f"Error: file not found: {a.data}")
            return 2
        print("No data found, creating the sample dataset...")
        d, t = simulate()
        os.makedirs("data", exist_ok=True)
        d.to_csv(a.data, index=False); t.to_csv("data/ground_truth.csv", index=False)
    try:
        df, features = load_csv(a.data, a.unit_col, a.cycle_col, a.features)
    except ValueError as e:
        print(f"Error: {e}")
        return 2
    truth_path = a.truth or os.path.join(os.path.dirname(a.data), "ground_truth.csv")
    truth = pd.read_csv(truth_path) if os.path.exists(truth_path) and (a.truth or a.data == "data/sensor_data.csv") else None
    print(f"Loaded {len(df):,} rows, {df[a.unit_col].nunique()} machines. Sensors used: {', '.join(features)}")

    common = dict(window=a.window, latent=a.latent, epochs=a.epochs, healthy_fraction=a.healthy_fraction, seed=a.seed)
    print("\nTraining the autoencoder on healthy data only...")
    det, info = fit_pipeline(df, features, a.unit_col, a.cycle_col, "autoencoder", **common)
    print(f"Stopped after {info['history']['epochs_run']} epochs (best validation loss {info['history']['best_val']:.4f}).")
    metrics, rows = evaluate_detector(det, df, info["test_units"], truth, a.unit_col, a.cycle_col)
    pca, pinfo = fit_pipeline(df, features, a.unit_col, a.cycle_col, "pca", verbose=False, **common)
    pmetrics, _ = evaluate_detector(pca, df, pinfo["test_units"], truth, a.unit_col, a.cycle_col)

    print(f"\nResults on {metrics['machines_tested']} machines the model never saw during training:")
    print(f"{'':34s}{'Autoencoder':>14s}{'PCA baseline':>14s}")
    for k, v in metrics.items():
        if k != "machines_tested":
            print(f"  {k:32s}{str(v):>14s}{str(pmetrics[k]):>14s}")
    print(f"\nAlarm threshold: {det.cfg['threshold']:.3f}. An alarm needs the health index above it for {det.cfg['patience']} cycles in a row.")

    saved = {"autoencoder": metrics, "pca_baseline": pmetrics, "test_units": [int(u) for u in info["test_units"]],
             "train_units": [int(u) for u in info["train_units"]], "epochs_run": info["history"]["epochs_run"],
             "per_machine": rows, "evaluated_with_ground_truth": truth is not None}
    save_detector(det, a.out, saved)
    print(f"Model saved to {a.out}/")
    if a.plots:
        from pdm import plots
        plots.training_curve(info["history"], os.path.join(a.plots, "training_curve.png"))
        plots.health_examples(det, df, rows, truth, os.path.join(a.plots, "health_index_examples.png"), a.unit_col, a.cycle_col)
        plots.comparison({"autoencoder": metrics, "pca": pmetrics}, os.path.join(a.plots, "autoencoder_vs_pca.png"))
        print(f"Figures saved to {a.plots}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
