"""Show the saved model's results on the machines it never saw.   python evaluate.py [--model models] [--data data/sensor_data.csv]"""
import argparse
import os
import sys

import pandas as pd

from pdm import load_detector
from pdm.data import load_csv
from pdm.evaluate import evaluate_detector

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models")
ap.add_argument("--data", default="data/sensor_data.csv")
ap.add_argument("--truth", default=None)
a = ap.parse_args()
try:
    det, saved = load_detector(a.model)
    c = det.cfg
    df, _ = load_csv(a.data, c["unit_col"], c["cycle_col"], c["features"])
except (OSError, ValueError) as e:
    sys.exit(f"Error: {e}")
test_units = saved.get("test_units") or sorted(df[c["unit_col"]].unique())
tpath = a.truth or os.path.join(os.path.dirname(a.data), "ground_truth.csv")
truth = pd.read_csv(tpath) if os.path.exists(tpath) else None
metrics, rows = evaluate_detector(det, df, test_units, truth, c["unit_col"], c["cycle_col"])
print(f"Machines tested: {metrics['machines_tested']}\n")
for k, v in metrics.items():
    print(f"  {k:32s}{v}")
print(f"\n{'machine':>8}{'life':>6}{'wear began':>12}{'alarm at':>10}{'warning time':>14}")
for r in rows:
    print(f"{r['unit']:>8}{r['life']:>6}{r['degrade_start']:>12}{str(r['alarm_cycle'] or '-'):>10}{(str(r['lead_time']) + ' cycles') if r['lead_time'] is not None else ('FALSE ALARM' if r['false_alarm'] else 'missed'):>14}")
