"""Check sensor data with the trained model.
   python predict.py data/sensor_data.csv --out predictions.csv
   python predict.py newdata.csv --unit-col machine

The file needs the same sensor columns the model was trained on. If it has no machine column, all rows are treated as
one machine. If it has no cycle column, rows are numbered in order.
"""
import argparse
import sys

import pandas as pd

from pdm import load_detector
from pdm.detect import ADVICE

ap = argparse.ArgumentParser(description="Score sensor data for signs of wear", epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("csv")
ap.add_argument("--model", default="models")
ap.add_argument("--out", default="predictions.csv")
a = ap.parse_args()
try:
    det, _ = load_detector(a.model)
    u, c = det.cfg["unit_col"], det.cfg["cycle_col"]
    df = pd.read_csv(a.csv)
    missing = [f for f in det.features if f not in df.columns]
    if missing:
        sys.exit(f"Error: the file is missing sensor columns: {missing}")
    if u not in df.columns:
        df[u] = 1
    if c not in df.columns:
        df[c] = df.groupby(u).cumcount() + 1
    df = df.sort_values([u, c])
    results = det.score_frame(df, u, c)
except (OSError, ValueError) as e:
    sys.exit(f"Error: {e}")

rows = []
for unit, res in results.items():
    for i, cyc in enumerate(res["cycles"]):
        rows.append({u: unit, c: cyc, "health_index": round(float(res["ratio"][i]), 3), "status": res["status"][i],
                     "estimated_rul": None if res["rul"][i] is None else round(res["rul"][i]),
                     "most_abnormal_sensor": det.top_sensors(res, i, 1)[0][0]})
out = pd.DataFrame(rows)
out.to_csv(a.out, index=False)
print(f"Scored {len(out):,} rows from {len(results)} machine(s). Saved to {a.out}\n")
print("Health index: 1.0 = alarm level.")
for unit, res in results.items():
    last = len(res["cycles"]) - 1
    top = ", ".join(f"{n} ({r:.1f}x normal)" for n, r in det.top_sensors(res, last))
    print(f"\nMachine {unit}: status {res['status'][last].upper()} at cycle {res['cycles'][last]}"
          + (f", alarm first raised at cycle {res['alarm_cycle']}" if res["alarm_cycle"] else ", no alarm raised"))
    print(f"  {ADVICE[res['status'][last]]}")
    if res["status"][last] != "normal":
        print(f"  Most abnormal sensors: {top}")
    if res["rul"][last] is not None:
        print(f"  Rough estimate of remaining life: about {res['rul'][last]:.0f} cycles")
