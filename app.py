"""Web dashboard.   python app.py   then open http://127.0.0.1:5000

Replay any machine cycle by cycle and watch the health index, alarm and sensor clues develop. Upload your own CSV to score it.
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from pdm import fit_pipeline, load_detector, save_detector
from pdm.data import load_csv
from pdm.detect import ADVICE

MAX_ROWS = 200_000


def load_or_train(model_dir, data_path):
    """Load the saved model. If it cannot be loaded (for example after a PyTorch upgrade), retrain it from the data."""
    try:
        return load_detector(model_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[info] Could not load the saved model ({type(e).__name__}). Training a new one from {data_path} ...")
        df, features = load_csv(data_path)
        det, _ = fit_pipeline(df, features, verbose=False)
        save_detector(det, model_dir, {})
        return load_detector(model_dir)


def create_app(model_dir="models", data_path="data/sensor_data.csv", truth_path=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    det, saved = load_or_train(model_dir, data_path)
    cfg = det.cfg
    u_col, c_col = cfg["unit_col"], cfg["cycle_col"]
    df, _ = load_csv(data_path, u_col, c_col, cfg["features"])
    truth_path = truth_path or os.path.join(os.path.dirname(data_path), "ground_truth.csv")
    truth = pd.read_csv(truth_path).set_index("unit") if os.path.exists(truth_path) else None
    test_units = set(saved.get("test_units", []))
    cache = {}

    def analyze(unit):
        if unit not in cache:
            g = df[df[u_col] == unit]
            if g.empty:
                return None
            cache[unit] = (g.sort_values(c_col), det.score_unit(g, c_col))
        return cache[unit]

    def r(v, nd=3):
        return None if v is None else round(float(v), nd)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.get("/api/model")
    def model_info():
        keys = ("features", "window", "latent", "hidden", "alpha", "median_window", "warmup", "patience", "critical_mult",
                "healthy_fraction", "threshold", "failure_level")
        m = saved or {}
        return jsonify(config={k: cfg.get(k) for k in keys}, autoencoder=m.get("autoencoder"), pca_baseline=m.get("pca_baseline"),
                       advice=ADVICE)

    @app.get("/api/units")
    def units():
        out = []
        for u in sorted(df[u_col].unique()):
            _, res = analyze(u)
            row = {"unit": int(u), "cycles": len(res["cycles"]), "alarm_cycle": res["alarm_cycle"],
                   "final_status": res["status"][-1], "is_test": int(u) in test_units}
            if truth is not None and u in truth.index:
                row["true_fault"] = truth.loc[u, "failure_mode"]
            out.append(row)
        return jsonify(sorted(out, key=lambda x: (not x["is_test"], x["unit"])))

    @app.get("/api/unit/<int:unit>")
    def unit_detail(unit):
        got = analyze(unit)
        if got is None:
            return jsonify(error="Unknown machine."), 404
        g, res = got
        feats = cfg["features"]
        n = len(res["cycles"])
        out = {
            "unit": unit, "cycles": res["cycles"], "status": res["status"], "health": [r(v) for v in res["ratio"]],
            "sensors": {f: [r(v) for v in g[f].to_numpy()] for f in feats},
            "sensor_ratio": {f: [r(v, 2) for v in res["sensor_ratio"][:, j]] for j, f in enumerate(feats)},
            "alarm_cycle": res["alarm_cycle"], "failure_cycle": res["cycles"][-1],
            "rul_estimate": [None if v is None else round(v) for v in res["rul"]],
            "true_rul": [res["cycles"][-1] - c for c in res["cycles"]],
            "advice": ADVICE,
        }
        if truth is not None and unit in truth.index:
            out["truth"] = {"fault": truth.loc[unit, "failure_mode"], "degrade_start": int(truth.loc[unit, "degrade_start"])}
        return jsonify(out)

    @app.post("/api/score")
    def score():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify(error="Choose a CSV file first."), 400
        try:
            up = pd.read_csv(io.BytesIO(f.read()))
            if len(up) > MAX_ROWS:
                raise ValueError(f"The file has too many rows (limit {MAX_ROWS:,}).")
            missing = [c for c in cfg["features"] if c not in up.columns]
            if missing:
                raise ValueError(f"The file is missing sensor columns: {', '.join(missing)}")
            if u_col not in up.columns:
                up[u_col] = 1
            if c_col not in up.columns:
                up[c_col] = up.groupby(u_col).cumcount() + 1
            up[cfg["features"]] = up[cfg["features"]].apply(pd.to_numeric, errors="coerce")
            if up[cfg["features"]].isna().any().any():
                up[cfg["features"]] = up.groupby(u_col)[cfg["features"]].transform(lambda s: s.ffill().bfill())
            if up[cfg["features"]].isna().any().any():
                raise ValueError("Some sensor columns contain no valid numbers.")
            up = up.sort_values([u_col, c_col])
            results = det.score_frame(up, u_col, c_col)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            return jsonify(error=str(e)), 400
        summary, rows = [], []
        for u, res in results.items():
            last = len(res["cycles"]) - 1
            summary.append({"unit": int(u), "cycles": last + 1, "final_status": res["status"][last], "alarm_cycle": res["alarm_cycle"],
                            "health_now": r(res["ratio"][last]), "advice": ADVICE[res["status"][last]],
                            "top_sensors": [{"sensor": s, "times_normal": round(v, 1)} for s, v in det.top_sensors(res, last)],
                            "estimated_rul": None if res["rul"][last] is None else round(res["rul"][last])})
            for i, cyc in enumerate(res["cycles"]):
                rows.append({u_col: u, c_col: cyc, "health_index": r(res["ratio"][i]), "status": res["status"][i],
                             "estimated_rul": None if res["rul"][i] is None else round(res["rul"][i]),
                             "most_abnormal_sensor": det.top_sensors(res, i, 1)[0][0]})
        return jsonify(units=summary, csv=pd.DataFrame(rows).to_csv(index=False))

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="models")
    ap.add_argument("--data", default="data/sensor_data.csv")
    a = ap.parse_args()
    application = create_app(a.model, a.data)
    print(f"Open http://{a.host}:{a.port} in your browser (Ctrl+C to stop)")
    application.run(host=a.host, port=a.port, debug=False, threaded=True)
