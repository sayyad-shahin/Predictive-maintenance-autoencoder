import os, subprocess, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import torch
from pdm import fit_pipeline, load_detector, save_detector, simulate
from pdm.data import Scaler, healthy_part, load_csv, make_windows, split_units
from pdm.detect import estimate_rul, ewma, first_alarm, rolling_median, status_series
from pdm.evaluate import evaluate_detector
from pdm.model import Autoencoder, train_autoencoder

FEATS = ["load", "bearing_temp", "vibration", "motor_current", "oil_pressure", "outlet_pressure", "flow_rate", "rpm", "coolant_temp"]


class TestSimulation(unittest.TestCase):
    def test_shape_truth_and_repeatability(self):
        d, t = simulate(12, seed=1)
        self.assertEqual(list(d.columns), ["unit", "cycle", "load"] + FEATS[1:])
        self.assertEqual(d.groupby("unit").size().tolist(), t["life"].tolist())
        self.assertTrue(((t.degrade_start > 0.5 * t.life) & (t.degrade_start < 0.8 * t.life)).all())
        self.assertTrue(simulate(12, seed=1)[0].equals(d))
        self.assertFalse(simulate(12, seed=2)[0].equals(d))

    def test_faults_change_the_right_sensors(self):
        d, t = simulate(40, seed=3)
        for _, row in t[t.failure_mode == "bearing_wear"].iterrows():
            g = d[d.unit == row.unit]
            self.assertGreater(g.vibration.iloc[-10:].mean(), g.vibration.iloc[:40].mean() + 0.8)
        for _, row in t[t.failure_mode == "pump_wear"].iterrows():
            g = d[d.unit == row.unit]
            self.assertLess(g.outlet_pressure.iloc[-10:].mean(), g.outlet_pressure.iloc[:40].mean() - 0.3)


class TestData(unittest.TestCase):
    def test_scaler(self):
        X = np.random.default_rng(0).normal(5, 3, (500, 4))
        s = Scaler.fit(X); Z = s.transform(X)
        self.assertTrue(np.allclose(Z.mean(0), 0, atol=1e-9)); self.assertTrue(np.allclose(Z.std(0), 1))
        self.assertTrue(np.allclose(Scaler.from_dict(s.to_dict()).transform(X), Z))
        self.assertTrue(np.isfinite(Scaler.fit(np.ones((5, 2))).transform(np.ones((5, 2)))).all())   # constant sensor

    def test_windows(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        W = make_windows(X, 3)
        self.assertEqual(W.shape, (5, 6))
        self.assertEqual(W[0].tolist(), [0, 1, 0, 1, 0, 1])                  # start padded with the first row
        self.assertEqual(W[4].tolist(), [4, 5, 6, 7, 8, 9])                  # last row sees the 3 latest cycles
        self.assertTrue(np.array_equal(make_windows(X, 1), X))

    def test_split(self):
        a, b, c = split_units(range(1, 61), seed=0)
        self.assertEqual(len(set(a) | set(b) | set(c)), 60); self.assertFalse(set(a) & set(b) or set(a) & set(c) or set(b) & set(c))
        self.assertEqual(len(c), 15)
        self.assertEqual(split_units(range(1, 61), seed=0), (a, b, c))
        with self.assertRaises(ValueError): split_units([1, 2])

    def test_healthy_part(self):
        self.assertEqual(int(healthy_part(100, 0.4).sum()), 40)

    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.csv")
            pd.DataFrame({"unit": [1, 1, 1], "cycle": [3, 1, 2], "a": [1.0, np.nan, 3.0], "b": [1, 2, 3]}).to_csv(p, index=False)
            df, feats = load_csv(p)
            self.assertEqual(feats, ["a", "b"]); self.assertEqual(df.cycle.tolist(), [1, 2, 3]); self.assertFalse(df.a.isna().any())
            with self.assertRaises(ValueError): load_csv(p, unit_col="machine")
            with self.assertRaises(ValueError): load_csv(p, features=["zzz"])


class TestDetectionLogic(unittest.TestCase):
    def test_ewma_and_median(self):
        self.assertEqual(ewma([1, 1, 1], 0.5).tolist(), [1, 1, 1])
        spike = [1, 1, 50, 1, 1, 1]
        self.assertLess(rolling_median(spike, 5).max(), 2)                  # a single glitch is ignored
        self.assertGreater(ewma(spike, 0.5).max(), 10)

    def test_alarm_needs_patience(self):
        h = [0.5, 2, 0.5, 2, 2, 2, 2]
        self.assertEqual(first_alarm(h, 1, 3), 5)                           # the early blip does not count
        self.assertIsNone(first_alarm([0.5, 2, 0.5, 2, 0.5], 1, 2)); self.assertEqual(first_alarm(h, 1, 1), 1)

    def test_status_levels_and_latch(self):
        h = [0.1, 0.8, 1.5, 1.5, 1.5, 0.2, 4.0]
        s = status_series(h, 1.0, 3, 3.0)
        self.assertEqual(s, ["normal", "watch", "watch", "watch", "warning", "warning", "critical"])   # alarm stays on after dropping

    def test_rul_estimate(self):
        cycles = np.arange(1, 101)
        health = (0.02 * cycles) ** 2                                       # sqrt(health) grows 0.02 per cycle
        est = estimate_rul(cycles, health, 50, failure_level=(0.02 * 130) ** 2)
        self.assertAlmostEqual(est[79], 130 - 80, delta=1.0)
        self.assertIsNone(est[10])
        self.assertTrue(all(v is None for v in estimate_rul(cycles, health, None, 5.0)))
        self.assertTrue(all(v is None for v in estimate_rul(cycles, health, 50, None)))
        self.assertTrue(all(v is None for v in estimate_rul(cycles, np.ones(100), 50, 5.0)))      # flat trend: no estimate


class TestModel(unittest.TestCase):
    def test_autoencoder_learns(self):
        rng = np.random.default_rng(0)
        z = rng.normal(size=(600, 2)); X = np.hstack([z, z @ rng.normal(size=(2, 4))]).astype(np.float32)
        model, hist = train_autoencoder(X[:500], X[500:], hidden=(16,), latent=2, epochs=80, patience=80)
        self.assertLess(hist["best_val"], 0.3 * hist["val"][0])
        self.assertEqual(model(torch.tensor(X[:3])).shape, (3, 6))
        self.assertEqual(Autoencoder(9, (32, 16), 3).latent, 3)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.truth = simulate(36, seed=21)
        cls.det, cls.info = fit_pipeline(cls.df, FEATS, verbose=False, epochs=60, window=3, seed=1)

    def test_calibration_and_metrics(self):
        c = self.det.cfg
        self.assertGreater(c["threshold"], 0); self.assertEqual(len(c["sensor_baseline"]), 9); self.assertGreater(c["failure_level"], c["threshold"])
        m, rows = evaluate_detector(self.det, self.df, self.info["test_units"], self.truth)
        self.assertEqual(m["machines_tested"], len(rows))
        self.assertGreaterEqual(m["detection_rate"], 0.6)
        self.assertLess(m["false_alarm_cycle_rate"], 0.05)
        self.assertGreater(m["auroc_last_50_cycles"], 0.8)
        for r in rows:
            if r["detected"]:
                self.assertEqual(r["lead_time"], r["life"] - r["alarm_cycle"]); self.assertGreater(r["lead_time"], 0)

    def test_scores_a_degrading_machine(self):
        u = self.info["test_units"][0]
        res = self.det.score_unit(self.df[self.df.unit == u])
        self.assertEqual(len(res["health"]), (self.df.unit == u).sum())
        self.assertEqual(res["status"][5], "normal"); self.assertEqual(res["status"][-1], "critical")
        top = [n for n, _ in self.det.top_sensors(res, len(res["health"]) - 1, 3)]
        mode = self.truth.set_index("unit").loc[u, "failure_mode"]
        expected = {"bearing_wear": {"vibration", "bearing_temp"}, "pump_wear": {"outlet_pressure", "flow_rate"},
                    "electrical_fault": {"motor_current", "rpm"}}[mode]
        self.assertTrue(expected & set(top), f"{mode}: top sensors were {top}")        # the root-cause sensors show up

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            save_detector(self.det, d, {"hello": 1})
            loaded, metrics = load_detector(d)
        self.assertEqual(metrics, {"hello": 1})
        g = self.df[self.df.unit == 1]
        a, b = self.det.score_unit(g), loaded.score_unit(g)
        self.assertTrue(np.allclose(a["health"], b["health"], atol=1e-5)); self.assertEqual(a["alarm_cycle"], b["alarm_cycle"])

    def test_pca_baseline_and_bad_method(self):
        det, _ = fit_pipeline(self.df, FEATS, method="pca", verbose=False, window=1)
        self.assertIsNone(det.model)
        with self.assertRaises(ValueError): save_detector(det, tempfile.mkdtemp())
        with self.assertRaises(ValueError): fit_pipeline(self.df, FEATS, method="magic", verbose=False)

    def test_no_failure_level_without_run_to_failure(self):
        det, _ = fit_pipeline(self.df, FEATS, verbose=False, epochs=5, run_to_failure=False)
        self.assertIsNone(det.cfg["failure_level"])
        self.assertTrue(all(v is None for v in det.score_unit(self.df[self.df.unit == 1])["rul"]))


class TestCommandLine(unittest.TestCase):
    def run_cmd(self, *args):
        return subprocess.run([sys.executable, "-W", "ignore", *args], cwd=ROOT, capture_output=True, text=True, timeout=300)

    def test_predict(self):
        with tempfile.TemporaryDirectory() as d:
            src = pd.read_csv(os.path.join(ROOT, "data", "sensor_data.csv"))
            p, out = os.path.join(d, "in.csv"), os.path.join(d, "out.csv")
            src[src.unit.isin([3, 5])].to_csv(p, index=False)
            r = self.run_cmd("predict.py", p, "--out", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            res = pd.read_csv(out)
            self.assertEqual(sorted(res.unit.unique()), [3, 5]); self.assertIn("status", res.columns)
            self.assertEqual(res[res.unit == 3].status.iloc[-1], "critical")
            self.assertIn("Machine 3", r.stdout)
            bad = os.path.join(d, "bad.csv"); src[["unit", "cycle", "load"]].head(5).to_csv(bad, index=False)
            self.assertNotEqual(self.run_cmd("predict.py", bad).returncode, 0)

    def test_train_small_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            data, truth = simulate(12, seed=5)
            data.to_csv(os.path.join(d, "s.csv"), index=False); truth.to_csv(os.path.join(d, "ground_truth.csv"), index=False)
            r = self.run_cmd("train.py", "--data", os.path.join(d, "s.csv"), "--truth", os.path.join(d, "ground_truth.csv"),
                             "--epochs", "5", "--out", os.path.join(d, "m"), "--plots", "")
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertTrue(os.path.exists(os.path.join(d, "m", "autoencoder.pt")))
            self.assertEqual(self.run_cmd("train.py", "--data", os.path.join(d, "nope.csv")).returncode, 2)
            self.assertEqual(self.run_cmd("train.py", "--data", os.path.join(d, "s.csv"), "--unit-col", "machine").returncode, 2)


if __name__ == "__main__":
    unittest.main()
