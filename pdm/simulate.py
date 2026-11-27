"""Simulated run-to-failure data for a fleet of pumps.

Each machine ("unit") runs for a few hundred cycles under changing load until it fails. For the first part of its life
it is healthy. Then one of three faults starts to develop and grows until failure:
  bearing_wear     -> more vibration and bearing heat
  pump_wear        -> lower outlet pressure and flow
  electrical_fault -> higher motor current, lower speed, warmer coolant
Sensors are noisy, depend (non-linearly) on the load, differ a little from machine to machine, and have rare spikes.

This is SYNTHETIC data so the project runs anywhere. To use real data, see the README.
"""
import numpy as np
import pandas as pd

SENSORS = ["bearing_temp", "vibration", "motor_current", "oil_pressure", "outlet_pressure", "flow_rate", "rpm", "coolant_temp"]
NOISE = {"bearing_temp": 0.6, "vibration": 0.10, "motor_current": 0.25, "oil_pressure": 0.05,
         "outlet_pressure": 0.07, "flow_rate": 1.2, "rpm": 4.0, "coolant_temp": 0.4}
MODES = ["bearing_wear", "pump_wear", "electrical_fault"]


def _healthy(load):
    """Expected sensor values at a given load (percent), including mild non-linear behaviour."""
    return {
        "bearing_temp": 48 + 0.20 * load + 0.0006 * load ** 2,
        "vibration": 1.5 + 0.006 * load + 0.45 * np.exp(-((load - 75) / 9) ** 2),        # resonance bump near 75% load
        "motor_current": 6 + 0.07 * load + 0.0004 * load ** 2,
        "oil_pressure": 4.2 - 0.004 * load,
        "outlet_pressure": 7.5 - 0.012 * load - 0.00006 * load ** 2,
        "flow_rate": 120 + 0.35 * load - 0.0008 * load ** 2,
        "rpm": 1480 - 0.35 * load,
        "coolant_temp": 30 + 0.10 * load,
    }


def simulate(n_units=60, seed=7):
    """Return (data, truth). data: one row per unit and cycle. truth: when each unit started to degrade and why."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for unit in range(1, n_units + 1):
        life = int(np.clip(rng.normal(230, 40), 150, 330))
        start = int(life * rng.uniform(0.55, 0.75))
        mode = MODES[rng.integers(len(MODES))]
        severity = rng.uniform(0.8, 1.4)
        shape = rng.uniform(1.3, 2.0)
        offset = {s: rng.normal(0, 1.5 * NOISE[s]) for s in SENSORS}      # every machine is slightly different
        level = rng.choice([50, 75, 100])
        for t in range(1, life + 1):
            if rng.random() < 0.08:
                level = rng.choice([50, 75, 100])
            load = level + rng.normal(0, 3)
            v = _healthy(load)
            g = ((t - start) / (life - start)) ** shape if t > start else 0.0
            if mode == "bearing_wear":
                v["vibration"] += 3.2 * g * severity
                v["bearing_temp"] += 14 * g * severity
                v["motor_current"] += 0.8 * g
            elif mode == "pump_wear":
                v["outlet_pressure"] -= 1.4 * g * severity
                v["flow_rate"] -= 22 * g * severity
                v["oil_pressure"] -= 0.5 * g
                v["motor_current"] += 0.6 * g
            else:
                v["motor_current"] += 3.0 * g * severity
                v["rpm"] -= 45 * g * severity
                v["coolant_temp"] += 6 * g * severity
            v["bearing_temp"] += 3 * g                                       # every fault warms the bearing a little
            row = {"unit": unit, "cycle": t, "load": round(float(load), 2)}
            for s in SENSORS:
                sd = NOISE[s] * (1 + 1.2 * g * (s in ("vibration", "motor_current")))
                x = v[s] + offset[s] + rng.normal(0, sd)
                if rng.random() < 0.003:                                     # rare sensor glitch
                    x += rng.choice([-1, 1]) * 6 * NOISE[s]
                row[s] = round(float(x), 3)
            rows.append(row)
        truth.append({"unit": unit, "life": life, "degrade_start": start, "failure_mode": mode, "severity": round(severity, 2)})
    return pd.DataFrame(rows), pd.DataFrame(truth)


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Generate simulated run-to-failure sensor data")
    ap.add_argument("--units", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    d, t = simulate(a.units, a.seed)
    os.makedirs(a.out, exist_ok=True)
    d.to_csv(os.path.join(a.out, "sensor_data.csv"), index=False)
    t.to_csv(os.path.join(a.out, "ground_truth.csv"), index=False)
    print(f"Wrote {len(d):,} rows from {a.units} machines to {a.out}/sensor_data.csv (+ ground_truth.csv)")
