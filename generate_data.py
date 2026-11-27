"""Create the sample dataset (data/sensor_data.csv and data/ground_truth.csv).
   python generate_data.py [--units 60] [--seed 7] [--out data]"""
import runpy

runpy.run_module("pdm.simulate", run_name="__main__")
