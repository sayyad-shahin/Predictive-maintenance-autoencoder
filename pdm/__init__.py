"""Predictive maintenance with an autoencoder: learn what 'healthy' looks like, flag machines that drift away from it."""
from .simulate import SENSORS, simulate
from .detect import Detector
from .pipeline import fit_pipeline, load_detector, save_detector

__all__ = ["SENSORS", "simulate", "Detector", "fit_pipeline", "load_detector", "save_detector"]
