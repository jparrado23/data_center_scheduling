"""Synthetic data generators for toy MILP examples."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ModelConfig


def generate_toy_jobs() -> pd.DataFrame:
    """Create a small deterministic job table for notebooks and tests.

    The returned frame contains a handful of flexible workloads with distinct
    durations, power demands, and admissible start windows so that the toy MILP
    instance is easy to inspect by hand.
    """

    return pd.DataFrame(
        [
            {
                "job_id": "train_small",
                "category": "training",
                "duration": 3,
                "power": 18.0,
                "earliest_start": 0,
                "latest_start": 10,
            },
            {
                "job_id": "batch_eval",
                "category": "inference",
                "duration": 2,
                "power": 10.0,
                "earliest_start": 4,
                "latest_start": 16,
            },
            {
                "job_id": "preprocess",
                "category": "data_processing",
                "duration": 4,
                "power": 8.0,
                "earliest_start": 0,
                "latest_start": 12,
            },
            {
                "job_id": "fine_tune",
                "category": "training",
                "duration": 3,
                "power": 14.0,
                "earliest_start": 8,
                "latest_start": 20,
            },
        ]
    )


def generate_toy_clusters() -> pd.DataFrame:
    """Create four heterogeneous clusters with capacities and compatibility."""

    return pd.DataFrame(
        [
            {
                "cluster_id": "gpu_training_1",
                "capacity": 32.0,
                "compatible_categories": ["training"],
            },
            {
                "cluster_id": "gpu_training_2",
                "capacity": 24.0,
                "compatible_categories": ["training", "inference"],
            },
            {
                "cluster_id": "gpu_inference",
                "capacity": 22.0,
                "compatible_categories": ["inference"],
            },
            {
                "cluster_id": "cpu_batch",
                "capacity": 16.0,
                "compatible_categories": ["data_processing"],
            },
        ]
    )


def generate_toy_hourly_inputs(num_hours: int = 24) -> pd.DataFrame:
    """Create deterministic renewable and grid-price profiles.

    The resulting time series is smooth enough to be visually interpretable in
    plots while still producing a nontrivial optimization problem.
    """

    hours = np.arange(num_hours)
    renewable_available = np.maximum(0.0, 34.0 * np.sin((hours - 6) / 12 * np.pi))
    grid_price = np.where((hours >= 17) & (hours <= 21), 145.0, 85.0)
    grid_price = np.where((hours >= 0) & (hours <= 5), 65.0, grid_price)

    return pd.DataFrame(
        {
            "hour": hours,
            "renewable_available": renewable_available.round(3),
            "grid_price": grid_price.astype(float),
        }
    )


def generate_toy_config() -> ModelConfig:
    """Create a compact configuration that keeps the toy instance feasible.

    The values are tuned so the generated data fit within the contracted power
    and cluster-capacity assumptions used by the notebooks.
    """

    return ModelConfig(
        contracted_power=70.0,
        renewable_price=50.0,
        peak_price=1000.0,
        delta_t=1.0,
    )


def generate_toy_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelConfig]:
    """Return the complete synthetic dataset used throughout the examples.

    This is a convenience wrapper around the individual generators so tests and
    notebooks can bootstrap the same deterministic instance with one call.
    """

    return generate_toy_jobs(), generate_toy_hourly_inputs(), generate_toy_clusters(), generate_toy_config()
