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
                "job_id": "inference_24h",
                "category": "inference",
                "duration": 24,
                "power": 0.0011,
                "earliest_start": 0,
                "latest_start": 0,
            },
            {
                "job_id": "fine_tune_small",
                "category": "fine_tuning",
                "duration": 3,
                "power": 0.0032,
                "earliest_start": 0,
                "latest_start": 4,
            },
            {
                "job_id": "training_medium",
                "category": "training",
                "duration": 7,
                "power": 0.0062,
                "earliest_start": 0,
                "latest_start": 16,
            },
            {
                "job_id": "preprocess",
                "category": "preprocessing",
                "duration": 3,
                "power": 0.0014,
                "earliest_start": 0,
                "latest_start": 21,
            },
        ]
    )


def generate_toy_clusters() -> pd.DataFrame:
    """Create four heterogeneous clusters with capacities and compatibility."""

    return pd.DataFrame(
        [
            {
                "cluster_id": "cluster_a",
                "capacity": 0.0104,
                "compatible_categories": ["inference"],
            },
            {
                "cluster_id": "cluster_b",
                "capacity": 0.0935,
                "compatible_categories": ["fine_tuning", "training", "preprocessing"],
            },
            {
                "cluster_id": "cluster_c",
                "capacity": 0.1248,
                "compatible_categories": ["fine_tuning", "training", "preprocessing"],
            },
            {
                "cluster_id": "cluster_d",
                "capacity": 0.025,
                "compatible_categories": ["fine_tuning", "preprocessing"],
            },
        ]
    )


def generate_toy_hourly_inputs(num_hours: int = 24) -> pd.DataFrame:
    """Create deterministic renewable and grid-price profiles.

    The resulting time series is smooth enough to be visually interpretable in
    plots while still producing a nontrivial optimization problem.
    """

    hours = np.arange(num_hours)
    renewable_available = np.maximum(0.0, 0.08 * np.sin((hours - 6) / 12 * np.pi))
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

    The values are tuned so the generated data fit within the cluster-capacity
    assumptions used by the notebooks.
    """

    return ModelConfig(
        contracted_power=0.20,
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
