"""Placeholders for future real-data loading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import ModelConfig


def load_processed_inputs(data_dir: str | Path = "data/processed") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelConfig]:
    """Load processed jobs, hourly inputs, clusters, and configuration from disk.

    The loader expects `jobs.csv`, `hourly_inputs.csv`, `clusters.csv`, and
    `config.json` in the target directory.
    """

    data_path = Path(data_dir)
    jobs_df = pd.read_csv(data_path / "jobs.csv")
    hourly_df = pd.read_csv(data_path / "hourly_inputs.csv")
    clusters_df = pd.read_csv(data_path / "clusters.csv")
    with (data_path / "config.json").open("r", encoding="utf-8") as file:
        config_data = json.load(file)
    return jobs_df, hourly_df, clusters_df, ModelConfig(**config_data)
