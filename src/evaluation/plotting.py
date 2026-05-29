"""Plotting helpers for notebook result inspection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_hourly_profiles(hourly_results: pd.DataFrame) -> plt.Figure:
    """Plot the solved load and supply profiles on a single time-series chart.

    The figure overlays scheduled workload load, renewable availability,
    renewable usage, and grid usage to make the energy balance visible.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    hour = hourly_results["hour"]

    ax.plot(hour, hourly_results["total_load"], marker="o", label="Scheduled workload load")
    ax.plot(hour, hourly_results["renewable_available"], linestyle="--", label="Renewable availability")
    ax.plot(hour, hourly_results["renewable_consumption"], marker="s", label="Renewable consumption")
    ax.plot(hour, hourly_results["grid_consumption"], marker="s", label="Grid consumption")

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.set_title("MILP Schedule Load and Energy-Source Profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
