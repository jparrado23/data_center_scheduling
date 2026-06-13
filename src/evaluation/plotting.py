"""Plotting helpers for notebook result inspection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_hourly_profiles(hourly_results: pd.DataFrame) -> plt.Figure:
    """Plot the solved load and supply profiles on a single time-series chart.

    The figure overlays flexible load, optional fixed baseline, total load,
    renewable availability, renewable usage, curtailment, and grid residual
    demand to make the energy balance visible.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    hour = hourly_results["hour"]

    if "baseline_load" in hourly_results.columns:
        ax.plot(hour, hourly_results["baseline_load"], linestyle=":", label="Fixed baseline load")
    if "flexible_load" in hourly_results.columns:
        ax.plot(hour, hourly_results["flexible_load"], marker="o", label="Flexible workload load")
    ax.plot(hour, hourly_results["total_load"], marker="o", linewidth=2, label="Total facility load")
    ax.plot(hour, hourly_results["renewable_available"], linestyle="--", label="Renewable availability")
    ax.plot(hour, hourly_results["renewable_consumption"], marker="s", label="Renewable consumption")
    if "renewable_curtailment" in hourly_results.columns:
        ax.plot(hour, hourly_results["renewable_curtailment"], linestyle="--", label="Renewable curtailment")
    ax.plot(hour, hourly_results["grid_consumption"], marker="s", label="Grid consumption")

    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.set_title("MILP Schedule Load and Energy-Source Profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_cluster_loads(cluster_hourly_results: pd.DataFrame) -> plt.Figure:
    """Plot hourly load for each cluster against its capacity."""

    clusters = list(cluster_hourly_results["cluster_id"].unique())
    fig, axes = plt.subplots(len(clusters), 1, figsize=(12, 2.5 * len(clusters)), sharex=True)
    if len(clusters) == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters, strict=True):
        data = cluster_hourly_results[cluster_hourly_results["cluster_id"] == cluster]
        capacity = float(data["capacity"].iloc[0])
        ax.step(data["hour"], data["cluster_load"], where="mid", label="Cluster load")
        ax.axhline(capacity, color="tab:red", linestyle="--", label="Capacity")
        ax.set_title(cluster)
        ax.set_ylabel("Power (MW)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Hour")
    fig.tight_layout()
    return fig


def plot_schedule_gantt(schedule_df: pd.DataFrame) -> plt.Figure:
    """Plot a simple Gantt chart of selected jobs by assigned cluster."""

    ordered = schedule_df.sort_values(["assigned_cluster", "start_hour", "job_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(ordered))))

    category_codes = {category: idx for idx, category in enumerate(sorted(ordered["category"].unique()))}
    colors = plt.get_cmap("tab10")

    for row_index, row in ordered.iterrows():
        ax.barh(
            row_index,
            row["duration"],
            left=row["start_hour"],
            color=colors(category_codes[row["category"]] % 10),
            edgecolor="black",
            alpha=0.85,
        )
        ax.text(
            row["start_hour"] + row["duration"] / 2,
            row_index,
            row["job_id"],
            ha="center",
            va="center",
            fontsize=8,
        )

    y_labels = [f"{row.assigned_cluster}: {row.job_id}" for row in ordered.itertuples(index=False)]
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Hour")
    ax.set_title("Selected Job Schedule")
    ax.grid(True, axis="x", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors(code % 10), label=category)
        for category, code in category_codes.items()
    ]
    ax.legend(handles=handles, title="Category", loc="upper right")
    fig.tight_layout()
    return fig
