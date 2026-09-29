import pandas as pd
import pytest

from src.data.gpu_unit_adapter import adapt_gpu_unit


def _jobs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": "j1",
                "gpu_count_required": 3,
                "node_units_required": 32,
                "block_units_required": 3,
            }
        ]
    )


def _clusters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cluster_id": "c1",
                "gpu_capacity": 23,
                "gpu_count": 23,
                "node_capacity": 286,
                "block_capacity": 23,
            }
        ]
    )


def test_adapt_gpu_unit_node_mode_uses_node_columns():
    jobs, clusters = adapt_gpu_unit(_jobs(), _clusters(), "node")

    assert jobs.loc[0, "gpu_count_required"] == 32
    assert clusters.loc[0, "gpu_capacity"] == 286
    assert clusters.loc[0, "gpu_count"] == 286


def test_adapt_gpu_unit_block_mode_uses_block_columns():
    jobs, clusters = adapt_gpu_unit(_jobs(), _clusters(), "block")

    assert jobs.loc[0, "gpu_count_required"] == 3
    assert clusters.loc[0, "gpu_capacity"] == 23
    assert clusters.loc[0, "gpu_count"] == 23


def test_adapt_gpu_unit_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        adapt_gpu_unit(_jobs(), _clusters(), "raw")  # type: ignore[arg-type]
