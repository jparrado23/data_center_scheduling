import sys

from scripts.solve_thesis_scenario import parse_args
from src.data.scenarios import CLUSTER_MODE_ALIBABA_GPU_TYPES


def test_parse_args_uses_exported_default_cluster_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["solve_thesis_scenario.py"])

    args = parse_args()

    assert args.cluster_mode == CLUSTER_MODE_ALIBABA_GPU_TYPES

