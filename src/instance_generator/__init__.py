"""Feasible synthetic instance generation for Gurobi stress tests."""

from src.instance_generator.generator import (
    FeasibilityReport,
    SAMPLING_MODE_ALIBABA,
    SAMPLING_MODE_PARAMETRIC,
    SyntheticInstance,
    SyntheticInstanceConfig,
    generate_feasible_instance,
)
from src.instance_generator.hard_instances import (
    BenchmarkInstanceSpec,
    DifficultyConfig,
    build_energy_inputs,
    build_small_clusters,
    create_hard_benchmark_pack,
    default_hard_benchmark_specs,
    generate_difficulty_instance,
    instance_diagnostics,
    write_hard_instance,
)

__all__ = [
    "BenchmarkInstanceSpec",
    "DifficultyConfig",
    "FeasibilityReport",
    "SAMPLING_MODE_ALIBABA",
    "SAMPLING_MODE_PARAMETRIC",
    "SyntheticInstance",
    "SyntheticInstanceConfig",
    "build_energy_inputs",
    "build_small_clusters",
    "create_hard_benchmark_pack",
    "default_hard_benchmark_specs",
    "generate_difficulty_instance",
    "generate_feasible_instance",
    "instance_diagnostics",
    "write_hard_instance",
]
