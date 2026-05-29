"""Configuration objects for the data-center scheduling model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Core scalar parameters used by the MILP model."""

    contracted_power: float = 100.0
    renewable_price: float = 50.0
    peak_price: float = 1000.0
    delta_t: float = 1.0
