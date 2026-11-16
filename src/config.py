"""Configuration objects for the data-center scheduling model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Core scalar parameters used by the MILP model."""

    contracted_power: float = 100.0
    renewable_price: float = 50.0
    peak_price: float = 1000.0
    delta_t: float = 1.0
    pue: float = 1.0
    battery_power_capacity: float = 0.0
    battery_energy_capacity: float = 0.0
    battery_initial_soc: float = 0.0
    battery_final_soc: float | None = None
    battery_charge_efficiency: float = 0.95
    battery_discharge_efficiency: float = 0.95

    def __post_init__(self) -> None:
        if self.delta_t <= 0:
            raise ValueError("delta_t must be positive")
        if self.pue <= 0:
            raise ValueError("pue must be positive")
        if self.battery_power_capacity < 0:
            raise ValueError("battery_power_capacity must be non-negative")
        if self.battery_energy_capacity < 0:
            raise ValueError("battery_energy_capacity must be non-negative")
        if self.battery_initial_soc < 0:
            raise ValueError("battery_initial_soc must be non-negative")
        if self.battery_initial_soc > self.battery_energy_capacity:
            raise ValueError("battery_initial_soc cannot exceed battery_energy_capacity")
        if self.battery_final_soc is not None:
            if self.battery_final_soc < 0:
                raise ValueError("battery_final_soc must be non-negative")
            if self.battery_final_soc > self.battery_energy_capacity:
                raise ValueError("battery_final_soc cannot exceed battery_energy_capacity")
        if not 0 < self.battery_charge_efficiency <= 1:
            raise ValueError("battery_charge_efficiency must be in (0, 1]")
        if not 0 < self.battery_discharge_efficiency <= 1:
            raise ValueError("battery_discharge_efficiency must be in (0, 1]")
