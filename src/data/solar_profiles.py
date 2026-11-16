"""Import PVGIS solar generation time series and build hourly profiles."""

from __future__ import annotations

from pathlib import Path
from datetime import date

import pandas as pd

from src.data.validation import validate_hourly_inputs


PVGIS_DATA_HEADER = "time,"


def _find_pvgis_data_start(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if line.startswith(PVGIS_DATA_HEADER):
            return index
    raise ValueError("PVGIS file does not contain a time-series header")


def load_pvgis_timeseries(
    path: str | Path,
    *,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
) -> pd.DataFrame:
    """Load a PVGIS hourly time-series CSV and return normalized solar data.

    `P` is the simulated PV system output in W. The returned `power_mw` column
    is scaled to `target_capacity_kwp` when provided; otherwise it keeps the
    source file's nominal capacity.
    """

    source_path = Path(path)
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    data_start = _find_pvgis_data_start(lines)
    raw_df = pd.read_csv(source_path, skiprows=data_start)
    raw_df = raw_df[raw_df["time"].astype(str).str.match(r"^\d{8}:\d{4}$", na=False)].copy()
    if raw_df.empty:
        raise ValueError("PVGIS file does not contain any hourly data rows")

    timestamp = pd.to_datetime(raw_df["time"], format="%Y%m%d:%H%M")
    scale = 1.0 if target_capacity_kwp is None else float(target_capacity_kwp) / float(base_capacity_kwp)

    solar_df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "year": timestamp.dt.year.astype(int),
            "month": timestamp.dt.month.astype(int),
            "day": timestamp.dt.day.astype(int),
            "hour": timestamp.dt.hour.astype(int),
            "source_power_w": raw_df["P"].astype(float),
            "power_w": raw_df["P"].astype(float) * scale,
        }
    )
    solar_df["power_mw"] = solar_df["power_w"] / 1_000_000.0
    return solar_df


def build_monthly_hourly_solar_profile(
    path: str | Path,
    *,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
) -> pd.DataFrame:
    """Build monthly average solar availability for each hour of day."""

    solar_df = load_pvgis_timeseries(
        path,
        base_capacity_kwp=base_capacity_kwp,
        target_capacity_kwp=target_capacity_kwp,
    )
    profile_df = (
        solar_df.groupby(["month", "hour"], as_index=False)
        .agg(
            renewable_available=("power_mw", "mean"),
            avg_power_kw=("power_w", lambda values: float(values.mean() / 1000.0)),
            sample_count=("power_mw", "size"),
        )
        .sort_values(["month", "hour"], ignore_index=True)
    )

    expected_pairs = pd.MultiIndex.from_product([range(1, 13), range(24)], names=["month", "hour"])
    observed_pairs = pd.MultiIndex.from_frame(profile_df[["month", "hour"]])
    missing_pairs = expected_pairs.difference(observed_pairs)
    if len(missing_pairs) > 0:
        raise ValueError(f"monthly profile is missing month-hour pairs: {list(missing_pairs)[:5]}")
    return profile_df


def build_monthly_hourly_inputs(
    path: str | Path,
    month: int,
    *,
    grid_price: float = 0.0,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
) -> pd.DataFrame:
    """Build a 24-row hourly input table for one monthly solar profile."""

    if month < 1 or month > 12:
        raise ValueError("month must be between 1 and 12")
    profile_df = build_monthly_hourly_solar_profile(
        path,
        base_capacity_kwp=base_capacity_kwp,
        target_capacity_kwp=target_capacity_kwp,
    )
    month_df = profile_df.loc[profile_df["month"] == month, ["hour", "renewable_available"]].copy()
    month_df["grid_price"] = float(grid_price)
    hourly_df = month_df[["hour", "renewable_available", "grid_price"]].reset_index(drop=True)
    validate_hourly_inputs(hourly_df)
    return hourly_df


def build_daily_hourly_solar_profile(
    path: str | Path,
    target_date: str | date,
    *,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
) -> pd.DataFrame:
    """Build the actual 24-hour solar availability profile for one date."""

    parsed_date = pd.to_datetime(target_date).date()
    solar_df = load_pvgis_timeseries(
        path,
        base_capacity_kwp=base_capacity_kwp,
        target_capacity_kwp=target_capacity_kwp,
    )
    day_df = solar_df.loc[solar_df["timestamp"].dt.date == parsed_date].copy()
    if len(day_df) != 24:
        raise ValueError(f"expected 24 hourly rows for {parsed_date}, found {len(day_df)}")
    return (
        day_df[["year", "month", "day", "hour", "power_mw"]]
        .rename(columns={"power_mw": "renewable_available"})
        .sort_values("hour", ignore_index=True)
    )


def build_daily_hourly_inputs(
    path: str | Path,
    target_date: str | date,
    *,
    grid_price: float = 0.0,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
) -> pd.DataFrame:
    """Build a 24-row hourly input table for one actual PVGIS date."""

    profile_df = build_daily_hourly_solar_profile(
        path,
        target_date,
        base_capacity_kwp=base_capacity_kwp,
        target_capacity_kwp=target_capacity_kwp,
    )
    hourly_df = profile_df[["hour", "renewable_available"]].copy()
    hourly_df["grid_price"] = float(grid_price)
    hourly_df = hourly_df[["hour", "renewable_available", "grid_price"]].reset_index(drop=True)
    validate_hourly_inputs(hourly_df)
    return hourly_df


def update_hourly_inputs_solar(
    hourly_inputs_df: pd.DataFrame,
    solar_profile_df: pd.DataFrame,
    month: int,
) -> pd.DataFrame:
    """Replace `renewable_available` in an existing hourly inputs table."""

    month_profile = solar_profile_df.loc[solar_profile_df["month"] == month]
    if month_profile.empty:
        raise ValueError(f"solar profile does not contain month {month}")

    renewable_by_hour = dict(
        zip(month_profile["hour"].astype(int), month_profile["renewable_available"].astype(float), strict=True)
    )
    updated_df = hourly_inputs_df.copy()
    updated_df["renewable_available"] = updated_df["hour"].astype(int).map(renewable_by_hour)
    if updated_df["renewable_available"].isna().any():
        missing_hours = updated_df.loc[updated_df["renewable_available"].isna(), "hour"].tolist()
        raise ValueError(f"missing solar availability for hours: {missing_hours}")
    validate_hourly_inputs(updated_df)
    return updated_df


def update_hourly_inputs_daily_solar(
    hourly_inputs_df: pd.DataFrame,
    pvgis_path: str | Path,
    target_date: str | date,
    *,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
) -> pd.DataFrame:
    """Replace `renewable_available` with one actual date from PVGIS."""

    day_profile = build_daily_hourly_solar_profile(
        pvgis_path,
        target_date,
        base_capacity_kwp=base_capacity_kwp,
        target_capacity_kwp=target_capacity_kwp,
    )
    renewable_by_hour = dict(
        zip(day_profile["hour"].astype(int), day_profile["renewable_available"].astype(float), strict=True)
    )
    updated_df = hourly_inputs_df.copy()
    updated_df["renewable_available"] = updated_df["hour"].astype(int).map(renewable_by_hour)
    if updated_df["renewable_available"].isna().any():
        missing_hours = updated_df.loc[updated_df["renewable_available"].isna(), "hour"].tolist()
        raise ValueError(f"missing solar availability for hours: {missing_hours}")
    validate_hourly_inputs(updated_df)
    return updated_df


def write_monthly_hourly_solar_profile(
    pvgis_path: str | Path,
    output_path: str | Path,
    *,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
) -> pd.DataFrame:
    """Write all monthly average hourly solar profiles to CSV."""

    profile_df = build_monthly_hourly_solar_profile(
        pvgis_path,
        base_capacity_kwp=base_capacity_kwp,
        target_capacity_kwp=target_capacity_kwp,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(output, index=False)
    return profile_df


def write_monthly_hourly_inputs(
    pvgis_path: str | Path,
    output_path: str | Path,
    month: int,
    *,
    existing_hourly_inputs_path: str | Path | None = None,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
    grid_price: float = 0.0,
) -> pd.DataFrame:
    """Write one month's 24-hour solar profile as model-ready hourly inputs."""

    if existing_hourly_inputs_path is None:
        hourly_df = build_monthly_hourly_inputs(
            pvgis_path,
            month,
            grid_price=grid_price,
            base_capacity_kwp=base_capacity_kwp,
            target_capacity_kwp=target_capacity_kwp,
        )
    else:
        profile_df = build_monthly_hourly_solar_profile(
            pvgis_path,
            base_capacity_kwp=base_capacity_kwp,
            target_capacity_kwp=target_capacity_kwp,
        )
        existing_df = pd.read_csv(existing_hourly_inputs_path)
        hourly_df = update_hourly_inputs_solar(existing_df, profile_df, month)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(output, index=False)
    return hourly_df


def write_daily_hourly_inputs(
    pvgis_path: str | Path,
    output_path: str | Path,
    target_date: str | date,
    *,
    existing_hourly_inputs_path: str | Path | None = None,
    base_capacity_kwp: float = 250.0,
    target_capacity_kwp: float | None = None,
    grid_price: float = 0.0,
) -> pd.DataFrame:
    """Write one actual PVGIS date as model-ready hourly inputs."""

    if existing_hourly_inputs_path is None:
        hourly_df = build_daily_hourly_inputs(
            pvgis_path,
            target_date,
            grid_price=grid_price,
            base_capacity_kwp=base_capacity_kwp,
            target_capacity_kwp=target_capacity_kwp,
        )
    else:
        existing_df = pd.read_csv(existing_hourly_inputs_path)
        hourly_df = update_hourly_inputs_daily_solar(
            existing_df,
            pvgis_path,
            target_date,
            base_capacity_kwp=base_capacity_kwp,
            target_capacity_kwp=target_capacity_kwp,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(output, index=False)
    return hourly_df
