"""Import OMIE MARGINALPDBC day-ahead price files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.data.validation import validate_hourly_inputs


PriceColumn = Literal["first", "last", "average"]


def _select_price(row: pd.Series, price_column: PriceColumn) -> float:
    if price_column == "first":
        return float(row["price_1"])
    if price_column == "last":
        return float(row["price_2"])
    if price_column == "average":
        return float((row["price_1"] + row["price_2"]) / 2.0)
    raise ValueError(f"unsupported price_column: {price_column!r}")


def load_marginalpdbc(
    path: str | Path,
    *,
    price_column: PriceColumn = "last",
    source_hours_are_one_based: bool = True,
) -> pd.DataFrame:
    """Load one OMIE `MARGINALPDBC` text file into an hourly price dataframe.

    The source file is semicolon-delimited, with a header line and then rows of:

    `year;month;day;hour;price_1;price_2;`

    The returned dataframe keeps both source price columns and adds the selected
    model-ready `grid_price` column.
    """

    rows: list[dict[str, int | float]] = []
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line == "*" or line.upper() == "MARGINALPDBC;":
                continue

            parts = [part for part in line.split(";") if part != ""]
            if len(parts) != 6:
                raise ValueError(f"invalid MARGINALPDBC row at line {line_number}: {raw_line.rstrip()!r}")

            year, month, day, hour = (int(parts[index]) for index in range(4))
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "day": day,
                    "source_hour": hour,
                    "price_1": float(parts[4]),
                    "price_2": float(parts[5]),
                }
            )

    if len(rows) != 24:
        raise ValueError(f"expected 24 hourly price rows, found {len(rows)}")

    prices_df = pd.DataFrame(rows)
    hour_offset = 1 if source_hours_are_one_based else 0
    prices_df["hour"] = prices_df["source_hour"] - hour_offset
    if sorted(prices_df["hour"].tolist()) != list(range(24)):
        raise ValueError("price file must contain one row for each model hour 0..23")

    prices_df["grid_price"] = prices_df.apply(_select_price, axis=1, price_column=price_column)
    return prices_df[
        ["year", "month", "day", "source_hour", "hour", "price_1", "price_2", "grid_price"]
    ].sort_values("hour", ignore_index=True)


def build_hourly_inputs_from_marginalpdbc(
    path: str | Path,
    *,
    price_column: PriceColumn = "last",
    renewable_available: float = 0.0,
    source_hours_are_one_based: bool = True,
) -> pd.DataFrame:
    """Create a model-ready `hourly_inputs.csv` table from a price file."""

    prices_df = load_marginalpdbc(
        path,
        price_column=price_column,
        source_hours_are_one_based=source_hours_are_one_based,
    )
    hourly_df = pd.DataFrame(
        {
            "hour": prices_df["hour"].astype(int),
            "renewable_available": float(renewable_available),
            "grid_price": prices_df["grid_price"].astype(float),
        }
    )
    validate_hourly_inputs(hourly_df)
    return hourly_df


def update_hourly_inputs_prices(
    hourly_inputs_df: pd.DataFrame,
    price_path: str | Path,
    *,
    price_column: PriceColumn = "last",
    source_hours_are_one_based: bool = True,
) -> pd.DataFrame:
    """Replace `grid_price` in an existing hourly inputs table."""

    prices_df = load_marginalpdbc(
        price_path,
        price_column=price_column,
        source_hours_are_one_based=source_hours_are_one_based,
    )
    updated_df = hourly_inputs_df.copy()
    price_by_hour = dict(zip(prices_df["hour"].astype(int), prices_df["grid_price"].astype(float), strict=True))
    updated_df["grid_price"] = updated_df["hour"].astype(int).map(price_by_hour)
    if updated_df["grid_price"].isna().any():
        missing_hours = updated_df.loc[updated_df["grid_price"].isna(), "hour"].tolist()
        raise ValueError(f"missing prices for hours: {missing_hours}")
    validate_hourly_inputs(updated_df)
    return updated_df


def write_hourly_inputs_from_marginalpdbc(
    price_path: str | Path,
    output_path: str | Path,
    *,
    existing_hourly_inputs_path: str | Path | None = None,
    price_column: PriceColumn = "last",
    renewable_available: float = 0.0,
    source_hours_are_one_based: bool = True,
) -> pd.DataFrame:
    """Write model-ready hourly inputs using OMIE grid prices."""

    if existing_hourly_inputs_path is None:
        hourly_df = build_hourly_inputs_from_marginalpdbc(
            price_path,
            price_column=price_column,
            renewable_available=renewable_available,
            source_hours_are_one_based=source_hours_are_one_based,
        )
    else:
        existing_df = pd.read_csv(existing_hourly_inputs_path)
        hourly_df = update_hourly_inputs_prices(
            existing_df,
            price_path,
            price_column=price_column,
            source_hours_are_one_based=source_hours_are_one_based,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly_df.to_csv(output, index=False)
    return hourly_df
