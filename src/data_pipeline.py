"""
Data Pipeline Module for Hotel Bar Inventory Forecasting.
Handles loading raw transaction logs, verifying conservation logic,
and constructing the complete Cartesian product daily panel.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RAW_PATH = str(PROJECT_ROOT / "data" / "raw" / "bar_inventory_data.csv")
DEFAULT_PROCESSED_PATH = str(PROJECT_ROOT / "data" / "processed" / "daily_bar_consumption.csv")


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Loads raw bar inventory transaction records and parses timestamps."""
    if filepath is None:
        filepath = DEFAULT_RAW_PATH
    df = pd.read_csv(filepath)
    df["Date Time Served"] = pd.to_datetime(df["Date Time Served"])
    df["Date"] = df["Date Time Served"].dt.floor("D")  # type: ignore
    return df


def verify_conservation_logic(df: pd.DataFrame, tolerance: float = 1e-4) -> Dict[str, Any]:
    """
    Verifies the inventory conservation balance equation:
    Closing Balance == Opening Balance + Purchase - Consumed
    """
    expected_closing = df["Opening Balance (ml)"] + df["Purchase (ml)"] - df["Consumed (ml)"]
    absolute_diff = (df["Closing Balance (ml)"] - expected_closing).abs()
    violations = absolute_diff > tolerance

    results = {
        "total_records": len(df),
        "exact_matches": (absolute_diff == 0).sum(),
        "within_tolerance": (~violations).sum(),
        "violations_count": violations.sum(),
        "max_absolute_error": float(absolute_diff.max()),
        "mean_absolute_error": float(absolute_diff.mean()),
    }
    return results


def aggregate_daily_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates irregular transaction logs into daily consumption per bar, brand, and alcohol type.
    """
    daily = (
        df.groupby(["Date", "Bar Name", "Brand Name", "Alcohol Type"], as_index=False)
        .agg({"Consumed (ml)": "sum", "Purchase (ml)": "sum"})
    )
    return pd.DataFrame(daily)


def build_complete_daily_panel(
    daily_agg: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Constructs a complete Cartesian grid across (Date, Bar Name, Brand Name, Alcohol Type).
    Fills zero-consumption days explicitly with 0.0 ml so forecasting models learn intermittent demand.
    """
    item_combos = (
        daily_agg[["Bar Name", "Brand Name", "Alcohol Type"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    min_date = daily_agg["Date"].min() if start_date is None else pd.to_datetime(start_date)
    max_date = daily_agg["Date"].max() if end_date is None else pd.to_datetime(end_date)
    dates = pd.date_range(start=min_date, end=max_date, freq="D", name="Date")

    grid_rows = []
    for d in dates:
        temp = item_combos.copy()
        temp["Date"] = d
        grid_rows.append(temp)
    full_grid = pd.concat(grid_rows, ignore_index=True)

    panel = full_grid.merge(
        daily_agg,
        on=["Date", "Bar Name", "Brand Name", "Alcohol Type"],
        how="left"
    )

    panel["Consumed (ml)"] = panel["Consumed (ml)"].fillna(0.0)
    panel["Purchase (ml)"] = panel["Purchase (ml)"].fillna(0.0)
    panel = panel.sort_values(["Bar Name", "Brand Name", "Date"]).reset_index(drop=True)

    return panel


def run_pipeline(
    raw_path: Optional[str] = None,
    processed_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs the end-to-end data pipeline."""
    if raw_path is None:
        raw_path = DEFAULT_RAW_PATH
    if processed_path is None:
        processed_path = DEFAULT_PROCESSED_PATH

    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_path)
    audit = verify_conservation_logic(df)
    daily_agg = aggregate_daily_consumption(df)
    panel = build_complete_daily_panel(daily_agg)

    panel.to_csv(processed_path, index=False)
    return panel, audit


if __name__ == "__main__":
    panel, audit = run_pipeline()
    print("=== Inventory Conservation Logic Audit ===")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    print("\n=== Processed Daily Panel Summary ===")
    print(f"  Rows: {len(panel)}")
    print(f"  Date Range: {panel['Date'].min().strftime('%Y-%m-%d')} to {panel['Date'].max().strftime('%Y-%m-%d')} ({panel['Date'].nunique()} days)")
    print(f"  Unique Bars: {panel['Bar Name'].nunique()}")
    print(f"  Unique Brands: {panel['Brand Name'].nunique()}")
    print(f"  Total Series (Bar x Brand): {panel[['Bar Name', 'Brand Name']].drop_duplicates().shape[0]}")
    print(f"  Zero-Consumption Days Ratio: {(panel['Consumed (ml)'] == 0).mean():.2%}")
    print(f"  Total Volume Consumed: {panel['Consumed (ml)'].sum():,.2f} ml")
