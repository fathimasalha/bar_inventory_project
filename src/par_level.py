"""
Par Level and Safety Stock Formulation Module.
Computes dynamic inventory par levels, safety stocks, and operational reorder flags
based on demand forecasts, demand volatility, supplier lead time, and target service levels.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_par_level(
    predicted_daily_demand: float,
    std_daily_demand: float,
    lead_time_days: int = 2,
    service_level_z: float = 1.645
) -> Tuple[float, float, float]:
    """
    Computes dynamic Par Level and Safety Stock.
    
    Formula:
        Lead Time Demand = Predicted Daily Demand * Lead Time
        Safety Stock = Z * (Std Daily Demand * sqrt(Lead Time))
        Par Level = Lead Time Demand + Safety Stock

    Parameters:
        predicted_daily_demand: Expected daily consumption (ml)
        std_daily_demand: Standard deviation / volatility of daily demand (ml)
        lead_time_days: Supplier delivery lead time in days (default: 2)
        service_level_z: Z-score factor (1.645 for 95%, 2.326 for 99%)
    
    Returns:
        (par_level, safety_stock, lead_time_demand)
    """
    lead_time_demand = max(0.0, float(predicted_daily_demand) * lead_time_days)
    safety_stock = max(0.0, float(service_level_z) * (float(std_daily_demand) * np.sqrt(lead_time_days)))
    par_level = lead_time_demand + safety_stock
    return par_level, safety_stock, lead_time_demand


def generate_inventory_recommendations(
    current_inventory_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lead_time_days: int = 2,
    service_level_z: float = 1.645
) -> pd.DataFrame:
    """
    Produces actionable bar-level inventory recommendations comparing current closing balances
    against dynamic par levels and safety stocks.
    """
    merged = current_inventory_df.merge(
        forecast_df,
        on=["Bar Name", "Brand Name"],
        how="inner"
    )

    recs = []
    for _, row in merged.iterrows():
        daily_demand = row["predicted_daily_demand"]
        demand_std = row["std_daily_demand"]
        current_stock = row["Closing Balance (ml)"]

        par_level, safety_stock, lt_demand = compute_par_level(
            daily_demand, demand_std, lead_time_days, service_level_z
        )

        if current_stock <= 1e-4:
            status = "Stockout Emergency"
            reorder_qty = max(0.0, par_level - current_stock)
        elif current_stock < safety_stock:
            status = "Urgent Reorder"
            reorder_qty = max(0.0, par_level - current_stock)
        elif current_stock < par_level:
            status = "Reorder Needed"
            reorder_qty = max(0.0, par_level - current_stock)
        elif current_stock > 1.5 * par_level and par_level > 0:
            status = "Overstocked"
            reorder_qty = 0.0
        else:
            status = "Adequate Stock"
            reorder_qty = 0.0

        recs.append({
            "Bar Name": row["Bar Name"],
            "Brand Name": row["Brand Name"],
            "Alcohol Type": row.get("Alcohol Type", "Unknown"),
            "Current Stock (ml)": round(current_stock, 2),
            "Predicted Daily Demand (ml)": round(daily_demand, 2),
            "Demand Std Dev (ml)": round(demand_std, 2),
            "Safety Stock (ml)": round(safety_stock, 2),
            "Par Level (ml)": round(par_level, 2),
            "Recommended Order (ml)": round(reorder_qty, 2),
            "Status": status,
            "Overstock Flag": status == "Overstocked"
        })

    rec_df = pd.DataFrame(recs)
    priority_map = {
        "Stockout Emergency": 0,
        "Urgent Reorder": 1,
        "Reorder Needed": 2,
        "Adequate Stock": 3,
        "Overstocked": 4
    }
    rec_df["priority"] = rec_df["Status"].map(priority_map)
    rec_df = rec_df.sort_values(["priority", "Recommended Order (ml)"], ascending=[True, False]).drop(columns=["priority"])
    return rec_df


if __name__ == "__main__":
    par, ss, ltd = compute_par_level(predicted_daily_demand=250.0, std_daily_demand=80.0, lead_time_days=2, service_level_z=1.645)
    print("=== Par Level Unit Test ===")
    print(f"  Lead Time Demand (2d): {ltd:.2f} ml")
    print(f"  Safety Stock (95%):    {ss:.2f} ml")
    print(f"  Par Level:             {par:.2f} ml")
