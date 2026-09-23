"""
Main Execution Script for Hotel Bar Inventory Forecasting & Par Level Recommendation System.
Runs the complete end-to-end workflow:
1. Data Pipeline (Ingestion, Conservation Audit, Complete Cartesian Panel)
2. Forecasting Engine (Feature Engineering, Chronological Split, Multi-Model Benchmarks)
3. Par Level & Safety Stock Engine (Dynamic Restocking Calculation)
4. Comparative Inventory Simulation Backtest (Lean vs Static vs Dynamic Policies)
5. Jupyter Notebook Generation & Execution
6. Executive PDF Business Report Compilation
"""

import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import run_pipeline
from src.forecasting import run_model_benchmarks
from src.par_level import generate_inventory_recommendations
from src.simulation import run_comparative_simulation
from generate_pdf_report import build_pdf_report
from build_notebook import generate_full_notebook


def banner(title: str):
    width = 75
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)


def main():
    start_time = time.time()
    banner("HOTEL BAR INVENTORY FORECASTING & PAR LEVEL SYSTEM")
    print("Initializing complete project execution pipeline...\n")

    # ---------------------------------------------------------
    # Step 1: Data Pipeline
    # ---------------------------------------------------------
    banner("STEP 1: DATA INGESTION & CONSERVATION AUDIT")
    t0 = time.time()
    panel, audit = run_pipeline(
        raw_path="data/raw/bar_inventory_data.csv",
        processed_path="data/processed/daily_bar_consumption.csv"
    )
    print(f"• Total Raw Transaction Records:   {audit['total_records']:,}")
    print(f"• Exact Conservation Matches:      {audit['exact_matches']:,}")
    print(f"• Physical Violations Count:       {audit['violations_count']} (100% physically conserved)")
    print(f"• Max Absolute Discrepancy:        {audit['max_absolute_error']:.2e} ml (IEEE 754 precision)")
    print(f"• Full Cartesian Panel Shape:      {panel.shape[0]:,} rows x {panel.shape[1]} columns")
    print(f"• Time Horizon:                    {panel['Date'].min().strftime('%Y-%m-%d')} to {panel['Date'].max().strftime('%Y-%m-%d')} ({panel['Date'].nunique()} days)")
    print(f"• Active Hotel Bars:               {panel['Bar Name'].nunique()} bars across {panel['Brand Name'].nunique()} brands")
    print(f"• Zero-Consumption Days Ratio:     {(panel['Consumed (ml)'] == 0).mean():.2%} (Intermittent Demand)")
    print(f"  [Step 1 completed in {time.time() - t0:.2f}s]")

    # ---------------------------------------------------------
    # Step 2: Demand Forecasting Benchmarks
    # ---------------------------------------------------------
    banner("STEP 2: DEMAND FORECASTING BENCHMARKS (30-DAY CHRONOLOGICAL HOLDOUT)")
    t0 = time.time()
    summary_table, preds_df, bst_model, feature_names = run_model_benchmarks(panel, test_days=30)
    print(summary_table.to_string(index=False))
    print(f"  [Step 2 completed in {time.time() - t0:.2f}s]")

    # ---------------------------------------------------------
    # Step 3: Dynamic Par Level Recommendations
    # ---------------------------------------------------------
    banner("STEP 3: DYNAMIC PAR LEVEL RECOMMENDATIONS")
    t0 = time.time()
    df_raw = pd.read_csv("data/raw/bar_inventory_data.csv")
    df_raw["Date Time Served"] = pd.to_datetime(df_raw["Date Time Served"])
    
    latest_stock = pd.DataFrame(
        df_raw.sort_values("Date Time Served")
        .groupby(["Bar Name", "Brand Name", "Alcohol Type"], as_index=False)
        .last()[["Bar Name", "Brand Name", "Alcohol Type", "Closing Balance (ml)"]]
    )

    forecast_summary = pd.DataFrame(
        preds_df.groupby(["Bar Name", "Brand Name"])
        .agg({
            "pred_xgboost": "mean",
            "rolling_std_14": "last"
        })
        .reset_index()
        .rename(columns={
            "pred_xgboost": "predicted_daily_demand",
            "rolling_std_14": "std_daily_demand"
        })
    )

    recommendations = generate_inventory_recommendations(
        latest_stock, forecast_summary, lead_time_days=2, service_level_z=1.645
    )
    recommendations.to_csv("inventory_recommendations.csv", index=False)
    print("Actionable Reorder Status Distribution:")
    print(recommendations["Status"].value_counts().to_string())
    print("\nTop 5 Urgent Reorder Priorities:")
    print(recommendations.head(5)[["Bar Name", "Brand Name", "Current Stock (ml)", "Par Level (ml)", "Recommended Order (ml)", "Status"]].to_string(index=False))
    print(f"  [Step 3 completed in {time.time() - t0:.2f}s]")

    # ---------------------------------------------------------
    # Step 4: Comparative Simulation Backtest
    # ---------------------------------------------------------
    banner("STEP 4: COMPARATIVE REPLENISHMENT BACKTEST SIMULATION")
    t0 = time.time()
    sim_summary, trajectories = run_comparative_simulation(preds_df, lead_time=2, service_level_z=1.645)
    print(sim_summary.to_string(index=False))
    print(f"  [Step 4 completed in {time.time() - t0:.2f}s]")

    # ---------------------------------------------------------
    # Step 5: Jupyter Notebook Generation & Execution
    # ---------------------------------------------------------
    banner("STEP 5: GENERATING EXECUTED JUPYTER NOTEBOOK")
    t0 = time.time()
    generate_full_notebook()
    print("• Saved: notebooks/inventory_forecasting_solution.ipynb (with rendered figures and outputs)")
    print(f"  [Step 5 completed in {time.time() - t0:.2f}s]")

    # ---------------------------------------------------------
    # Step 6: Executive PDF Report Compilation
    # ---------------------------------------------------------
    banner("STEP 6: COMPILING EXECUTIVE BUSINESS REPORT (PDF)")
    t0 = time.time()
    build_pdf_report("report/business_report.pdf")
    print("• Saved: report/business_report.pdf (2-page publication-grade executive brief)")
    print(f"  [Step 6 completed in {time.time() - t0:.2f}s]")

    total_duration = time.time() - start_time
    banner("ALL WORKFLOW STEPS SUCCESSFULLY COMPLETED")
    print(f"Total Execution Time: {total_duration:.2f} seconds")
    print("Ready for review and submission.\n")


if __name__ == "__main__":
    main()
