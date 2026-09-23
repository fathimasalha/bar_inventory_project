"""
Forecasting Module for Hotel Bar Inventory.
Includes feature engineering, chronological train/test splitting,
model training (Naive 7-day, Rolling Mean, Holt-Winters, and XGBoost Regressor),
and evaluation using MAE, RMSE, and WAPE.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Anchor project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RAW_PATH = str(PROJECT_ROOT / "data" / "raw" / "bar_inventory_data.csv")


def calculate_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and WAPE (Weighted Absolute Percentage Error) in pure NumPy.
    WAPE is zero-safe and optimal for intermittent bar demand.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_pred_arr = np.clip(y_pred_arr, 0, None)  # demand cannot be negative

    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
    sum_actual = float(np.sum(y_true_arr))
    wape = float(np.sum(np.abs(y_true_arr - y_pred_arr)) / sum_actual) if sum_actual > 0 else 0.0

    return {
        "MAE (ml)": float(round(mae, 2)),
        "RMSE (ml)": float(round(rmse, 2)),
        "WAPE (%)": float(round(wape * 100.0, 2))
    }


def engineer_features(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates time-series lag and rolling demand features per Bar and Brand,
    strictly shifted to avoid target data leakage.
    """
    df = panel_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Bar Name", "Brand Name", "Date"]).reset_index(drop=True)

    grp = df.groupby(["Bar Name", "Brand Name"])["Consumed (ml)"]

    # Lags
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    # Rolling window averages (strictly shifted by 1 to prevent lookahead)
    df["rolling_mean_7"] = pd.Series(grp.shift(1).rolling(7).mean()).reset_index(drop=True)
    df["rolling_mean_14"] = pd.Series(grp.shift(1).rolling(14).mean()).reset_index(drop=True)
    df["rolling_mean_28"] = pd.Series(grp.shift(1).rolling(28).mean()).reset_index(drop=True)

    # Rolling window standard deviations (volatility indicators for safety stock)
    df["rolling_std_7"] = pd.Series(grp.shift(1).rolling(7).std()).fillna(0.0).reset_index(drop=True)
    df["rolling_std_14"] = pd.Series(grp.shift(1).rolling(14).std()).fillna(0.0).reset_index(drop=True)
    df["rolling_std_28"] = pd.Series(grp.shift(1).rolling(28).std()).fillna(0.0).reset_index(drop=True)

    # Calendar features
    df["dayofweek"] = df["Date"].dt.dayofweek  # type: ignore
    df["is_weekend"] = df["dayofweek"].isin([4, 5, 6]).astype(int)  # Fri, Sat, Sun
    df["month"] = df["Date"].dt.month  # type: ignore
    df["day"] = df["Date"].dt.day  # type: ignore

    return df


def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a chronological train/test split.
    Uses the final test_days as holdout test set.
    """
    max_date = df["Date"].max()
    split_date = max_date - pd.Timedelta(days=test_days - 1)

    clean_df = df.dropna(subset=["lag_28"]).reset_index(drop=True)
    train_df = clean_df[clean_df["Date"] < split_date].reset_index(drop=True)
    test_df = clean_df[clean_df["Date"] >= split_date].reset_index(drop=True)

    return pd.DataFrame(train_df), pd.DataFrame(test_df)


def run_model_benchmarks(
    panel_df: pd.DataFrame,
    test_days: int = 30
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, List[str]]:
    """
    Trains and benchmarks:
    1. 7-Day Naive Seasonal Baseline (lag_7)
    2. 14-Day Rolling Mean Baseline
    3. Holt-Winters Exponential Smoothing
    4. XGBoost ML Regressor
    Returns benchmark table, predictions DataFrame, trained model, and feature list.
    """
    feat_df = engineer_features(panel_df)
    train_df, test_df = split_train_test(feat_df, test_days=test_days)

    y_test = np.asarray(test_df["Consumed (ml)"], dtype=float)
    results = {}
    preds_df = pd.DataFrame(test_df[["Date", "Bar Name", "Brand Name", "Alcohol Type", "Consumed (ml)", "rolling_std_14"]].copy())

    # Model 1: 7-day Naive Seasonal
    preds_df["pred_naive_7"] = test_df["lag_7"].fillna(0.0).to_numpy(dtype=float)
    results["Naive 7-Day Seasonal"] = calculate_metrics(y_test, preds_df["pred_naive_7"])

    # Model 2: 14-day Rolling Mean
    preds_df["pred_rolling_14"] = test_df["rolling_mean_14"].fillna(0.0).to_numpy(dtype=float)
    results["14-Day Rolling Mean"] = calculate_metrics(y_test, preds_df["pred_rolling_14"])

    # Model 3: Holt-Winters Exponential Smoothing per series
    hw_preds = []
    train_dict = {
        tuple(k) if isinstance(k, (tuple, list)) else (k,): np.asarray(v["Consumed (ml)"], dtype=float)
        for k, v in train_df.groupby(["Bar Name", "Brand Name"])
    }
    
    for name, group in test_df.groupby(["Bar Name", "Brand Name"], sort=False):
        key = tuple(name) if isinstance(name, (tuple, list)) else (name,)
        train_series = train_dict.get(key, np.array([], dtype=float))
        if len(train_series) >= 28 and float(np.sum(train_series)) > 50:
            try:
                model = ExponentialSmoothing(
                    train_series,
                    seasonal_periods=7,
                    trend="add",
                    seasonal="add",
                    initialization_method="estimated"
                ).fit(optimized=True)
                forecast = model.forecast(len(group))
                hw_preds.extend(np.clip(forecast, 0, None))
            except Exception:
                hw_preds.extend(np.asarray(group["rolling_mean_7"].fillna(0.0), dtype=float))
        else:
            hw_preds.extend(np.asarray(group["rolling_mean_7"].fillna(0.0), dtype=float))

    preds_df["pred_holt_winters"] = np.array(hw_preds, dtype=float)
    results["Holt-Winters (Additive)"] = calculate_metrics(y_test, preds_df["pred_holt_winters"])

    # Model 4: XGBoost Regressor
    num_features = [
        "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7", "rolling_std_14", "rolling_std_28",
        "dayofweek", "is_weekend", "month", "day"
    ]
    
    # Categorical dummy features cast to float (prevents boolean dtype issues)
    all_combos = pd.concat([train_df, test_df], ignore_index=True)
    dummies = pd.get_dummies(all_combos[["Bar Name", "Brand Name", "Alcohol Type"]], drop_first=True).astype(float)
    
    X_all = pd.concat([all_combos[num_features], dummies], axis=1)
    feature_names = list(X_all.columns)

    X_train = X_all.iloc[:len(train_df)].copy()
    y_train = np.asarray(train_df["Consumed (ml)"], dtype=float)
    X_test = X_all.iloc[len(train_df):].copy()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "seed": 42
    }
    
    bst = xgb.train(params, dtrain, num_boost_round=300)
    xgb_preds = np.clip(bst.predict(dtest), 0, None)
    preds_df["pred_xgboost"] = xgb_preds
    results["XGBoost ML Regressor"] = calculate_metrics(y_test, preds_df["pred_xgboost"])

    summary_table = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    summary_table = summary_table.sort_values("WAPE (%)").reset_index(drop=True)

    return pd.DataFrame(summary_table), pd.DataFrame(preds_df), bst, feature_names


if __name__ == "__main__":
    from data_pipeline import load_raw_data, aggregate_daily_consumption, build_complete_daily_panel

    df = load_raw_data(DEFAULT_RAW_PATH)
    daily_agg = aggregate_daily_consumption(df)
    panel = build_complete_daily_panel(daily_agg)

    summary_table, preds_df, model, fnames = run_model_benchmarks(panel, test_days=30)
    print("\n=== Forecasting Model Benchmark Results (Holdout 30 Days) ===")
    print(summary_table.to_string(index=False))
