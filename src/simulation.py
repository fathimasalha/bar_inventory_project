"""
Inventory Simulation Engine Module.
Simulates daily discrete-event inventory replenishment (Order-Up-To Par Level policy),
tracking lead times, pipeline orders, stockout occurrences, lost volume,
average inventory holding, and inventory turnover.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Union


def simulate_inventory_trajectory(
    demand_series: Union[np.ndarray, List[float], pd.Series],
    par_levels: Union[float, np.ndarray, List[float], pd.Series],
    lead_time: int = 2,
    initial_stock: float = None,
    reorder_point: Union[float, np.ndarray] = None
) -> Dict[str, Any]:
    """
    Executes a discrete daily inventory simulation using an Order-Up-To Par Level policy.
    
    Parameters:
        demand_series: Array of actual daily consumption (ml)
        par_levels: Float or array of par levels for each day (ml)
        lead_time: Delivery lead time in days (L)
        initial_stock: Starting inventory level (defaults to par_levels[0] or par_levels)
        reorder_point: Threshold triggering an order (defaults to par_level)
    
    Returns:
        Dictionary of KPIs and daily trajectory history.
    """
    demands = np.asarray(demand_series, dtype=float)
    n_days = len(demands)

    if np.isscalar(par_levels):
        pars = np.full(n_days, float(par_levels))
    else:
        pars = np.asarray(par_levels, dtype=float)

    if initial_stock is None:
        stock = pars[0] if len(pars) > 0 else 1000.0
    else:
        stock = float(initial_stock)

    stockouts = 0
    lost_volume = 0.0
    fulfilled_volume = 0.0
    total_orders_placed = 0
    total_volume_ordered = 0.0

    # Pending orders: list of [days_remaining, order_quantity]
    pending_orders: List[List[float]] = []

    stock_history = []
    order_history = []
    stockout_history = []

    for day in range(n_days):
        day_demand = demands[day]
        current_par = pars[day]
        day_rop = current_par if reorder_point is None else (reorder_point if np.isscalar(reorder_point) else reorder_point[day])

        # 1. Receive incoming shipments
        for order in pending_orders:
            order[0] -= 1
            if order[0] <= 0:
                stock += order[1]
        pending_orders = [o for o in pending_orders if o[0] > 0]

        # 2. Satisfy customer demand
        if stock >= day_demand:
            stock -= day_demand
            fulfilled_volume += day_demand
            stockout_today = 0
        else:
            unmet = day_demand - stock
            lost_volume += unmet
            fulfilled_volume += stock
            stockout_today = 1
            stockouts += 1
            stock = 0.0

        # 3. Replenishment check (Order-Up-To Par Level)
        pipeline_inventory = sum(o[1] for o in pending_orders)
        effective_inventory = stock + pipeline_inventory

        if effective_inventory < day_rop:
            order_qty = max(0.0, current_par - effective_inventory)
            if order_qty > 0:
                pending_orders.append([lead_time, order_qty])
                total_orders_placed += 1
                total_volume_ordered += order_qty
                order_today = order_qty
            else:
                order_today = 0.0
        else:
            order_today = 0.0

        stock_history.append(stock)
        order_history.append(order_today)
        stockout_history.append(stockout_today)

    total_demand = np.sum(demands)
    avg_inventory = float(np.mean(stock_history)) if len(stock_history) > 0 else 0.0
    turnover = (total_demand / avg_inventory) if avg_inventory > 0 else 0.0
    fill_rate = (fulfilled_volume / total_demand * 100.0) if total_demand > 0 else 100.0
    service_level = ((n_days - stockouts) / n_days * 100.0) if n_days > 0 else 100.0

    return {
        "total_days": n_days,
        "total_demand_ml": float(total_demand),
        "fulfilled_volume_ml": float(fulfilled_volume),
        "lost_volume_ml": float(lost_volume),
        "stockout_days": int(stockouts),
        "service_level_pct": float(service_level),
        "volume_fill_rate_pct": float(fill_rate),
        "average_inventory_ml": float(avg_inventory),
        "turnover_ratio": float(turnover),
        "orders_placed_count": int(total_orders_placed),
        "total_volume_ordered_ml": float(total_volume_ordered),
        "history_stock": stock_history,
        "history_orders": order_history,
        "history_stockouts": stockout_history
    }


def run_comparative_simulation(
    test_panel_df: pd.DataFrame,
    lead_time: int = 2,
    service_level_z: float = 1.645
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], Any]]:
    """
    Backtests 3 distinct operational replenishment policies across all 96 bar-brand series:
    1. Static Fixed Par (traditional static rule)
    2. Heuristic Low Par (lean/undercapitalized)
    3. Dynamic ML Par Level (proposed dynamic rule)
    """
    policy_metrics = {
        "Lean Baseline (1d Buffer)": {"stockouts": 0, "lost_vol": 0.0, "avg_inv": 0.0, "tot_dem": 0.0, "fulfilled": 0.0},
        "Static Average Par": {"stockouts": 0, "lost_vol": 0.0, "avg_inv": 0.0, "tot_dem": 0.0, "fulfilled": 0.0},
        "Dynamic ML Par Level (Proposed)": {"stockouts": 0, "lost_vol": 0.0, "avg_inv": 0.0, "tot_dem": 0.0, "fulfilled": 0.0}
    }

    trajectories = {}

    for (bar, brand), group in test_panel_df.groupby(["Bar Name", "Brand Name"]):
        actuals = group["Consumed (ml)"].values
        n_days = len(actuals)
        if n_days == 0:
            continue

        mean_demand = np.mean(actuals)
        std_demand = np.std(actuals) if np.std(actuals) > 0 else 10.0

        # Policy 1: Lean baseline (covers only lead time with no safety buffer)
        par_lean = max(100.0, mean_demand * lead_time)
        res_lean = simulate_inventory_trajectory(actuals, par_lean, lead_time=lead_time)

        # Policy 2: Static Average Par (fixed static buffer using 1.645 * std * sqrt(L))
        static_safety = service_level_z * std_demand * np.sqrt(lead_time)
        par_static = max(200.0, (mean_demand * lead_time) + static_safety)
        res_static = simulate_inventory_trajectory(actuals, par_static, lead_time=lead_time)

        # Policy 3: Dynamic ML Par (dynamic daily forecast + dynamic rolling std)
        daily_forecast = group.get("pred_lgbm", pd.Series(mean_demand, index=group.index)).values
        rolling_std = group.get("rolling_std_14", pd.Series(std_demand, index=group.index)).values
        rolling_std = np.where(rolling_std <= 0, std_demand, rolling_std)

        par_dynamic = (daily_forecast * lead_time) + (service_level_z * rolling_std * np.sqrt(lead_time))
        par_dynamic = np.maximum(par_dynamic, 150.0)  # sensible minimum threshold
        res_dynamic = simulate_inventory_trajectory(actuals, par_dynamic, lead_time=lead_time)

        # Aggregate metrics
        for name, res in [("Lean Baseline (1d Buffer)", res_lean),
                          ("Static Average Par", res_static),
                          ("Dynamic ML Par Level (Proposed)", res_dynamic)]:
            policy_metrics[name]["stockouts"] += res["stockout_days"]
            policy_metrics[name]["lost_vol"] += res["lost_volume_ml"]
            policy_metrics[name]["avg_inv"] += res["average_inventory_ml"]
            policy_metrics[name]["tot_dem"] += res["total_demand_ml"]
            policy_metrics[name]["fulfilled"] += res["fulfilled_volume_ml"]

        trajectories[(bar, brand)] = {
            "lean": res_lean,
            "static": res_static,
            "dynamic": res_dynamic,
            "actuals": actuals
        }

    # Summary table
    n_series = test_panel_df[["Bar Name", "Brand Name"]].drop_duplicates().shape[0]
    total_item_days = len(test_panel_df)

    summary_rows = []
    for policy, m in policy_metrics.items():
        tot_dem = m["tot_dem"]
        fill_rate = (m["fulfilled"] / tot_dem * 100.0) if tot_dem > 0 else 100.0
        service_lvl = ((total_item_days - m["stockouts"]) / total_item_days * 100.0) if total_item_days > 0 else 100.0
        avg_inv_total = m["avg_inv"]
        turnover = (tot_dem / avg_inv_total) if avg_inv_total > 0 else 0.0

        summary_rows.append({
            "Policy": policy,
            "Total Stockout Incidents": m["stockouts"],
            "Lost Demand (ml)": round(m["lost_vol"], 1),
            "Fulfillment Fill Rate (%)": round(fill_rate, 2),
            "Service Level (%)": round(service_lvl, 2),
            "Avg Total Holding Stock (ml)": round(avg_inv_total, 1),
            "Inventory Turnover Ratio": round(turnover, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, trajectories
