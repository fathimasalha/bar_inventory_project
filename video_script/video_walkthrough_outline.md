# 3–5 Minute Video Presentation Walkthrough Outline & Script
## Hotel Bar Inventory Forecasting & Dynamic Par Level Recommendation System

- **Target Duration:** 3:30 to 4:15 minutes
- **Audience:** Executive Reviewers, VP of Food & Beverage, Lead Data Scientist
- **Supporting Materials:** `notebooks/inventory_forecasting_solution.ipynb`, `report/business_report.pdf`

---

## 1. Timing & Scene Overview Table

| Timestamp | Section | Visual on Screen | Core Talking Points |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:45** | **Problem Statement & Business Context** | Title Slide / `report/business_report.pdf` KPI Header Cards | Frame the dual operational dilemma: weekend stockouts of fast-moving brands vs. dead capital and spoilage from slow-moving overstock across 6 hotel properties. |
| **0:45 – 1:45** | **Data Engineering & Demand Modeling** | Notebook: Conservation audit table, ABC Pareto plot, and Day-of-Week seasonality surge chart | 6,575 bottle transaction logs verified; 83.88% zero-demand sparsity addressed via complete Cartesian grid; 2.52x weekend demand surge quantified; benchmarked Naive, Rolling Mean, Holt-Winters, and XGBoost using zero-safe WAPE. |
| **1:45 – 3:00** | **Dynamic Par Level Math & Simulation Engine** | Notebook: Formula markdown, Actionable Recommendations table, and Simulated Inventory Trajectory plot | Explain dynamic par formula: $\text{Par} = (\hat{D}_{\text{daily}} \cdot L) + Z \cdot (\sigma_D \sqrt{L})$ with $L=2, Z=1.645$. Walk through 30-day discrete replenishment backtest demonstrating -66.2% stockout reduction and 94.9% service fulfillment. |
| **3:00 – 4:00** | **Managerial Impact, Scalability & Production Safeguards** | Executive summary comparison table & automated 4-stage daily production architecture | Operational ROI: daily 06:00 AM automated PO generation; proactive risk mitigation for lead-time variance, data drift (rolling WAPE alerts), and weekly physical audits for pour discrepancies. |

---

## 2. Detailed Spoken Script & Screen Action Guide

### Section 1: Problem Statement & Operational Dilemma (0:00 – 0:45)
- **Visual Action:** Display the opening slide and scroll to Section 1 of `report/business_report.pdf` highlighting the four KPI cards (**1.97M ml Consumed, -66.2% Stockout Reduction, 94.9% Service Level, 4.4x Inventory Turnover**).
- **Spoken Voiceover:**
  > *"Hello everyone. In hotel food and beverage operations, managing bar inventory is a delicate balancing act. If you run out of high-demand spirits like Captain Morgan or Bacardi on a bustling Friday night, guest satisfaction drops immediately, premium cocktail service grinds to a halt, and high-margin revenue is lost forever.*
  >
  > *To compensate, bar managers historically over-order on gut feeling. But overstocking slow-moving brands ties up liquid working capital, congests cramped backroom storerooms, and spikes the risk of breakage and shrinkage.*
  >
  > *Today, I’m presenting an end-to-end analytics solution built for a hotel chain operating 6 distinct bars across 16 brands. Our system transforms raw transaction logs into machine-learning demand forecasts, calculates dynamic par levels that automatically adapt to weekend spikes, and proves a 66% reduction in stockouts through historical simulation."*

---

### Section 2: Data Pipeline, EDA & Forecasting Strategy (0:45 – 1:45)
- **Visual Action:** Switch to `notebooks/inventory_forecasting_solution.ipynb`. Highlight Cell 2 (Conservation Check), Cell 4 (ABC Pareto Curve), and Cell 5 (Weekend Spike Chart).
- **Spoken Voiceover:**
  > *"We begin with data integrity. We ingested 6,575 bottle transaction logs spanning 366 days and audited physical inventory conservation: Opening Balance plus Purchases minus Consumed must equal Closing Balance. Across all 6,575 rows, our maximum error was less than 10^-10 ml, verifying complete data consistency.*
  >
  > *Crucially, bars do not pour every spirit daily. Keeping only active sales days blinds models to quiet periods. We constructed a full Cartesian product panel—Date by Bar by Brand—revealing that 83.88% of series-days have zero consumption. This intermittent demand makes standard MAPE invalid due to division-by-zero, so we adopted Weighted Absolute Percentage Error (WAPE) as our primary benchmark metric.*
  >
  > *Our EDA surfaced two critical operational patterns:*
  > *First, an ABC Pareto velocity breakdown showed that the top 5 brands drive over 70% of total volume.*
  > *Second, Friday and Saturday consumption surges to 2.52 times the weekday average.*
  >
  > *To forecast daily demand without data leakage, we executed an 80/20 chronological holdout split and benchmarked 4 models: 7-day Naive, 14-day Rolling Mean, Holt-Winters Exponential Smoothing, and an XGBoost Regressor trained on autoregressive lags, rolling volatilities, and calendar flags. XGBoost achieved the lowest RMSE of 146.49 ml, capturing sharp non-linear demand spikes far better than simple rolling averages."*

---

### Section 3: Dynamic Par Level Formulation & Backtest Simulation (1:45 – 3:00)
- **Visual Action:** Scroll to Cell 8 (Par Level Formulation & Recommendations Table) and Cell 10 (Simulation Trajectory Chart showing dynamic stock curve vs. static par line).
- **Spoken Voiceover:**
  > *"Next, we translate predictions into actionable restocking rules. Par level is the stock needed to cover demand over supplier lead time, plus a safety stock buffer to absorb volatility:*
  >
  > $$\text{Par Level} = (\hat{D}_{\text{daily}} \times L) + Z \cdot (\sigma_{\text{daily}} \sqrt{L})$$
  >
  > *Here, supplier lead time $L$ is 2 days, and for a 95% service level, $Z$ is 1.645. Unlike static hospitality pars that remain flat year-round, our par levels dynamically expand on Thursday mornings ahead of weekend surges and contract on Sunday nights, keeping weekday holding costs exceptionally lean.*
  >
  > *To validate the policy before real-world rollout, we built a discrete-event inventory simulation engine backtesting all 96 bar-brand channels over 30 days. We compared three operational policies:*
  > 1. *A Lean Baseline, which resulted in 432 stockout incidents and an abysmal 34.8% fill rate.*
  > 2. *A Static Average Par, which had 155 stockouts.*
  > 3. *Our Proposed Dynamic ML Par, which reduced stockouts to just 146 incidents—a 66.2% reduction compared to baseline—while hitting our exact target of 94.93% service level fulfillment.*
  >
  > *As shown in the trajectory plot for Captain Morgan at Smith's Bar, the green dynamic policy absorbs high-volume weekend withdrawals smoothly while triggering timely replenishment orders 48 hours in advance."*

---

### Section 4: Business Impact, Scalability & Production Architecture (3:00 – 4:00)
- **Visual Action:** Switch back to `report/business_report.pdf`, focusing on Section 5 (Production Architecture Table and Safeguards).
- **Spoken Voiceover:**
  > *"From an executive standpoint, how do bar managers actually use this?*
  > *At 05:00 AM daily, an automated cron job pulls register closes, updates rolling lag features, runs model inference, and delivers a prioritized purchase order list to managers' dashboards by 06:30 AM.*
  >
  > *Items are categorized into five intuitive operational tiers: Stockout Emergency, Urgent Reorder, Reorder Needed, Adequate Stock, and Overstocked.*
  >
  > *Finally, we architected safeguards for real-world failure modes:*
  > *First, if vendor deliveries are delayed beyond 2 days, our safety stock formula expands to incorporate lead-time variance $\sigma_L$.*
  > *Second, we monitor for data drift using rolling 14-day WAPE, automatically triggering model retraining if error drifts by more than 15%.*
  > *Third, unrecorded over-pouring or spillage is caught and reconciled via weekly physical audits.*
  >
  > *In summary, this system protects guest satisfaction during peak hours while eliminating hundreds of liters of dead inventory capital. Thank you, and I look forward to your questions."*

---

## 3. Reviewer Q&A Prep & Defense Points

1. **Q: Why use WAPE instead of MAPE or RMSE alone?**
   - *A:* Bar demand has 83.88% zero-consumption days. MAPE requires dividing by actual demand ($y_t$), which results in division-by-zero errors or infinite penalties on low-volume days. WAPE ($\frac{\sum |y - \hat{y}|}{\sum y}$) is mathematically stable, volume-weighted, and the recognized retail benchmark for intermittent demand.
2. **Q: How does the system handle discrete bottle purchases if consumption is in ml?**
   - *A:* While simulation and tracking are conducted continuously in milliliters to capture precise cocktail pours, the ERP ordering layer applies a bottle rounding ceiling: $\text{Bottles to Order} = \lceil \frac{\text{Recommended Order (ml)}}{750 \text{ ml}} \rceil$ (or $1,000 \text{ ml}$ for liters).
3. **Q: How does the model prevent lookahead bias?**
   - *A:* All rolling means and standard deviations strictly apply `.shift(1)` so they only observe consumption up to day $t-1$. The evaluation split is purely chronological (final 30 days), never randomized $k$-fold cross validation.
