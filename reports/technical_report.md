# Predicting Shipment ETA and Delay Propagation Using AI

## Technical Report

**Aiqon AI Internship — Take-Home Assignment**

---

## 1. Introduction

Estimated Time of Arrival (ETA) is one of the most critical signals in logistics planning. However, delays rarely occur in isolation — a late approval at the origin can cascade through seller preparation and transit, amplifying the final delivery delay. This report presents an AI system that (1) predicts final shipment ETA, (2) classifies delay likelihood, and (3) explains which factors contribute most to delays through interpretable SHAP-based explanations.

## 2. Dataset

### 2.1 Data Source
We use the **Brazilian E-Commerce Public Dataset by Olist** (Kaggle), containing ~99,441 real orders from 2016–2018, placed on the Olist marketplace and fulfilled by multiple sellers across Brazil.

**Key tables used:**
- `olist_orders_dataset.csv` — order timestamps across stages
- `olist_order_items_dataset.csv` — product, seller, freight details
- `olist_customers_dataset.csv` / `olist_sellers_dataset.csv` — locations
- `olist_geolocation_dataset.csv` — lat/lng coordinates

### 2.2 Why This Dataset?
This dataset naturally models multi-stage shipment journeys with real timestamps:

| Stage | From → To | Description |
|-------|-----------|-------------|
| Stage 1 | Purchase → Approval | Order processing time |
| Stage 2 | Approval → Carrier Handoff | Seller preparation & packaging |
| Stage 3 | Carrier Handoff → Customer Delivery | Transit / last-mile delivery |

Each order also has a platform-estimated delivery date (`order_estimated_delivery_date`), providing ground truth for evaluating ETA predictions.

### 2.3 Data Cleaning
- Filtered to **delivered orders** with complete timestamps across all stages (96,455 orders)
- Removed 1,373 records with negative stage durations (data quality issues)
- Final dataset: **95,082 shipments** with 52 engineered features

## 3. Feature Engineering

### 3.1 Stage-Level Duration Features
For each shipment, we compute duration (in hours) at each stage:
- `stage1_approval_hours`: Time from purchase to approval
- `stage2_preparation_hours`: Time from approval to carrier handoff
- `stage3_transit_hours`: Time from carrier handoff to customer delivery
- `total_delivery_hours`: End-to-end delivery time (prediction target)

### 3.2 Delay Propagation Features (Key Contribution)
A critical insight is that delays compound across stages. We engineer features that capture this cascade:

1. **Stage excess delay**: Deviation from median duration at each stage (`stage1_excess_hours`, `stage2_excess_hours`)
2. **Cumulative pre-transit delay**: Sum of excess delays before the package enters transit
3. **Stage interaction**: Product of stage 1 and stage 2 excess delays — captures compounding effects where slow approval AND slow preparation together produce disproportionate final delays
4. **Seller limit compliance**: Whether the seller handed off to the carrier before the shipping deadline
5. **Transit ratio**: Fraction of total delivery time spent in transit vs. pre-transit stages

### 3.3 Historical Performance Features
- **Seller average delay**: Historical mean delay for each seller (reliability proxy)
- **Seller order count**: Volume of past orders (experienced vs. new seller)
- **Route congestion**: Average delay and transit time for each seller-state → customer-state route pair

### 3.4 Geographic & Temporal Features
- **Haversine distance**: Great-circle distance between seller and customer locations
- **Same-state flag**: Whether delivery is local or interstate
- **Freight per km**: Proxy for shipment complexity/weight
- **Temporal**: Day of week, hour, month, weekend flags

## 4. Modeling Approach

### 4.1 Task 1: ETA Prediction (Regression)
**Model**: XGBoost Regressor (500 estimators, depth 6, learning rate 0.05)

**Target**: `total_delivery_hours` — actual end-to-end delivery time

**Design choice**: I include Stage 1 and Stage 2 features because in a real-world system, these would be known once the carrier picks up the shipment. The model predicts the remaining delivery time at that point.

### 4.2 Task 2: Delay Classification (Binary)
**Model**: XGBoost Classifier with `scale_pos_weight` for class imbalance

**Target**: `is_delayed` — 1 if actual delivery exceeds estimated delivery date

**Class distribution**: 8.2% delayed (7,792) vs 91.8% on-time (87,290). We use `scale_pos_weight = n_neg/n_pos` to handle the imbalance.

### 4.3 Explainability
**Method**: SHAP (SHapley Additive exPlanations) with TreeExplainer

For each prediction, SHAP decomposes the model output into per-feature contributions, answering: *"Which factors caused this specific shipment to be delayed?"*

## 5. Results

### 5.1 ETA Prediction Performance

| Metric | Value |
|--------|-------|
| **MAE** | 96.37 hours (4.02 days) |
| **RMSE** | 167.41 hours (6.98 days) |
| **R² Score** | 0.4484 |
| **MAPE** | 37.26% |
| **5-Fold CV MAE** | 96.15 ± 0.59 hours |

The low CV standard deviation (0.59) indicates stable, generalizable predictions. The MAE of ~4 days is reasonable given the high variance in Brazilian logistics (mean delivery time is 12.6 days).

### 5.2 Delay Classification Performance

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.8511 |
| **Accuracy** | 84.26% |
| **Recall (Delayed)** | 67.72% |
| **Precision (Delayed)** | 29.76% |
| **F1 Score** | 0.4135 |

The model achieves strong AUC (0.85), successfully catching 68% of delayed shipments. The lower precision reflects the class imbalance — only 8.2% of shipments are delayed, so even a good model will flag some on-time shipments as potentially delayed. In logistics, **high recall is more valuable** than high precision, since missing a delay is costlier than a false alarm.

### 5.3 Key Findings from SHAP Analysis

**Top factors driving ETA predictions** (from SHAP feature importance):
1. `estimated_delivery_hours` — The platform's own estimate is the strongest baseline signal
2. `stage2_preparation_hours` — Seller preparation time is the most variable and impactful stage
3. `route_avg_transit` / `route_avg_delay` — Route-level historical performance captures congestion patterns
4. `cumulative_delay_pre_transit` — Delays that accumulate before transit propagate into final delivery
5. `seller_avg_delay` — Seller reliability is a strong predictor of individual shipment delays

**Delay propagation insight**: The SHAP analysis confirms that delays are not independent across stages. The `stage1_x_stage2_interaction` feature shows that shipments with compounding delays at both approval and preparation stages suffer disproportionately worse final delays.

## 6. Challenges Addressed

1. **Delay propagation across stages**: Modeled through cumulative delay features and stage interaction terms that capture cascading effects
2. **Missing/incomplete data**: Handled via median imputation and filtering of records with impossible negative durations
3. **Route variability**: Captured through state-pair route statistics (average delay, transit time, volume)
4. **Class imbalance**: Addressed with `scale_pos_weight` in XGBoost and emphasis on recall over precision
5. **Congestion × processing interaction**: The `freight_per_km` and route volume features proxy for congestion effects

## 7. Assumptions

1. The first seller in a multi-item order is representative of the primary fulfillment path
2. Historical seller/route performance (computed on the full dataset) approximates what would be available in a production system
3. The Olist platform's estimated delivery date serves as a reasonable baseline to measure delay against
4. Haversine distance approximates actual road/shipping distance

## 8. Reproducibility

```bash
# Clone repository, place Olist data in data/
pip install -r requirements.txt
python main.py
```

All models, plots, and metrics are saved to `models/` and `outputs/` respectively.

## 9. Conclusion

This system demonstrates that shipment delays can be effectively predicted and explained by modeling the multi-stage journey as an interconnected pipeline rather than treating each stage independently. The delay propagation features — particularly cumulative pre-transit delays and stage interaction terms — capture the cascading nature of logistics delays and provide actionable insights for intervention (e.g., prioritizing shipments from historically slow sellers on congested routes).
