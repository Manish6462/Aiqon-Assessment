"""
demo.py - Interactive CLI Demonstration of Shipment ETA & Delay Propagation System

Demonstrates:
1. Real-time ETA prediction & Delay Risk scoring on sample shipments
2. Root-cause explainability (top factors driving the delay)
3. Interactive "What-If" Delay Propagation Simulation (test how upstream delays cascade)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import shap

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import ETA_FEATURES, DELAY_FEATURES


def load_artifacts():
    """Load trained models and processed data."""
    print("Loading trained models and processed dataset...")
    
    # Load models
    eta_model = xgb.XGBRegressor()
    eta_model.load_model("models/eta_model.json")
    
    delay_model = xgb.XGBClassifier()
    delay_model.load_model("models/delay_model.json")
    
    # Load dataset
    data_file = Path("Data/processed_shipments.csv")
    if not data_file.exists():
        data_file = Path("data/processed_shipments.csv")
        
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} shipments.\n")
    return eta_model, delay_model, df


def explain_prediction(model, row_df, top_k=4):
    """Compute local SHAP contributions for a single shipment."""
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(row_df)
    
    # Handle single row
    vals = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
    feature_impacts = sorted(
        zip(row_df.columns, vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return feature_impacts[:top_k]


def display_shipment_card(idx, row, eta_model, delay_model):
    """Display detailed prediction card for a shipment."""
    feature_row = pd.DataFrame([row[ETA_FEATURES]])
    
    # Fill NAs
    for col in feature_row.columns:
        if feature_row[col].dtype in ["float64", "float32", "int64", "int32"]:
            feature_row[col] = feature_row[col].fillna(0)
            
    # Predictions
    pred_eta_hours = float(eta_model.predict(feature_row)[0])
    pred_delay_prob = float(delay_model.predict_proba(feature_row)[0][1])
    is_delayed_pred = pred_delay_prob >= 0.5
    
    actual_hours = row.get("total_delivery_hours", np.nan)
    actual_delayed = bool(row.get("is_delayed", False))
    route = f"{row.get('seller_state', 'N/A')} -> {row.get('customer_state', 'N/A')}"
    
    print("\n" + "=" * 65)
    print(f"  SHIPMENT INSPECTION: Order #{row.get('order_id', idx)[:16]}...")
    print("=" * 65)
    print(f"  Route:               {route} ({row.get('distance_km', 0):.0f} km)")
    print(f"  Stage 1 (Approval):  {row.get('stage1_approval_hours', 0):.1f} hours")
    print(f"  Stage 2 (Seller):    {row.get('stage2_preparation_hours', 0):.1f} hours")
    print(f"  Pre-Transit Delay:   {row.get('cumulative_delay_pre_transit', 0):+.1f} hours")
    print("-" * 65)
    print(f"  Actual Delivery:     {actual_hours:.1f} hrs ({actual_hours/24:.1f} days)")
    print(f"  Predicted ETA:       {pred_eta_hours:.1f} hrs ({pred_eta_hours/24:.1f} days)")
    print(f"  ETA Prediction Error: {abs(pred_eta_hours - actual_hours):.1f} hrs ({abs(pred_eta_hours - actual_hours)/24:.1f} days)")
    print("-" * 65)
    delay_tag = "[DELAYED]" if actual_delayed else "[ON-TIME]"
    pred_tag = "HIGH RISK" if is_delayed_pred else "LOW RISK"
    print(f"  Actual Status:       {delay_tag}")
    print(f"  Delay Probability:   {pred_delay_prob*100:.1f}% -> {pred_tag}")
    
    # Explainability
    print("-" * 65)
    print("  ROOT CAUSE EXPLANATION (Top Contributing Factors):")
    eta_factors = explain_prediction(eta_model, feature_row, top_k=3)
    for feat, imp in eta_factors:
        direction = "+" if imp > 0 else "-"
        print(f"   * {feat:<32}: {direction}{abs(imp):.1f} hours to ETA")
        
    delay_factors = explain_prediction(delay_model, feature_row, top_k=3)
    print("  Delay Risk Factors (SHAP Log-Odds impact):")
    for feat, imp in delay_factors:
        sign = "Increases delay risk" if imp > 0 else "Reduces delay risk"
        print(f"   * {feat:<32}: {sign} ({imp:+.2f})")
    print("=" * 65 + "\n")


def run_what_if_simulation(eta_model, delay_model, df):
    """Simulate how an upstream delay in Stage 2 propagates to final ETA and delay risk."""
    print("\n" + "=" * 65)
    print("  WHAT-IF SCENARIO: DELAY PROPAGATION SIMULATION")
    print("=" * 65)
    print("  We simulate a seller facing operational bottlenecks, increasing")
    print("  Stage 2 (preparation & packing) from 12 hours up to 120 hours.")
    print("=" * 65)
    
    # Pick a standard baseline shipment
    sample = df[(df["is_same_state"] == 0) & (df["is_delayed"] == 0)].iloc[10].copy()
    base_row = pd.DataFrame([sample[ETA_FEATURES]])
    for col in base_row.columns:
        base_row[col] = base_row[col].fillna(0)
        
    print(f"  Route: {sample['seller_state']} -> {sample['customer_state']} ({sample['distance_km']:.0f} km)\n")
    print(f"  {'Seller Prep':<14} | {'Pre-Transit':<14} | {'Predicted ETA':<16} | {'Delay Risk':<12} | {'Risk Status'}")
    print("  " + "-" * 72)
    
    for prep_hours in [12, 24, 48, 72, 96, 120]:
        sim_row = base_row.copy()
        sim_row["stage2_preparation_hours"] = prep_hours
        median_prep = 33.9
        excess = prep_hours - median_prep
        sim_row["stage2_excess_hours"] = excess
        sim_row["cumulative_delay_pre_transit"] = sim_row["stage1_excess_hours"].values[0] + excess
        sim_row["stage1_x_stage2_interaction"] = sim_row["stage1_excess_hours"].values[0] * excess
        
        eta = float(eta_model.predict(sim_row)[0])
        prob = float(delay_model.predict_proba(sim_row)[0][1])
        status = "CRITICAL" if prob > 0.6 else ("ELEVATED" if prob > 0.3 else "NORMAL")
        
        print(f"  {prep_hours:>4} hrs ({prep_hours/24:.1f}d) | {excess:>+5.1f} hrs excess | {eta:>5.1f} hrs ({eta/24:>4.1f}d) | {prob*100:>6.1f}% risk  | [{status}]")
    print("  " + "-" * 72)
    print("  Notice: Upstream warehouse delays directly propagate downstream,")
    print("  amplifying total delivery time and triggering high delay alerts!\n")


def main():
    eta_model, delay_model, df = load_artifacts()
    
    # Find sample on-time and delayed shipments
    delayed_samples = df[df["is_delayed"] == 1]
    ontime_samples = df[df["is_delayed"] == 0]
    
    print("\n" + "#" * 65)
    print("  AIQON ASSESSMENT -- SHIPMENT ETA & DELAY SYSTEM DEMO")
    print("#" * 65)
    
    # 1. On-Time Sample Inspection
    print("\n>>> DEMO 1: Inspecting an On-Time Shipment")
    sample_ontime = ontime_samples.iloc[42]
    display_shipment_card(42, sample_ontime, eta_model, delay_model)
    
    # 2. Delayed Sample Inspection
    print("\n>>> DEMO 2: Inspecting a Delayed Shipment (Cascade Bottleneck)")
    sample_delayed = delayed_samples.iloc[15]
    display_shipment_card(15, sample_delayed, eta_model, delay_model)
    
    # 3. What-If Delay Propagation
    print("\n>>> DEMO 3: Delay Propagation Sensitivity Analysis")
    run_what_if_simulation(eta_model, delay_model, df)
    
    print("#" * 65)
    print("  Demo completed successfully.")
    print("  To run full model retraining & evaluation: python main.py")
    print("  To regenerate PDF report:                  python generate_pdf_report.py")
    print("#" * 65 + "\n")


if __name__ == "__main__":
    main()
