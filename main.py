"""
main.py - Entry point for the Shipment ETA & Delay Propagation System

Runs the complete pipeline:
1. Data loading & preprocessing
2. Feature engineering (stage durations + delay propagation)
3. ETA prediction model (XGBoost Regressor)
4. Delay classification model (XGBoost Classifier)
5. SHAP explanations
6. Evaluation plots & metrics report
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_pipeline import run_pipeline
from src.model import (
    train_eta_model,
    train_delay_model,
    generate_shap_explanations,
    generate_evaluation_plots,
)


def save_metrics_report(eta_results: dict, delay_results: dict, output_dir: str = "outputs"):
    """Save a JSON metrics report for reference."""
    report = {
        "eta_prediction": {
            "model": "XGBoost Regressor",
            "metrics": {k: round(v, 4) for k, v in eta_results["metrics"].items()},
            "train_samples": len(eta_results["X_train"]),
            "test_samples": len(eta_results["X_test"]),
            "num_features": len(eta_results["features"]),
        },
        "delay_classification": {
            "model": "XGBoost Classifier",
            "metrics": {k: round(v, 4) for k, v in delay_results["metrics"].items()},
            "train_samples": len(delay_results["X_train"]),
            "test_samples": len(delay_results["X_test"]),
            "num_features": len(delay_results["features"]),
        },
    }

    output_path = Path(output_dir) / "metrics_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Metrics report saved to {output_path}")

    return report


def main():
    """Execute the complete ETA prediction & delay propagation pipeline."""
    start_time = time.time()

    print("\n" + "+" + "=" * 58 + "+")
    print("|  SHIPMENT ETA & DELAY PROPAGATION PREDICTION SYSTEM     |")
    print("|  Aiqon AI Internship - Take Home Assignment              |")
    print("+" + "=" * 58 + "+\n")

    # -- Step 1: Data Pipeline --
    df = run_pipeline(data_dir="data")

    # -- Step 2: Train ETA Model --
    eta_results = train_eta_model(df, output_dir="models")

    # -- Step 3: Train Delay Model --
    delay_results = train_delay_model(df, output_dir="models")

    # -- Step 4: SHAP Explanations --
    print("\n" + "=" * 60)
    print("GENERATING SHAP EXPLANATIONS")
    print("=" * 60)
    eta_shap = generate_shap_explanations(eta_results, "eta_regressor", output_dir="outputs")
    delay_shap = generate_shap_explanations(delay_results, "delay_classifier", output_dir="outputs")

    # -- Step 5: Evaluation Plots --
    generate_evaluation_plots(eta_results, delay_results, df, output_dir="outputs")

    # -- Step 6: Save Metrics --
    report = save_metrics_report(eta_results, delay_results, output_dir="outputs")

    # -- Summary --
    elapsed = time.time() - start_time
    print("\n" + "+" + "=" * 58 + "+")
    print("|  PIPELINE COMPLETE                                       |")
    print("+" + "=" * 58 + "+")
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Shipments processed: {len(df)}")
    print(f"  ETA MAE: {eta_results['metrics']['MAE']:.2f} hours ({eta_results['metrics']['MAE']/24:.2f} days)")
    print(f"  Delay F1: {delay_results['metrics']['F1']:.4f}, AUC: {delay_results['metrics']['AUC']:.4f}")
    print("+" + "=" * 58 + "+")
    print("  Outputs saved to:")
    print("    models/   - Trained XGBoost models (.json)")
    print("    outputs/  - SHAP plots, evaluation charts, metrics")
    print("+" + "=" * 58 + "+\n")


if __name__ == "__main__":
    main()
