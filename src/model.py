"""
model.py - ETA Prediction & Delay Classification Models

Implements:
1. XGBoost Regressor for ETA prediction (total delivery hours)
2. XGBoost Classifier for delay prediction (is_delayed binary)
3. SHAP-based explainability for both models
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Feature selection
# ──────────────────────────────────────────────────────────────

# Features for ETA regression (predicting total_delivery_hours)
# We exclude stage3_transit_hours and total_delivery_hours since those
# are what we're predicting (information leakage).
# We DO include stage1 and stage2 features because in a real system
# you'd know these once the carrier picks up the shipment.
ETA_FEATURES = [
    # Stage durations (known at carrier handoff)
    "stage1_approval_hours",
    "stage2_preparation_hours",
    # Delay propagation features
    "stage1_excess_hours",
    "stage2_excess_hours",
    "cumulative_delay_pre_transit",
    "stage1_x_stage2_interaction",
    "seller_missed_limit",
    "carrier_handoff_vs_limit_hours",
    # Route & logistics
    "distance_km",
    "total_price",
    "total_freight",
    "num_items",
    "freight_per_km",
    "is_same_state",
    # Temporal
    "purchase_dayofweek",
    "purchase_hour",
    "purchase_month",
    "is_weekend_purchase",
    # Historical performance
    "seller_avg_delay",
    "seller_avg_prep",
    "seller_order_count",
    "route_avg_delay",
    "route_avg_transit",
    "route_volume",
    # Estimated delivery (platform estimate)
    "estimated_delivery_hours",
]

# Features for delay classification
# Same features - we predict if shipment will be delayed
DELAY_FEATURES = ETA_FEATURES.copy()

ETA_TARGET = "total_delivery_hours"
DELAY_TARGET = "is_delayed"


def prepare_features(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Prepare features and target, handling missing values."""
    # Select features that exist in the dataframe
    available_features = [f for f in feature_cols if f in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"  Warning: Missing features: {missing}")

    X = df[available_features].copy()
    y = df[target_col].copy()

    # Fill NaN with median for numeric columns
    for col in X.columns:
        if X[col].dtype in ["float64", "float32", "int64", "int32"]:
            X[col] = X[col].fillna(X[col].median())

    # Drop remaining NaN rows
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    return X, y, available_features


# ──────────────────────────────────────────────────────────────
# ETA Regression Model
# ──────────────────────────────────────────────────────────────

def train_eta_model(df: pd.DataFrame, output_dir: str = "models"):
    """Train XGBoost regressor to predict total delivery time in hours."""
    print("\n" + "=" * 60)
    print("TRAINING ETA PREDICTION MODEL (XGBoost Regressor)")
    print("=" * 60)

    X, y, features = prepare_features(df, ETA_FEATURES, ETA_TARGET)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=1))) * 100

    print(f"\n  ETA Prediction Results:")
    print(f"  {'MAE':>20}: {mae:.2f} hours ({mae/24:.2f} days)")
    print(f"  {'RMSE':>20}: {rmse:.2f} hours ({rmse/24:.2f} days)")
    print(f"  {'R² Score':>20}: {r2:.4f}")
    print(f"  {'MAPE':>20}: {mape:.2f}%")

    # Cross-validation
    cv_scores = cross_val_score(
        xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42),
        X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    print(f"  {'5-Fold CV MAE':>20}: {-cv_scores.mean():.2f} ± {cv_scores.std():.2f} hours")

    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    model.save_model(str(output_path / "eta_model.json"))
    print(f"\n  Model saved to {output_path / 'eta_model.json'}")

    results = {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "features": features,
        "metrics": {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape,
                     "CV_MAE_mean": -cv_scores.mean(), "CV_MAE_std": cv_scores.std()},
    }

    return results


# ──────────────────────────────────────────────────────────────
# Delay Classification Model
# ──────────────────────────────────────────────────────────────

def train_delay_model(df: pd.DataFrame, output_dir: str = "models"):
    """Train XGBoost classifier to predict if a shipment will be delayed."""
    print("\n" + "=" * 60)
    print("TRAINING DELAY CLASSIFICATION MODEL (XGBoost Classifier)")
    print("=" * 60)

    X, y, features = prepare_features(df, DELAY_FEATURES, DELAY_TARGET)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Class balance: {y.value_counts().to_dict()}")

    # Handle class imbalance with scale_pos_weight
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="logloss",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n  Delay Classification Results:")
    print(f"  {'Accuracy':>20}: {acc:.4f}")
    print(f"  {'Precision':>20}: {prec:.4f}")
    print(f"  {'Recall':>20}: {rec:.4f}")
    print(f"  {'F1 Score':>20}: {f1:.4f}")
    print(f"  {'AUC-ROC':>20}: {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))

    # Save model
    output_path = Path(output_dir)
    model.save_model(str(output_path / "delay_model.json"))
    print(f"  Model saved to {output_path / 'delay_model.json'}")

    results = {
        "model": model,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_prob": y_prob,
        "features": features,
        "metrics": {"Accuracy": acc, "Precision": prec, "Recall": rec,
                     "F1": f1, "AUC": auc},
    }

    return results


# ──────────────────────────────────────────────────────────────
# SHAP Explainability
# ──────────────────────────────────────────────────────────────

def generate_shap_explanations(results: dict, model_name: str, output_dir: str = "outputs"):
    """Generate SHAP plots explaining model predictions."""
    print(f"\n  Generating SHAP explanations for {model_name}...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    model = results["model"]
    X_test = results["X_test"]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- 1. Summary plot (global feature importance) ---
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title(f"SHAP Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_summary_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. Bar plot (mean absolute SHAP) ---
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.title(f"Mean |SHAP| - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_bar_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3. Waterfall plot for a specific delayed prediction ---
    # Find a sample that was predicted as delayed/high ETA
    if model_name == "delay_classifier":
        delayed_idx = np.where(results["y_pred"] == 1)[0]
        if len(delayed_idx) > 0:
            sample_idx = delayed_idx[0]
        else:
            sample_idx = 0
    else:
        # For ETA, pick a high-ETA prediction
        sample_idx = np.argmax(results["y_pred"])

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False, max_display=12)
    plt.title(f"SHAP Waterfall (Sample Prediction) - {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_waterfall_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 4. Dependence plot for top feature ---
    top_feature = results["features"][
        np.argmax(np.abs(shap_values.values).mean(axis=0))
    ]
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(top_feature, shap_values.values, X_test, show=False)
    plt.title(f"SHAP Dependence: {top_feature} - {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / f"shap_dependence_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  SHAP plots saved to {output_path}/")

    return shap_values


# ──────────────────────────────────────────────────────────────
# Visualization & Evaluation Plots
# ──────────────────────────────────────────────────────────────

def generate_evaluation_plots(eta_results: dict, delay_results: dict, df: pd.DataFrame,
                              output_dir: str = "outputs"):
    """Generate comprehensive evaluation and analysis plots."""
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION PLOTS")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid", palette="husl")

    # --- 1. ETA: Actual vs Predicted scatter ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    y_test = eta_results["y_test"]
    y_pred = eta_results["y_pred"]

    ax.scatter(y_test / 24, y_pred / 24, alpha=0.1, s=5, color="#3498db")
    max_val = max(y_test.max(), y_pred.max()) / 24
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Delivery Time (days)", fontsize=12)
    ax.set_ylabel("Predicted Delivery Time (days)", fontsize=12)
    ax.set_title("ETA Prediction: Actual vs Predicted", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, min(max_val, 60))
    ax.set_ylim(0, min(max_val, 60))
    plt.tight_layout()
    plt.savefig(output_path / "eta_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 2. ETA Error Distribution ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    errors = (y_pred - y_test.values) / 24  # in days
    ax.hist(errors, bins=80, color="#2ecc71", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Prediction Error (days)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("ETA Prediction Error Distribution", fontsize=14, fontweight="bold")
    ax.set_xlim(-15, 15)
    stats_text = f"Mean Error: {errors.mean():.2f} days\nMedian Error: {np.median(errors):.2f} days\nStd: {errors.std():.2f} days"
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_path / "eta_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 3. Delay: Confusion Matrix ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cm = confusion_matrix(delay_results["y_test"], delay_results["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["On-Time", "Delayed"],
                yticklabels=["On-Time", "Delayed"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Delay Classification: Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "delay_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 4. Stage Duration Distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    stages = [
        ("stage1_approval_hours", "Stage 1: Approval", "#3498db"),
        ("stage2_preparation_hours", "Stage 2: Preparation", "#e74c3c"),
        ("stage3_transit_hours", "Stage 3: Transit", "#2ecc71"),
    ]
    for ax, (col, title, color) in zip(axes, stages):
        data = df[col].clip(upper=df[col].quantile(0.95))
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(data.median(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Median: {data.median():.1f}h")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Hours", fontsize=10)
        ax.legend()
    fig.suptitle("Distribution of Stage Durations", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "stage_duration_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 5. Delay Propagation: Stage delays vs final delay ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    stage_cols = ["stage1_approval_hours", "stage2_preparation_hours", "stage3_transit_hours"]
    stage_names = ["Stage 1: Approval", "Stage 2: Preparation", "Stage 3: Transit"]
    for ax, col, name in zip(axes, stage_cols, stage_names):
        sample = df.sample(min(5000, len(df)), random_state=42)
        colors = ["#e74c3c" if d == 1 else "#2ecc71" for d in sample["is_delayed"]]
        ax.scatter(sample[col], sample["delay_hours"], alpha=0.2, s=5, c=colors)
        ax.set_xlabel(f"{name} (hours)", fontsize=10)
        ax.set_ylabel("Final Delay (hours)", fontsize=10)
        ax.set_title(f"Delay Propagation: {name}", fontsize=11, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0, sample[col].quantile(0.95))
        ax.set_ylim(sample["delay_hours"].quantile(0.01), sample["delay_hours"].quantile(0.99))
    fig.suptitle("How Stage Delays Propagate to Final Delay", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "delay_propagation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 6. Delay rate by route volume ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    route_data = df.groupby("route").agg(
        delay_rate=("is_delayed", "mean"),
        volume=("order_id", "count"),
    ).reset_index()
    route_data = route_data[route_data["volume"] >= 20]  # filter low-volume routes
    scatter = ax.scatter(
        route_data["volume"], route_data["delay_rate"] * 100,
        s=route_data["volume"] / 2, alpha=0.6, c=route_data["delay_rate"],
        cmap="RdYlGn_r", edgecolors="gray", linewidth=0.5,
    )
    ax.set_xlabel("Route Volume (# shipments)", fontsize=12)
    ax.set_ylabel("Delay Rate (%)", fontsize=12)
    ax.set_title("Delay Rate vs Route Volume", fontsize=14, fontweight="bold")
    plt.colorbar(scatter, label="Delay Rate")
    plt.tight_layout()
    plt.savefig(output_path / "delay_rate_by_route.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 7. XGBoost Feature Importance (built-in) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (results, title) in zip(axes, [
        (eta_results, "ETA Regressor"),
        (delay_results, "Delay Classifier"),
    ]):
        importances = results["model"].feature_importances_
        feature_imp = pd.DataFrame({
            "feature": results["features"],
            "importance": importances,
        }).sort_values("importance", ascending=True).tail(15)

        ax.barh(feature_imp["feature"], feature_imp["importance"], color="#3498db")
        ax.set_title(f"Feature Importance: {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance (Gain)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / "feature_importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  All plots saved to {output_path}/")


if __name__ == "__main__":
    # This module is imported and called from main.py
    print("Import this module and call train_eta_model() / train_delay_model()")
