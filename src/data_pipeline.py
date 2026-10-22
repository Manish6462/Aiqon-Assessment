"""
data_pipeline.py - Data Loading, Cleaning, and Merging Pipeline

Loads the Olist Brazilian E-Commerce dataset and creates a unified
shipment journey table with multi-stage timestamps for delay analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great-circle distance between two points on Earth (in km)."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def load_raw_data(data_dir: str = "data") -> dict:
    """Load all Olist CSV files into a dictionary of DataFrames."""
    data_path = Path(data_dir)
    if not data_path.exists():
        alt_path = Path("Data") if data_dir.lower() == "data" and Path("Data").exists() else Path("data")
        if alt_path.exists():
            data_path = alt_path
    datasets = {}

    file_mapping = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "products": "olist_products_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
    }

    for key, filename in file_mapping.items():
        filepath = data_path / filename
        if filepath.exists():
            datasets[key] = pd.read_csv(filepath)
            print(f"  Loaded {key}: {datasets[key].shape}")
        else:
            print(f"  WARNING: {filename} not found at {filepath}")

    return datasets


def get_geolocation_lookup(geo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a zip-code -> (lat, lng) lookup by averaging coordinates per zip.
    The raw geolocation table has multiple entries per zip code.
    """
    geo_lookup = (
        geo_df.groupby("geolocation_zip_code_prefix")
        .agg({"geolocation_lat": "mean", "geolocation_lng": "mean"})
        .reset_index()
    )
    return geo_lookup


def build_shipment_table(datasets: dict) -> pd.DataFrame:
    """
    Merge orders, items, customers, sellers, and geolocation into a
    single shipment journey table.

    Each row represents a shipment journey with:
    - Multi-stage timestamps
    - Origin (seller) and destination (customer) info
    - Distance, freight, and product details
    """
    print("\n[2/5] Building unified shipment table...")

    orders = datasets["orders"].copy()
    items = datasets["order_items"].copy()
    customers = datasets["customers"].copy()
    sellers = datasets["sellers"].copy()
    geo = datasets["geolocation"].copy()

    # --- Parse timestamps ---
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in timestamp_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    items["shipping_limit_date"] = pd.to_datetime(
        items["shipping_limit_date"], errors="coerce"
    )

    # --- Filter to delivered orders only (we need all timestamps) ---
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered = delivered.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    )
    print(f"  Delivered orders with complete timestamps: {len(delivered)}")

    # --- Aggregate order items per order ---
    # An order can have multiple items from different sellers.
    # We take the first seller (primary) and aggregate price/freight.
    items_agg = (
        items.groupby("order_id")
        .agg(
            seller_id=("seller_id", "first"),
            num_items=("order_item_id", "count"),
            total_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            shipping_limit_date=("shipping_limit_date", "max"),
        )
        .reset_index()
    )

    # --- Merge orders + items ---
    df = delivered.merge(items_agg, on="order_id", how="inner")
    print(f"  After merging with items: {len(df)}")

    # --- Merge customer location ---
    df = df.merge(
        customers[["customer_id", "customer_zip_code_prefix", "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
    )

    # --- Merge seller location ---
    df = df.merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]],
        on="seller_id",
        how="left",
    )

    # --- Add geolocation (lat/lng for distance calculation) ---
    geo_lookup = get_geolocation_lookup(geo)

    # Customer geo
    df = df.merge(
        geo_lookup.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "geolocation_lat": "customer_lat",
                "geolocation_lng": "customer_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )

    # Seller geo
    df = df.merge(
        geo_lookup.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "geolocation_lat": "seller_lat",
                "geolocation_lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )

    # --- Calculate seller-to-customer distance ---
    mask = df[["seller_lat", "seller_lng", "customer_lat", "customer_lng"]].notna().all(axis=1)
    df.loc[mask, "distance_km"] = df.loc[mask].apply(
        lambda r: haversine(r["seller_lng"], r["seller_lat"], r["customer_lng"], r["customer_lat"]),
        axis=1,
    )

    print(f"  Final shipment table: {len(df)} rows, {len(df.columns)} columns")
    return df


def engineer_stage_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute duration (in hours) for each shipment stage.

    Shipment stages (the multi-stage journey):
        Stage 1: Purchase → Approval           (order processing)
        Stage 2: Approval → Carrier Handoff     (seller preparation)
        Stage 3: Carrier Handoff → Delivery     (transit / last-mile)
        Total:   Purchase → Delivery            (end-to-end)

    Also computes:
        - Estimated total delivery time
        - Delay = actual - estimated (positive = late)
    """
    print("\n[3/5] Engineering stage durations & delay features...")

    # --- Stage durations in hours ---
    df["stage1_approval_hours"] = (
        (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    )
    df["stage2_preparation_hours"] = (
        (df["order_delivered_carrier_date"] - df["order_approved_at"]).dt.total_seconds() / 3600
    )
    df["stage3_transit_hours"] = (
        (df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]).dt.total_seconds() / 3600
    )
    df["total_delivery_hours"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    )

    # --- Estimated delivery time ---
    df["estimated_delivery_hours"] = (
        (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    )

    # --- Delay computation ---
    # Positive delay_hours means the shipment arrived LATER than estimated
    df["delay_hours"] = df["total_delivery_hours"] - df["estimated_delivery_hours"]
    df["is_delayed"] = (df["delay_hours"] > 0).astype(int)

    # --- Stage-level delay relative to shipping limit ---
    # Did the seller hand off to carrier before the shipping limit?
    df["carrier_handoff_vs_limit_hours"] = (
        (df["order_delivered_carrier_date"] - df["shipping_limit_date"]).dt.total_seconds() / 3600
    )
    df["seller_missed_limit"] = (df["carrier_handoff_vs_limit_hours"] > 0).astype(int)

    # --- Temporal features ---
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["is_weekend_purchase"] = (df["purchase_dayofweek"] >= 5).astype(int)

    # --- Remove negative or impossible durations (data quality) ---
    duration_cols = ["stage1_approval_hours", "stage2_preparation_hours", "stage3_transit_hours"]
    before = len(df)
    for col in duration_cols:
        df = df[df[col] >= 0]
    after = len(df)
    print(f"  Removed {before - after} rows with negative stage durations")

    # --- Delay statistics ---
    print(f"  Delayed shipments: {df['is_delayed'].sum()} / {len(df)} ({df['is_delayed'].mean()*100:.1f}%)")
    print(f"  Mean delay (delayed only): {df.loc[df['is_delayed']==1, 'delay_hours'].mean():.1f} hours")
    print(f"  Mean total delivery: {df['total_delivery_hours'].mean():.1f} hours")

    return df


def engineer_propagation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that capture DELAY PROPAGATION across stages.

    Key idea: delays at earlier stages compound and affect later stages.
    We model this through interaction features and cumulative delay signals.
    """
    print("\n[4/5] Engineering delay propagation features...")

    # --- Cumulative delay at each stage ---
    # Stage 1 delay: how much longer than median approval time?
    median_approval = df["stage1_approval_hours"].median()
    df["stage1_excess_hours"] = df["stage1_approval_hours"] - median_approval

    # Stage 2 delay: how much longer than median preparation time?
    median_prep = df["stage2_preparation_hours"].median()
    df["stage2_excess_hours"] = df["stage2_preparation_hours"] - median_prep

    # Cumulative delay entering Stage 3 (transit)
    df["cumulative_delay_pre_transit"] = df["stage1_excess_hours"] + df["stage2_excess_hours"]

    # --- Interaction features (delay propagation signals) ---
    # If approval is slow AND preparation is slow => compounding effect
    df["stage1_x_stage2_interaction"] = df["stage1_excess_hours"] * df["stage2_excess_hours"]

    # Ratio: what fraction of total time is spent in transit vs pre-transit?
    df["transit_ratio"] = df["stage3_transit_hours"] / df["total_delivery_hours"].clip(lower=1)
    df["prep_ratio"] = df["stage2_preparation_hours"] / df["total_delivery_hours"].clip(lower=1)

    # --- Route-level statistics (seller performance & route congestion) ---
    # Seller historical average delay
    seller_stats = df.groupby("seller_id").agg(
        seller_avg_delay=("delay_hours", "mean"),
        seller_avg_prep=("stage2_preparation_hours", "mean"),
        seller_order_count=("order_id", "count"),
    ).reset_index()
    df = df.merge(seller_stats, on="seller_id", how="left")

    # State-pair route congestion (seller_state → customer_state)
    df["route"] = df["seller_state"] + "_to_" + df["customer_state"]
    route_stats = df.groupby("route").agg(
        route_avg_delay=("delay_hours", "mean"),
        route_avg_transit=("stage3_transit_hours", "mean"),
        route_volume=("order_id", "count"),
    ).reset_index()
    df = df.merge(route_stats, on="route", how="left")

    # --- Same-state flag (local vs interstate delivery) ---
    df["is_same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)

    # --- Freight per km (proxy for shipment complexity) ---
    df["freight_per_km"] = df["total_freight"] / df["distance_km"].clip(lower=1)

    print(f"  Total features engineered: {len(df.columns)} columns")

    return df


def run_pipeline(data_dir: str = "data") -> pd.DataFrame:
    """Execute the complete data pipeline end-to-end."""
    print("=" * 60)
    print("SHIPMENT ETA & DELAY PROPAGATION - DATA PIPELINE")
    print("=" * 60)

    print("\n[1/5] Loading raw data...")
    datasets = load_raw_data(data_dir)

    df = build_shipment_table(datasets)
    df = engineer_stage_durations(df)
    df = engineer_propagation_features(df)

    print("\n[5/5] Saving processed dataset...")
    output_dir = Path(data_dir)
    if not output_dir.exists() and Path("Data").exists():
        output_dir = Path("Data")
    output_path = output_dir / "processed_shipments.csv"
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE - {len(df)} shipments, {len(df.columns)} features")
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = run_pipeline()
