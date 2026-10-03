"""Cancellations, non-product codes, outliers, date alignment."""
"""Cleaning and weekly aggregation for UCI Online Retail II."""
import pandas as pd


def clean_transactions(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df = df.copy()

    stock_code = df["StockCode"].fillna("").astype("string").str.strip().str.upper()
    description = df["Description"].fillna("").astype("string").str.strip().str.upper()
    invoice_no = df["InvoiceNo"].fillna("").astype("string").str.upper()

    df = df.assign(
        StockCode=stock_code,
        Description=description,
        InvoiceNo=invoice_no,
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # 1. Cancellations (invoice starts with C): drop them (or net, per params)
    is_cancel = df["InvoiceNo"].str.startswith("C")
    if params["cancellations"] == "drop":
        df = df[~is_cancel]
    # "net" mode: keep them, their Quantity is negative and sums against sales
    # (done after the quantity filter below)

    # 2. Remove non-product codes (postage, fees, manual adjustments, etc.)
    bad = {c.upper() for c in params["non_product_codes"]}
    df = df[~df["StockCode"].isin(bad)]
    df = df[~df["StockCode"].str.startswith("GIFT")]
    df = df[~df["StockCode"].str.startswith("ADJUST")]

    # 3. Remove invalid rows
    df = df[df["UnitPrice"] > 0]
    if params["cancellations"] == "drop":
        df = df[df["Quantity"] > 0]

    # 4. Flag (do not delete) extreme bulk orders for the spike scenario
    cutoff = df["Quantity"].quantile(params["outlier_percentile"] / 100)
    df["is_outlier"] = df["Quantity"] > cutoff

    # 5. Flag the lighting sub-catalogue
    kw = [k.upper() for k in params["lighting_keywords"]]
    df["is_lighting"] = df["Description"].apply(lambda d: any(k in d for k in kw))

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df.reset_index(drop=True)


def to_weekly_demand(df: pd.DataFrame, params: dict):
    """Return (weekly demand matrix SKU x week, sku master)."""
    df = df.copy()
    df["week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time

    weekly = (df.groupby(["StockCode", "week"])["Quantity"].sum()
                .unstack("week").sort_index(axis=1))

    # Full calendar so missing weeks become true zeros
    all_weeks = pd.date_range(weekly.columns.min(), weekly.columns.max(), freq="W-MON")
    weekly = weekly.reindex(columns=all_weeks)

    # Zeros only AFTER a SKU's first sale (before that = not yet stocked, stays NaN)
    first_active = weekly.notna().idxmax(axis=1).astype("datetime64[ns]")
    for sku, first in first_active.items():
        sku_pos = weekly.index.get_loc(sku)
        row = weekly.iloc[sku_pos].copy()
        first_ts = pd.Timestamp(first)
        before_first = pd.DatetimeIndex(weekly.columns) < first_ts
        weekly.iloc[sku_pos] = row.where(~before_first, other=float("nan")).fillna(0)
        weekly.iloc[sku_pos, before_first] = float("nan")

    # Keep SKUs with enough active weeks
    active_weeks = (weekly.fillna(0) > 0).sum(axis=1)
    weekly = weekly[active_weeks >= params["min_active_weeks"]]

    # SKU master: price, revenue, description, lighting flag
    g = df.groupby("StockCode")
    master = pd.DataFrame({
        "description": g["Description"].agg(lambda s: s.mode().iloc[0]),
        "unit_price": g["UnitPrice"].median(),
        "total_units": g["Quantity"].sum(),
        "total_revenue": g["Revenue"].sum(),
        "is_lighting": g["is_lighting"].max(),
    })
    master = master.loc[weekly.index].sort_values("total_revenue", ascending=False)
    return weekly, master