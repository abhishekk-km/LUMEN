"""Load UCI Online Retail II, clean, aggregate to SKU-week demand."""
"""Download (if needed), load, clean, and save weekly SKU demand.

Run from the project root:
    python -m src.data.load
"""
import urllib.request
import zipfile
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.clean import clean_transactions, to_weekly_demand

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


def download_raw() -> None:
    """Fetch the UCI zip once. Skips if a data file is already in data/raw."""
    RAW.mkdir(parents=True, exist_ok=True)
    if list(RAW.glob("*.xlsx")) or list(RAW.glob("*.csv")):
        return
    zip_path = RAW / "online_retail_II.zip"
    print("Downloading dataset from UCI (about 45 MB)...")
    try:
        urllib.request.urlretrieve(URL, zip_path)
    except Exception as e:
        raise SystemExit(
            f"\nAuto-download failed ({e}).\n"
            "Manual fix: open this link in your browser:\n"
            "  https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            "Click Download, unzip, and put the .xlsx file in data/raw/ . Then re-run."
        )
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)
    zip_path.unlink()
    print("Done.")


def read_raw() -> pd.DataFrame:
    """Read the raw file. Excel is slow, so cache a CSV copy after first read."""
    INTERIM.mkdir(parents=True, exist_ok=True)
    cache = INTERIM / "online_retail_raw.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["InvoiceDate"], low_memory=False)

    download_raw()
    files = list(RAW.glob("*.xlsx"))
    if files:
        print("Reading Excel (first time only, takes 2-5 minutes)...")
        sheets = pd.read_excel(files[0], sheet_name=None)   # both year sheets
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(next(RAW.glob("*.csv")), encoding="latin1", low_memory=False)

    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice",
                            "Customer ID": "CustomerID"})
    df.to_csv(cache, index=False)
    return df


def main() -> None:
    params = load_params()
    raw = read_raw()
    print(f"Raw rows: {len(raw):,}")

    clean = clean_transactions(raw, params)
    print(f"Clean rows: {len(clean):,}")

    weekly, master = to_weekly_demand(clean, params)
    print(f"SKUs kept: {len(weekly):,} | weeks: {weekly.shape[1]}")
    print(f"Lighting SKUs: {int(master['is_lighting'].sum()):,}")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "sku_weekly_demand.csv")
    master.to_csv(PROCESSED / "sku_master_base.csv")
    clean.to_csv(INTERIM / "clean_transactions.csv", index=False)
    print("Saved to data/processed/")


if __name__ == "__main__":
    main()