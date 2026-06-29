"""
╔══════════════════════════════════════════════════════════════╗
║         OLX Car Dataset - Data Cleaning & Preprocessing       ║
║         Run this script to understand how raw data is         ║
║         cleaned before training the ML model.                 ║
╚══════════════════════════════════════════════════════════════╝

STEP-BY-STEP DATA CLEANING GUIDE:
──────────────────────────────────
1. Price Column    → "Rs 18.50 Lacs" → 1,850,000 (numeric)
2. Mileage Column  → "90,000 km" / "New" → float number
3. Car Age         → 2026 - Year = Age in years
4. Condition       → "Used"/"New"/NaN → 0 or 1 (binary)
5. Brand & Model   → Extracted from Title using keyword matching
6. Missing Values  → Dropped where Price/Mileage missing, Seats filled with 5
7. Outliers        → Price: 5L–1.5Cr | Mileage: max 3 lakh km
8. Encoding        → One-Hot Encoding for Brand, Model, Fuel
"""

import pandas as pd
import numpy as np

# ─── Brand-Model Keyword Map ──────────────────────────────────────────────────
BRAND_MODEL_MAP = {
    'Toyota':  ['corolla', 'yaris', 'fortuner', 'vitz', 'aqua', 'passo',
                'raize', 'camry', 'surf', 'hilux', 'prius'],
    'Honda':   ['civic', 'city', 'br-v', 'vezel', 'accord', 'n wgn', 'n box'],
    'Suzuki':  ['alto', 'cultus', 'wagon r', 'mehran', 'swift', 'bolan',
                'ravi', 'liana', 'baleno', 'khyber'],
    'Hyundai': ['tucson', 'sonata', 'elantra', 'santa fe'],
    'Kia':     ['sportage', 'picanto', 'stonic', 'sorento'],
    'Nissan':  ['dayz', 'juke', 'sunny', 'moco'],
    'Daihatsu':['mira', 'cuore', 'move', 'hijet', 'boon'],
}

def extract_brand_model(title: str):
    """
    Title se Brand aur Model extract karta hai keyword matching se.
    Agar koi match na mile → ('Others', 'Others') return karta hai.
    """
    title = str(title).lower()
    for brand, models in BRAND_MODEL_MAP.items():
        for model in models:
            if model in title:
                return brand, model.title()
    return 'Others', 'Others'


def clean_price(price_str: str):
    """
    Raw price string ko numeric PKR mein convert karta hai.

    Examples:
        "Rs 18.50 Lacs"  → 1,850,000
        "Rs 3.90 Crore"  → 39,000,000
        "Rs 72 Lacs"     → 7,200,000
    """
    p = str(price_str).replace('Rs', '').replace(',', '').strip()
    if 'Crore' in p:
        return float(p.replace('Crore', '').strip()) * 10_000_000
    if 'Lacs' in p or 'Lac' in p:
        return float(p.replace('Lacs', '').replace('Lac', '').strip()) * 100_000
    try:
        return float(p)
    except ValueError:
        return np.nan


def clean_mileage(mileage_str: str):
    """
    Mileage string ko numeric km mein convert karta hai.

    Examples:
        "90,000 km" → 90000.0
        "New"       → 0.0   (naya car = 0 km)
    """
    m = str(mileage_str).replace(',', '').strip()
    if 'km' in m.lower():
        return float(m.lower().replace('km', '').strip())
    if 'new' in m.lower():
        return 0.0
    try:
        return float(m)
    except ValueError:
        return np.nan


def convert_condition(row):
    """
    Condition column ko binary mein convert karta hai.
        1 = New car
        0 = Used car

    Agar Condition missing hai (NaN), tab Mileage aur Age se decide karta hai:
        - Mileage ≤ 30,000 km AND Age ≤ 2 years → New (1)
        - Warna → Used (0)
    """
    miles = row['Mileage']
    cond  = row['Condition']
    age   = row['Car_Age']

    if pd.isna(cond):
        return 1 if (miles <= 30_000 and age <= 2) else 0
    return 1 if 'New' in str(cond) else 0


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Poora data cleaning pipeline — raw CSV → model-ready DataFrame.

    Returns cleaned DataFrame with all features.
    """
    print("=" * 60)
    print("  OLX CAR DATASET — DATA CLEANING PIPELINE")
    print("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"\n[1] Raw Data Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"    Missing values per column:\n{df.isnull().sum().to_string()}")

    # ── Step 2: Extract Brand & Model from Title ──────────────
    df[['Brand_Final', 'Model_Final']] = df['Title'].apply(
        lambda x: pd.Series(extract_brand_model(x))
    )
    print(f"\n[2] Brand/Model extracted from Title")
    print(f"    Brand distribution:\n{df['Brand_Final'].value_counts().to_string()}")

    # ── Step 3: Clean Price ───────────────────────────────────
    df['Price'] = df['Price'].apply(clean_price)
    print(f"\n[3] Price cleaned → numeric PKR")
    print(f"    Price range: {df['Price'].min():,.0f} – {df['Price'].max():,.0f} PKR")

    # ── Step 4: Clean Mileage ─────────────────────────────────
    df['Mileage'] = df['Mileage'].apply(clean_mileage)
    print(f"\n[4] Mileage cleaned → numeric km")

    # ── Step 5: Car Age ───────────────────────────────────────
    df['Car_Age'] = 2026 - df['Year']
    print(f"\n[5] Car_Age calculated (2026 - Year)")
    print(f"    Age range: {df['Car_Age'].min()} – {df['Car_Age'].max()} years")

    # ── Step 6: Drop rows with critical missing values ────────
    before = len(df)
    df = df.dropna(subset=['Price', 'Mileage', 'Make'])
    print(f"\n[6] Dropped rows with missing Price/Mileage/Make")
    print(f"    Rows dropped: {before - len(df)} | Remaining: {len(df)}")

    # ── Step 7: Fill missing Seats ────────────────────────────
    df['Seats'] = df['Seats'].fillna(5)
    print(f"\n[7] Missing Seats filled with 5 (default)")

    # ── Step 8: Outlier Removal ───────────────────────────────
    before = len(df)
    df = df[(df['Price'] >= 500_000) & (df['Price'] <= 15_000_000)]
    df = df[df['Mileage'] <= 300_000]
    print(f"\n[8] Outliers removed:")
    print(f"    Price: kept 5 Lacs – 1.5 Crore PKR")
    print(f"    Mileage: kept ≤ 3,00,000 km")
    print(f"    Rows removed: {before - len(df)} | Remaining: {len(df)}")

    # ── Step 9: Binary Condition ──────────────────────────────
    df['Condition'] = df.apply(convert_condition, axis=1)
    print(f"\n[9] Condition → Binary (New=1, Used=0)")
    print(f"    New: {df['Condition'].sum()} | Used: {(df['Condition']==0).sum()}")

    # ── Step 10: Fuel Cleaning ────────────────────────────────
    df['Fuel_Clean'] = df['Fuel'].apply(
        lambda x: x if x in ['Petrol', 'Hybrid'] else 'Others'
    )
    print(f"\n[10] Fuel types simplified → Petrol | Hybrid | Others")

    # ── Step 11: One-Hot Encoding ─────────────────────────────
    df_enc = pd.get_dummies(
        df,
        columns=['Brand_Final', 'Model_Final', 'Fuel_Clean'],
        prefix=['Brand', 'Model', 'Fuel'],
        dtype=int
    )

    # ── Step 12: Select Final Columns ────────────────────────
    feature_cols = (
        ['Price', 'Car_Age', 'Mileage', 'Seats', 'Condition']
        + [c for c in df_enc.columns if c.startswith(('Brand_', 'Model_', 'Fuel_'))]
    )
    df_final = df_enc[feature_cols]

    print(f"\n[11] One-Hot Encoding applied")
    print(f"     Final shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"\n{'='*60}")
    print(f"  ✅ DATA CLEANING COMPLETE")
    print(f"     Final columns: {df_final.columns.tolist()}")
    print(f"{'='*60}\n")

    return df_final


# ── Run directly to see cleaning output ───────────────────────────────────────
if __name__ == '__main__':
    df_clean = load_and_clean('olx_car_dataset_2025.csv')
    print("\nSample of cleaned data:")
    print(df_clean.head())
    print("\nData types:")
    print(df_clean.dtypes)
    print("\nBasic stats:")
    print(df_clean.describe())
