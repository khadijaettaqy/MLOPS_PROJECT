"""
preprocess.py

Data loading and exploration utility for the Heart Disease dataset.

Usage:
    python preprocess.py
    python preprocess.py --head 10
    python preprocess.py --stats
"""

import argparse
import pandas as pd
from pathlib import Path


FEATURE_COLS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
TARGET_COL = 'target'


DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[2] / "data" / "heart.csv")


def load_data(filepath=DEFAULT_DATA_PATH):
    """Load and clean the heart disease dataset."""
    df = pd.read_csv(filepath)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main():
    parser = argparse.ArgumentParser(description="Heart Disease Dataset Utility")
    parser.add_argument("--head", type=int, help="Show first N rows")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--file", type=str, default=DEFAULT_DATA_PATH, help="Path to heart.csv")
    args = parser.parse_args()

    df = load_data(args.file)

    print("\nLoaded Heart Disease Dataset")
    print("=" * 50)
    print(f"Samples : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Target  : {TARGET_COL}")
    print(f"Classes : {df[TARGET_COL].unique().tolist()}")
    print("=" * 50)

    if args.head:
        print(f"\nFirst {args.head} rows:\n")
        print(df.head(args.head))

    if args.stats:
        print("\nStatistical Summary:\n")
        print(df.describe())

        print("\nTarget Distribution:\n")
        print(df[TARGET_COL].value_counts())


if __name__ == "__main__":
    main()
