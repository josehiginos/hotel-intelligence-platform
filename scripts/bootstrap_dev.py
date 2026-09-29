#!/usr/bin/env python3
"""
Bootstrap local / Cloud Agent development state.

Default: initialize the SQLite analytics database from committed CSVs.
Optional: train cancellation models so the Streamlit predictor page has artifacts.

Usage:
  python scripts/bootstrap_dev.py
  python scripts/bootstrap_dev.py --train-cancellation
  python scripts/bootstrap_dev.py --train-cancellation --fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def init_database() -> None:
    from src.module_3_conversational.database.init_db import initialize_database

    initialize_database()


def train_cancellation(fast: bool = False) -> None:
    from src.module_1_predictive.cancellation.eda import load_data
    from src.module_1_predictive.cancellation.features import run_feature_engineering
    from src.module_1_predictive.cancellation.train import run_training_pipeline

    df = load_data()
    df, feature_cols = run_feature_engineering(df)
    run_training_pipeline(df, feature_cols, fast=fast)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Hotel Intelligence Platform data/models")
    parser.add_argument(
        "--train-cancellation",
        action="store_true",
        help="Train cancellation models into models/cancellation/",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Faster cancellation training (fewer models, no CV)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Hotel Intelligence Platform — Dev Bootstrap")
    print("=" * 60)

    print("\n[1/2] Initialize SQLite database")
    init_database()

    if args.train_cancellation:
        print("\n[2/2] Train cancellation models" + (" (fast)" if args.fast else ""))
        train_cancellation(fast=args.fast)
    else:
        print("\n[2/2] Skipping model training (pass --train-cancellation to enable)")

    print("\nBootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
