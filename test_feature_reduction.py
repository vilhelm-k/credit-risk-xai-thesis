#!/usr/bin/env python3
"""
Quick test to verify feature reduction changes.
This script checks that:
1. The removed features are no longer generated
2. The remaining features are still present
3. The total feature count matches expectations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

from credit_risk_xai.config import (
    FEATURE_CACHE_PATH,
    FEATURES_FOR_MODEL,
    RATIO_FEATURE_NAMES,
    CRISIS_FEATURE_NAMES,
    MACRO_FEATURE_NAMES,
    BASE_MODEL_FEATURES,
    NY_COLS,
    KEPT_RAW_COLS,
    ENGINEERED_FEATURE_NAMES
)

# Features that should have been removed
REMOVED_FEATURES = [
    'ser_aktiv',
    'ever_failed',
    'ratio_ebitda_interest_cov',
    'revenue_vs_gdp',
    'last_event_within_1y',
    'last_event_within_2y',
    'last_event_within_3y',
    'last_event_within_5y',
    'interest_avg_medium',
    'interest_avg_long',
    'inflation_trailing_3y'
]

def main():
    print("=" * 80)
    print("FEATURE REDUCTION VERIFICATION TEST")
    print("=" * 80)

    # Check configuration
    print("\n1. Configuration Check:")
    print(f"   - BASE_MODEL_FEATURES: {len(BASE_MODEL_FEATURES)} features")
    print(f"   - NY_COLS: {len(NY_COLS)} features")
    print(f"   - KEPT_RAW_COLS: {len(KEPT_RAW_COLS)} features")
    print(f"   - RATIO_FEATURE_NAMES: {len(RATIO_FEATURE_NAMES)} features")
    print(f"   - CRISIS_FEATURE_NAMES: {len(CRISIS_FEATURE_NAMES)} features")
    print(f"   - MACRO_FEATURE_NAMES: {len(MACRO_FEATURE_NAMES)} features")
    print(f"   - ENGINEERED_FEATURE_NAMES total: {len(ENGINEERED_FEATURE_NAMES)} features")
    print(f"   - FEATURES_FOR_MODEL total: {len(FEATURES_FOR_MODEL)} features")

    # Verify removed features are not in config
    print("\n2. Removed Features Check:")
    for feature in REMOVED_FEATURES:
        if feature in FEATURES_FOR_MODEL:
            print(f"   ❌ ERROR: {feature} still in FEATURES_FOR_MODEL!")
        else:
            print(f"   ✓ {feature} successfully removed from config")

    # Check if cached feature file exists
    if FEATURE_CACHE_PATH.exists():
        print(f"\n3. Loading cached features from {FEATURE_CACHE_PATH}")
        df = pd.read_parquet(FEATURE_CACHE_PATH)
        print(f"   - Dataset shape: {df.shape}")
        print(f"   - Total columns: {df.shape[1]}")

        # Check for removed features in actual data
        print("\n4. Checking for removed features in data:")
        removed_found = []
        for feature in REMOVED_FEATURES:
            if feature in df.columns:
                removed_found.append(feature)
                print(f"   ⚠️  WARNING: {feature} still exists in cached data")

        if removed_found:
            print(f"\n   Note: {len(removed_found)} removed features still in cache.")
            print("   This is expected if the feature engineering pipeline hasn't been re-run yet.")
            print("   Run 'python -m credit_risk_xai.features.engineer' to regenerate features.")
        else:
            print("   ✓ All removed features are absent from the data")

        # Check for expected features
        print("\n5. Spot-checking expected features:")
        expected_samples = [
            'ratio_personnel_cost',
            'ratio_ebit_interest_cov',  # Keep this one
            'years_since_last_credit_event',
            'event_count_total',
            'interest_avg_short',
            'term_spread',
            'real_revenue_growth',
            'ny_avkegkap'
        ]

        for feature in expected_samples:
            if feature in FEATURES_FOR_MODEL:
                status = "✓ in config"
                if FEATURE_CACHE_PATH.exists() and feature in df.columns:
                    status += " & data"
            else:
                status = "❌ MISSING from config!"
            print(f"   {feature}: {status}")
    else:
        print(f"\n3. Feature cache not found at {FEATURE_CACHE_PATH}")
        print("   Run the feature engineering pipeline to generate features.")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original feature count: 117")
    print(f"Features removed: {len(REMOVED_FEATURES)}")
    print(f"Expected new count: 106")
    print(f"Actual config count: {len(FEATURES_FOR_MODEL)}")

    if len(FEATURES_FOR_MODEL) == 106:
        print("\n✅ Feature reduction successful! Configuration updated correctly.")
    else:
        print(f"\n⚠️  Feature count mismatch. Expected 106, got {len(FEATURES_FOR_MODEL)}")

    print("\nNext steps:")
    print("1. Run feature engineering: python -m credit_risk_xai.features.engineer")
    print("2. Re-run your modeling notebook to test performance impact")
    print("3. Monitor AUC and PR-AUC to ensure no significant degradation")

if __name__ == "__main__":
    main()