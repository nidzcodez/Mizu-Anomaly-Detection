"""
Test Suite for MIZU Dashboard Inference Engine & Upload Validation
==================================================================
Tests:
1. Valid sample CSV inference (10 rows).
2. Schema validation error handling on missing columns CSV.
3. Robust handling of missing / NaN values via fitted SimpleImputer.
4. Single row inference & feature ablation attribution.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from app import load_all_models, run_mizu_inference

def test_dashboard_pipeline():
    print("===========================================================================")
    print("MIZU Dashboard Inference & Validation Test Suite")
    print("===========================================================================")

    # 1. Load models
    print("\n[Step 1] Loading pre-trained MIZU model artifacts...")
    preprocessor, iforest, lof, kmeans, pca, explainer = load_all_models()
    assert preprocessor.is_fitted, "Preprocessor must be fitted!"
    assert iforest.is_fitted, "IsolationForest must be fitted!"
    assert lof.is_fitted, "LOF must be fitted!"
    assert kmeans.is_fitted, "KMeans must be fitted!"
    assert pca.is_fitted, "PCA must be fitted!"
    print("✓ All 5 pipeline artifacts and explainer successfully loaded.")

    fixtures_dir = Path("tests/fixtures")

    # 2. Test valid sample
    print("\n[Step 2] Testing valid_sample.csv (10 rows, all 38 continuous features)...")
    valid_df = pd.read_csv(fixtures_dir / "valid_sample.csv")
    results, X_scaled, missing = run_mizu_inference(valid_df, preprocessor, iforest, lof, kmeans, pca)
    assert missing is None, f"Expected no missing columns, got: {missing}"
    assert results is not None, "Results dataframe must not be None"
    assert len(results) == 10, f"Expected 10 rows, got {len(results)}"
    assert "concordance" in results.columns
    assert "iforest_score" in results.columns
    assert "lof_score" in results.columns
    assert "cluster" in results.columns
    assert "PC1" in results.columns and "PC2" in results.columns
    print(f"✓ Valid sample passed! Results shape: {results.shape}, Concordance counts:\n{results['concordance'].value_counts().to_dict()}")

    # 3. Test missing columns failure handling
    print("\n[Step 3] Testing missing_columns.csv (Missing ZN-E and SED-S)...")
    missing_df = pd.read_csv(fixtures_dir / "missing_columns.csv")
    results, X_scaled, missing = run_mizu_inference(missing_df, preprocessor, iforest, lof, kmeans, pca)
    assert results is None, "Inference must halt and return None for results when schema fails"
    assert missing is not None, "Missing columns list must be returned"
    assert "ZN-E" in missing and "SED-S" in missing, f"Expected ZN-E and SED-S in missing list, got {missing}"
    print(f"✓ Schema validation correctly caught missing features: {missing}")

    # 4. Test missing / NaN values handling
    print("\n[Step 4] Testing with_nans.csv (Contains missing values across sensor streams)...")
    nans_df = pd.read_csv(fixtures_dir / "with_nans.csv")
    assert nans_df.isnull().sum().sum() > 0, "Fixture must contain NaNs"
    results, X_scaled, missing = run_mizu_inference(nans_df, preprocessor, iforest, lof, kmeans, pca)
    assert missing is None, "Columns must match"
    assert results is not None, "Results must be computed"
    # Verify no NaNs in transformed output or model scores
    assert not X_scaled.isnull().values.any(), "Transformed features must have 0 NaNs after median imputation"
    assert not results["iforest_score"].isnull().values.any(), "iForest scores must have 0 NaNs"
    assert not results["lof_score"].isnull().values.any(), "LOF scores must have 0 NaNs"
    print(f"✓ Missing value imputation passed! Imputed all NaNs seamlessly across {len(results)} rows.")

    # 5. Test single row inference & feature ablation
    print("\n[Step 5] Testing single_row.csv (1 row: D-13/3/90)...")
    single_df = pd.read_csv(fixtures_dir / "single_row.csv")
    results, X_scaled, missing = run_mizu_inference(single_df, preprocessor, iforest, lof, kmeans, pca)
    assert len(results) == 1, f"Expected 1 row, got {len(results)}"
    row_date = results["Date"].iloc[0]
    print(f"Single row date: {row_date}, iForest Score: {results['iforest_score'].iloc[0]}, LOF Score: {results['lof_score'].iloc[0]}, Cluster: {results['cluster'].iloc[0]}")
    
    # Run explainer on the single row
    top_df_iforest, meta_iforest = explainer.explain_sample(
        sample=X_scaled.iloc[[0]],
        raw_row=single_df.iloc[0],
        sample_id=row_date,
        model_type="iforest",
        top_n=5
    )
    assert len(top_df_iforest) == 5, "Must return top 5 features"
    assert meta_iforest["top_feature_code"] == "SED-S", f"Expected SED-S as top feature, got {meta_iforest['top_feature_code']}"
    print(f"✓ Feature ablation passed for single row! Top feature for {row_date}: {meta_iforest['top_feature']} (Δ Score: {meta_iforest['top_score_reduction']:.6f})")

    print("\n===========================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! Dashboard inference engine is production-ready.")
    print("===========================================================================")

if __name__ == "__main__":
    test_dashboard_pipeline()

