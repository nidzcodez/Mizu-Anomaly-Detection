"""
Comprehensive Test Suite for MIZU Multi-Format Ingestion & Validation
=====================================================================
Automated tests covering:
1. Valid 38-feature CSV.
2. Excel input (.xlsx).
3. PDF table extraction.
4. PDF key-value extraction.
5. Alternative feature names.
6. Missing features (<38 features).
7. Ambiguous mappings.
8. Unrecognised fields.
9. Missing and incompatible units.
10. Multiple observations.
11. Single observations.
12. Malformed and unsupported files.
13. Scanned PDF handling where OCR is unavailable.
14. Confirmation that incomplete observations NEVER receive ML scores.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.ingestion import (
    MultiFormatIngestionEngine,
    FeatureMapper,
    CANONICAL_FEATURES,
    CANONICAL_UNITS,
    convert_measurement_unit
)
from app import load_all_models, run_mizu_inference


def test_multiformat_ingestion_suite():
    print("===========================================================================")
    print("MIZU Multi-Format Ingestion & Validation Automated Test Suite")
    print("===========================================================================")

    fixtures_dir = Path("tests/fixtures")
    engine = MultiFormatIngestionEngine()
    mapper = FeatureMapper()
    preprocessor, iforest, lof, kmeans, pca, explainer = load_all_models()

    # -------------------------------------------------------------------------
    # TEST 1: Valid 38-feature CSV (Multiple Observations)
    # -------------------------------------------------------------------------
    print("\n[Test 1] Valid 38-feature CSV (Multiple Observations)...")
    res1 = engine.ingest_file(fixtures_dir / "valid_38_features.csv")
    rep1 = res1.report
    assert rep1.file_type == "csv"
    assert rep1.total_rows == 5
    assert rep1.detected_canonical_count == 38
    assert rep1.coverage_pct == 100.0
    assert rep1.is_complete_38 is True
    assert rep1.is_ml_permitted is True
    assert len(rep1.missing_features) == 0
    # Run ML inference
    results1, _, missing1 = run_mizu_inference(res1.standardized_df, preprocessor, iforest, lof, kmeans, pca)
    assert missing1 is None
    assert len(results1) == 5
    print("✓ Passed: 38/38 canonical features mapped, ML inference executed across 5 observations.")

    # -------------------------------------------------------------------------
    # TEST 2: Excel Input (.xlsx, Multiple Observations)
    # -------------------------------------------------------------------------
    print("\n[Test 2] Excel Input (.xlsx, Multiple Observations)...")
    res2 = engine.ingest_file(fixtures_dir / "valid_38_features.xlsx")
    rep2 = res2.report
    assert rep2.file_type == "excel"
    assert rep2.total_rows == 5
    assert rep2.detected_canonical_count == 38
    assert rep2.is_complete_38 is True
    assert rep2.is_ml_permitted is True
    # Run ML inference
    results2, _, missing2 = run_mizu_inference(res2.standardized_df, preprocessor, iforest, lof, kmeans, pca)
    assert missing2 is None
    assert len(results2) == 5
    print("✓ Passed: Excel file ingested, all 38 features recognized, ML inference executed.")

    # -------------------------------------------------------------------------
    # TEST 3: PDF Table Extraction
    # -------------------------------------------------------------------------
    print("\n[Test 3] PDF Table Extraction (table_report.pdf)...")
    res3 = engine.ingest_file(fixtures_dir / "table_report.pdf")
    rep3 = res3.report
    assert rep3.file_type == "pdf_table"
    assert rep3.total_rows == 5
    assert rep3.detected_canonical_count > 5
    assert rep3.is_complete_38 is False # Only 9 features in table
    assert rep3.is_ml_permitted is False # Incomplete -> Abstain!
    print(f"✓ Passed: PDF table extracted {rep3.detected_canonical_count}/38 features. ML abstention enforced.")

    # -------------------------------------------------------------------------
    # TEST 4: PDF Key-Value Extraction (Narrative lab report)
    # -------------------------------------------------------------------------
    print("\n[Test 4] PDF Key-Value Extraction (key_value_report.pdf)...")
    res4 = engine.ingest_file(fixtures_dir / "key_value_report.pdf")
    rep4 = res4.report
    assert rep4.file_type == "pdf_key_value"
    assert rep4.total_rows == 1
    assert rep4.detected_canonical_count > 10
    assert rep4.is_complete_38 is False
    assert rep4.is_ml_permitted is False
    print(f"✓ Passed: PDF key-value extracted {rep4.detected_canonical_count}/38 features. ML abstention enforced.")

    # -------------------------------------------------------------------------
    # TEST 5: Alternative Feature Names & Unit Conversion
    # -------------------------------------------------------------------------
    print("\n[Test 5] Alternative Feature Names (alternative_names.csv)...")
    res5 = engine.ingest_file(fixtures_dir / "alternative_names.csv")
    rep5 = res5.report
    assert rep5.detected_canonical_count == 38
    assert rep5.is_complete_38 is True
    assert rep5.is_ml_permitted is True
    assert len(rep5.unrecognised_features) == 0
    # Run ML inference
    results5, _, _ = run_mizu_inference(res5.standardized_df, preprocessor, iforest, lof, kmeans, pca)
    assert len(results5) == 5
    print("✓ Passed: All 38 alternative descriptive names mapped successfully. ML permitted.")

    # -------------------------------------------------------------------------
    # TEST 6: Single Observation Lab Report (Vertical Excel)
    # -------------------------------------------------------------------------
    print("\n[Test 6] Single Observation Lab Report (vertical_lab_report.xlsx)...")
    res6 = engine.ingest_file(fixtures_dir / "vertical_lab_report.xlsx")
    rep6 = res6.report
    assert rep6.total_rows == 1
    assert rep6.detected_canonical_count == 38
    assert rep6.is_complete_38 is True
    assert rep6.is_ml_permitted is True
    # Run ML inference
    results6, _, _ = run_mizu_inference(res6.standardized_df, preprocessor, iforest, lof, kmeans, pca)
    assert len(results6) == 1
    print("✓ Passed: Vertical single-observation lab report transposed and validated for ML inference.")

    # -------------------------------------------------------------------------
    # TEST 7: Missing Features & Strict Abstention (<38 features)
    # -------------------------------------------------------------------------
    print("\n[Test 7] Missing Features & Strict Abstention (incomplete_24_features.csv)...")
    res7 = engine.ingest_file(fixtures_dir / "incomplete_24_features.csv")
    rep7 = res7.report
    assert rep7.detected_canonical_count == 24
    assert rep7.coverage_pct == 63.16
    assert rep7.is_complete_38 is False
    assert rep7.is_ml_permitted is False
    assert len(rep7.missing_features) == 14
    assert "effluent" in rep7.missing_by_stage or "efficiency" in rep7.missing_by_stage
    print(f"✓ Passed: Incomplete input ({rep7.detected_canonical_count}/38) strictly abstains from ML scoring.")

    # -------------------------------------------------------------------------
    # TEST 8: Ambiguous Mappings Handling (ambiguous_columns.csv)
    # -------------------------------------------------------------------------
    print("\n[Test 8] Ambiguous Mappings Handling (ambiguous_columns.csv)...")
    res8 = engine.ingest_file(fixtures_dir / "ambiguous_columns.csv")
    rep8 = res8.report
    assert len(rep8.ambiguous_features) >= 4
    ambig_names = [a["original_name"] for a in rep8.ambiguous_features]
    assert "pH" in ambig_names
    assert "BOD" in ambig_names
    assert "COD" in ambig_names
    assert "Suspended Solids" in ambig_names
    # Verify candidates are populated
    ph_entry = next(a for a in rep8.ambiguous_features if a["original_name"] == "pH")
    assert "PH-E" in ph_entry["candidate_canonicals"] and "PH-S" in ph_entry["candidate_canonicals"]
    assert rep8.is_ml_permitted is False
    print("✓ Passed: Ambiguous stage-less columns flagged with candidates; not auto-bound.")

    # -------------------------------------------------------------------------
    # TEST 9: Unrecognised Fields Handling (unrecognised_fields.csv)
    # -------------------------------------------------------------------------
    print("\n[Test 9] Unrecognised Fields Handling (unrecognised_fields.csv)...")
    res9 = engine.ingest_file(fixtures_dir / "unrecognised_fields.csv")
    rep9 = res9.report
    assert "Water Temperature" in rep9.unrecognised_features
    assert "Sample Batch ID" in rep9.unrecognised_features
    assert "Nitrate Concentration" in rep9.unrecognised_features
    assert "Turbidity NTU" in rep9.unrecognised_features
    print("✓ Passed: Non-sensor parameters preserved in unrecognised_features audit list.")

    # -------------------------------------------------------------------------
    # TEST 10: Missing and Incompatible Units (incompatible_units.csv)
    # -------------------------------------------------------------------------
    print("\n[Test 10] Incompatible Units (incompatible_units.csv)...")
    res10 = engine.ingest_file(fixtures_dir / "incompatible_units.csv")
    rep10 = res10.report
    assert any("Incompatible" in w for w in rep10.unit_warnings)
    # Direct unit conversion test
    val_conv, status_conv, msg_conv = convert_measurement_unit(100.0, "m3/h", "Q-E")
    assert status_conv == "converted"
    assert val_conv == 2400.0 # 100 m3/h * 24 = 2400 m3/day
    val_incomp, status_incomp, msg_incomp = convert_measurement_unit(50.0, "mg/L", "Q-E")
    assert status_incomp == "incompatible"
    assert val_incomp is None
    print("✓ Passed: Mathematical conversion verified (m3/h -> m3/day); incompatible unit rejected.")

    # -------------------------------------------------------------------------
    # TEST 11: Malformed and Unsupported Files (malformed.xyz)
    # -------------------------------------------------------------------------
    print("\n[Test 11] Malformed / Unsupported File Handling (malformed.xyz)...")
    res11 = engine.ingest_file(fixtures_dir / "malformed.xyz")
    rep11 = res11.report
    assert rep11.is_ml_permitted is False
    assert rep11.error_message is not None
    assert "Unsupported file extension" in rep11.error_message
    print("✓ Passed: Unsupported file handled gracefully without crashing.")

    # -------------------------------------------------------------------------
    # TEST 12: Scanned PDF Handling (OCR Unavailable Degradation)
    # -------------------------------------------------------------------------
    print("\n[Test 12] Scanned PDF Handling (scanned_doc_simulated.pdf)...")
    res12 = engine.ingest_file(fixtures_dir / "scanned_doc_simulated.pdf")
    rep12 = res12.report
    assert rep12.file_type == "scanned_pdf"
    assert rep12.is_ml_permitted is False
    assert rep12.error_message is not None
    assert "Scanned PDF Detected (OCR Unavailable)" in rep12.error_message
    print("✓ Passed: Scanned raster PDF detected, clear OCR degradation notice delivered.")

    # -------------------------------------------------------------------------
    # TEST 13: Absolute Confirmation - Incomplete Observations NEVER Receive ML
    # -------------------------------------------------------------------------
    print("\n[Test 13] Strict Abstention Verification (Incomplete NEVER receives ML)...")
    incomplete_test_cases = [
        fixtures_dir / "incomplete_24_features.csv",
        fixtures_dir / "table_report.pdf",
        fixtures_dir / "key_value_report.pdf",
        fixtures_dir / "ambiguous_columns.csv",
        fixtures_dir / "missing_columns.csv"
    ]
    for test_path in incomplete_test_cases:
        extracted = engine.ingest_file(test_path)
        assert extracted.report.is_ml_permitted is False, f"ML must be forbidden for {test_path.name}"
        # If user code tries to run inference directly, missing columns must block it
        results_attempt, _, missing_attempt = run_mizu_inference(
            extracted.standardized_df if extracted.standardized_df is not None else extracted.raw_df,
            preprocessor, iforest, lof, kmeans, pca
        )
        assert results_attempt is None, f"Inference engine must return None results for {test_path.name}"
        assert missing_attempt is not None and len(missing_attempt) > 0
    print("✓ Passed: Verified across 5 test fixtures that incomplete inputs strictly abstain from ML.")

    # -------------------------------------------------------------------------
    # TEST 14: User Confirmation and Resolution of Ambiguous Mappings
    # -------------------------------------------------------------------------
    print("\n[Test 14] User Confirmation and Resolution of Ambiguous Mappings...")
    res14 = engine.ingest_file(fixtures_dir / "ambiguous_columns.csv")
    assert len(res14.report.ambiguous_features) >= 4
    # User resolves the ambiguous columns
    user_confirmations = {
        "pH": "PH-E",
        "BOD": "DBO-E",
        "COD": "DQO-E",
        "Suspended Solids": "SS-E"
    }
    updated_res14 = engine.apply_user_mappings(res14, user_confirmations, filename="ambiguous_resolved.csv")
    up_rep14 = updated_res14.report
    assert len(up_rep14.ambiguous_features) == 0, "All ambiguous features should now be resolved"
    assert "PH-E" in up_rep14.mapped_features
    assert "DBO-E" in up_rep14.mapped_features
    assert "DQO-E" in up_rep14.mapped_features
    assert "SS-E" in up_rep14.mapped_features
    assert "PH-E" in updated_res14.standardized_df.columns
    assert up_rep14.detected_canonical_count == 4
    print("✓ Passed: User manual resolution successfully updated ambiguous columns into confirmed canonical mappings.")

    print("\n===========================================================================")
    print("ALL 14 MULTI-FORMAT INGESTION & ABSTENTION TESTS PASSED SUCCESSFULLY!")
    print("===========================================================================")


if __name__ == "__main__":
    test_multiformat_ingestion_suite()
