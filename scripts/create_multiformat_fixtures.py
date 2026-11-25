"""
Generate Multi-Format Test Fixtures for MIZU Ingestion & Validation
===================================================================
Generates:
1. valid_38_features.csv (Multiple observations, canonical names)
2. alternative_names.csv (Multiple observations, descriptive names)
3. valid_38_features.xlsx (Excel spreadsheet with 38 canonical features)
4. vertical_lab_report.xlsx (Vertical Parameter | Value | Unit layout)
5. table_report.pdf (PDF with extracted table of measurements)
6. key_value_report.pdf (PDF with narrative text measurements)
7. scanned_doc_simulated.pdf (Raster image PDF with 0 extractable text)
8. incomplete_24_features.csv (Fewer than 38 features)
9. ambiguous_columns.csv (Headers without treatment stages: 'pH', 'BOD', etc.)
10. unrecognised_fields.csv (Non-sensor columns)
11. incompatible_units.csv (Incompatible physical units)
12. malformed.xyz (Unsupported file format)
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from src.data_loader import load_water_treatment_data, COLUMN_NAMES
from src.ingestion import CANONICAL_FEATURES, CANONICAL_UNITS

def main():
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    # Base dataset sample (5 rows)
    base_df = load_water_treatment_data().head(5).copy()
    if "Date_Parsed" in base_df.columns:
        base_df = base_df.drop(columns=["Date_Parsed"])

    # 1. Valid 38-feature CSV
    csv_valid_path = fixtures_dir / "valid_38_features.csv"
    base_df.to_csv(csv_valid_path, index=False)
    print(f"Created {csv_valid_path} ({base_df.shape})")

    # 2. Alternative Names CSV
    alt_name_map = {
        "Q-E": "Influent Flow Rate (m3/day)",
        "ZN-E": "Influent Zinc Concentration (mg/L)",
        "PH-E": "Raw pH",
        "DBO-E": "Biological Oxygen Demand Influent (mg/L)",
        "DQO-E": "Chemical Oxygen Demand Influent (mg/L)",
        "SS-E": "Influent Suspended Solids (mg/L)",
        "SSV-E": "Influent Volatile Suspended Solids (%)",
        "SED-E": "Influent Settleable Solids (mL/L)",
        "COND-E": "Influent Electrical Conductivity (uS/cm)",
        "PH-P": "Primary Settler pH",
        "DBO-P": "Primary Settler BOD (mg/L)",
        "SS-P": "Primary Settler Suspended Solids (mg/L)",
        "SSV-P": "Primary Volatile Solids (%)",
        "SED-P": "Primary Settleable Solids (mL/L)",
        "COND-P": "Primary Conductivity (uS/cm)",
        "PH-D": "Secondary Settler pH",
        "DBO-D": "Secondary Settler BOD (mg/L)",
        "DQO-D": "Secondary Settler COD (mg/L)",
        "SS-D": "Secondary Suspended Solids (mg/L)",
        "SSV-D": "Secondary Volatile Solids (%)",
        "SED-D": "Secondary Settleable Solids (mL/L)",
        "COND-D": "Secondary Conductivity (uS/cm)",
        "PH-S": "Effluent pH",
        "DBO-S": "Effluent BOD (mg/L)",
        "DQO-S": "Effluent COD (mg/L)",
        "SS-S": "Effluent Suspended Solids (mg/L)",
        "SSV-S": "Effluent Volatile Solids (%)",
        "SED-S": "Effluent Settleable Solids (mL/L)",
        "COND-S": "Effluent Conductivity (uS/cm)",
        "RD-DBO-P": "Primary BOD Removal Efficiency (%)",
        "RD-SS-P": "Primary Solids Removal (%)",
        "RD-SED-P": "Primary Settleable Solids Removal (%)",
        "RD-DBO-S": "Secondary BOD Removal Efficiency (%)",
        "RD-DQO-S": "Secondary COD Removal (%)",
        "RD-DBO-G": "Global Plant BOD Removal Efficiency (%)",
        "RD-DQO-G": "Global COD Removal (%)",
        "RD-SS-G": "Global Solids Removal (%)",
        "RD-SED-G": "Global Settleable Solids Removal (%)"
    }
    alt_df = base_df.rename(columns=alt_name_map)
    alt_csv_path = fixtures_dir / "alternative_names.csv"
    alt_df.to_csv(alt_csv_path, index=False)
    print(f"Created {alt_csv_path} ({alt_df.shape})")

    # 3. Valid 38-feature Excel
    xlsx_valid_path = fixtures_dir / "valid_38_features.xlsx"
    base_df.to_excel(xlsx_valid_path, index=False)
    print(f"Created {xlsx_valid_path} ({base_df.shape})")

    # 4. Vertical Lab Report Excel
    row1 = base_df.iloc[0]
    vert_records = []
    for c in CANONICAL_FEATURES:
        unit = CANONICAL_UNITS.get(c, "")
        vert_records.append({
            "Parameter": alt_name_map.get(c, c),
            "Value": row1[c],
            "Unit": unit
        })
    vert_df = pd.DataFrame(vert_records)
    vert_xlsx_path = fixtures_dir / "vertical_lab_report.xlsx"
    vert_df.to_excel(vert_xlsx_path, index=False)
    print(f"Created {vert_xlsx_path} ({vert_df.shape})")

    # 5. Table Report PDF (using matplotlib table export)
    # We create a 1-page PDF containing a structured table of 5 observations
    pdf_table_path = fixtures_dir / "table_report.pdf"
    # Select 10 representative columns for a clean PDF table layout
    sub_cols = ["Date", "Q-E", "ZN-E", "PH-E", "DBO-E", "SS-E", "PH-S", "DBO-S", "SS-S", "SED-S"]
    sub_df = base_df[sub_cols].round(2)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table_data = [sub_df.columns.tolist()] + sub_df.values.tolist()
    tab = ax.table(cellText=table_data, loc='center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.2, 1.5)
    plt.title("MIZU Water Treatment Daily Operational Log", fontsize=12, pad=10)
    plt.savefig(pdf_table_path, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Created {pdf_table_path}")

    # 6. Key-Value Text Report PDF
    pdf_kv_path = fixtures_dir / "key_value_report.pdf"
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    kv_text = "WATER TREATMENT PLANT LABORATORY CERTIFICATE OF ANALYSIS\n"
    kv_text += "=========================================================\n"
    kv_text += "Date: 1990-03-13\n"
    kv_text += "Facility ID: Plant Alpha\n\n"
    kv_text += "MEASURED OPERATIONAL PARAMETERS:\n"
    for c in CANONICAL_FEATURES[:20]: # 20 features
        desc = alt_name_map.get(c, c)
        val = row1[c]
        unit = CANONICAL_UNITS.get(c, "")
        kv_text += f"{desc}: {val} {unit}\n"
    kv_text += "\nAuthorized By: Lead Process Chemist\n"
    
    ax.text(0.05, 0.95, kv_text, fontsize=9, family='monospace', verticalalignment='top')
    plt.savefig(pdf_kv_path, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Created {pdf_kv_path}")

    # 7. Scanned PDF Simulation (Pure raster image, no digital text)
    # Create a small raster image and save as PDF
    pdf_scanned_path = fixtures_dir / "scanned_doc_simulated.pdf"
    img = Image.new('RGB', (600, 800), color=(245, 245, 245))
    img.save(pdf_scanned_path, "PDF", resolution=100.0)
    print(f"Created {pdf_scanned_path} (Pure raster PDF with zero text)")

    # 8. Incomplete 24-feature CSV (missing 14 features)
    incomplete_cols = ["Date"] + CANONICAL_FEATURES[:24]
    incomp_df = base_df[incomplete_cols].copy()
    incomp_csv_path = fixtures_dir / "incomplete_24_features.csv"
    incomp_df.to_csv(incomp_csv_path, index=False)
    print(f"Created {incomp_csv_path} ({incomp_df.shape})")

    # 9. Ambiguous Columns CSV (e.g. stage-less 'pH', 'BOD', 'COD', 'Suspended Solids')
    ambig_df = pd.DataFrame({
        "Date": ["D-1/3/90", "D-2/3/90"],
        "pH": [7.8, 7.7],
        "BOD": [215, 240],
        "COD": [450, 480],
        "Suspended Solids": [180, 200]
    })
    ambig_path = fixtures_dir / "ambiguous_columns.csv"
    ambig_df.to_csv(ambig_path, index=False)
    print(f"Created {ambig_path} ({ambig_df.shape})")

    # 10. Unrecognised Fields CSV (Non-sensor parameters)
    unrec_df = pd.DataFrame({
        "Date": ["D-1/3/90", "D-2/3/90"],
        "Q-E": [44101, 39024],
        "Water Temperature": [18.5, 19.0],
        "Sample Batch ID": ["BATCH-A1", "BATCH-A2"],
        "Nitrate Concentration": [12.4, 11.8],
        "Turbidity NTU": [4.5, 5.1]
    })
    unrec_path = fixtures_dir / "unrecognised_fields.csv"
    unrec_df.to_csv(unrec_path, index=False)
    print(f"Created {unrec_path} ({unrec_df.shape})")

    # 11. Incompatible Units CSV (Flow given in mg/L, Zinc given in m3/day)
    incomp_units_df = pd.DataFrame({
        "Date": ["D-1/3/90", "D-2/3/90"],
        "Q-E (mg/L)": [44101, 39024], # Incompatible flow unit!
        "ZN-E (m3/day)": [1.5, 3.0]   # Incompatible zinc unit!
    })
    incomp_units_path = fixtures_dir / "incompatible_units.csv"
    incomp_units_df.to_csv(incomp_units_path, index=False)
    print(f"Created {incomp_units_path} ({incomp_units_df.shape})")

    # 12. Malformed File
    malformed_path = fixtures_dir / "malformed.xyz"
    with open(malformed_path, "wb") as f:
        f.write(b"CORRUPT_NON_STANDARD_DATA_BYTE_STREAM_!@#$%^&*")
    print(f"Created {malformed_path}")

    print("\nAll multi-format fixtures generated successfully!")

if __name__ == "__main__":
    main()

