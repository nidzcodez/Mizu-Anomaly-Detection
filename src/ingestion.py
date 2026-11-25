"""
MIZU: Multi-Format Ingestion, Feature Mapping & Unit Conversion Module
======================================================================
Author: MIZU Team
Purpose:
  1. Multi-format ingestion: CSV, Excel (.xlsx, .xls), and PDF (tables & key-value reports).
  2. Scanned PDF detection with graceful OCR degradation messaging.
  3. Intelligent feature recognition with extensive alias dictionary and ambiguity detection.
  4. Unit detection and mathematically safe conversions.
  5. Strict abstention: prevents ML inference on incomplete (<38) feature vectors.
"""

import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data_loader import COLUMN_NAMES, get_feature_groups
from src.explainability import SENSOR_METADATA


# =============================================================================
# 1. CANONICAL SENSORS & UNIT CONVERSION DEFINITIONS
# =============================================================================

# List of 38 canonical continuous sensors (excluding 'Date')
CANONICAL_FEATURES: List[str] = [c for c in COLUMN_NAMES if c != "Date"]

# Canonical Units for all 38 continuous sensors
CANONICAL_UNITS: Dict[str, str] = {
    # Flow
    "Q-E": "m³/day",
    # Concentrations (mg/L)
    "ZN-E": "mg/L", "DBO-E": "mg/L", "DQO-E": "mg/L", "SS-E": "mg/L",
    "DBO-P": "mg/L", "SS-P": "mg/L",
    "DBO-D": "mg/L", "DQO-D": "mg/L", "SS-D": "mg/L",
    "DBO-S": "mg/L", "DQO-S": "mg/L", "SS-S": "mg/L",
    # pH units
    "PH-E": "pH units", "PH-P": "pH units", "PH-D": "pH units", "PH-S": "pH units",
    # Settleable Solids (mL/L)
    "SED-E": "mL/L", "SED-P": "mL/L", "SED-D": "mL/L", "SED-S": "mL/L",
    # Electrical Conductivity (µS/cm)
    "COND-E": "µS/cm", "COND-P": "µS/cm", "COND-D": "µS/cm", "COND-S": "µS/cm",
    # Volatile Solids (%)
    "SSV-E": "%", "SSV-P": "%", "SSV-D": "%", "SSV-S": "%",
    # Efficiencies (%)
    "RD-DBO-P": "%", "RD-SS-P": "%", "RD-SED-P": "%",
    "RD-DBO-S": "%", "RD-DQO-S": "%",
    "RD-DBO-G": "%", "RD-DQO-G": "%", "RD-SS-G": "%", "RD-SED-G": "%"
}


# =============================================================================
# 2. ALIAS DICTIONARY FOR SENSOR NORMALIZATION
# =============================================================================

def _normalize_string(s: str) -> str:
    """Normalize label: lowercase, strip punctuation, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    # Replace separators like hyphens, underscores, slashes with spaces
    cleaned = re.sub(r"[\-_\./\(\)\[\],:]+", " ", s.lower())
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", cleaned).strip()


# Documented Alias Mapping: Map normalized alternative strings directly to canonical code
FEATURE_ALIASES: Dict[str, str] = {
    # Stage 1: Influent (-E)
    "q e": "Q-E", "qe": "Q-E", "flow": "Q-E", "influent flow": "Q-E", "influent flow rate": "Q-E",
    "raw flow": "Q-E", "inflow": "Q-E", "entry flow": "Q-E", "plant inflow": "Q-E", "flow rate influent": "Q-E",
    "volumetric flow": "Q-E", "q influent": "Q-E", "flow influent": "Q-E", "intake flow": "Q-E",

    "zn e": "ZN-E", "zne": "ZN-E", "zinc": "ZN-E", "influent zinc": "ZN-E", "raw zinc": "ZN-E",
    "zinc concentration": "ZN-E", "zn influent": "ZN-E", "influent zn": "ZN-E", "zinc influent": "ZN-E",

    "ph e": "PH-E", "phe": "PH-E", "influent ph": "PH-E", "raw ph": "PH-E", "ph influent": "PH-E",
    "intake ph": "PH-E", "entry ph": "PH-E", "influent water ph": "PH-E",

    "dbo e": "DBO-E", "dboe": "DBO-E", "bod e": "DBO-E", "bode": "DBO-E", "influent bod": "DBO-E",
    "influent dbo": "DBO-E", "raw bod": "DBO-E", "raw dbo": "DBO-E", "bod5 influent": "DBO-E",
    "dbo5 influent": "DBO-E", "influent bod5": "DBO-E", "influent dbo5": "DBO-E",
    "biochemical oxygen demand influent": "DBO-E", "biological oxygen demand influent": "DBO-E",
    "influent biological oxygen demand": "DBO-E", "influent biochemical oxygen demand": "DBO-E",

    "dqo e": "DQO-E", "dqoe": "DQO-E", "cod e": "DQO-E", "code": "DQO-E", "influent cod": "DQO-E",
    "influent dqo": "DQO-E", "raw cod": "DQO-E", "raw dqo": "DQO-E", "chemical oxygen demand influent": "DQO-E",
    "influent chemical oxygen demand": "DQO-E", "influent cod total": "DQO-E",

    "ss e": "SS-E", "sse": "SS-E", "tss e": "SS-E", "influent ss": "SS-E", "influent tss": "SS-E",
    "influent suspended solids": "SS-E", "raw suspended solids": "SS-E", "raw ss": "SS-E",
    "suspended solids influent": "SS-E", "total suspended solids influent": "SS-E",
    "influent total suspended solids": "SS-E",

    "ssv e": "SSV-E", "ssve": "SSV-E", "vss e": "SSV-E", "influent ssv": "SSV-E", "influent vss": "SSV-E",
    "influent volatile suspended solids": "SSV-E", "influent volatile solids": "SSV-E",
    "volatile suspended solids influent": "SSV-E", "volatile solids influent": "SSV-E",

    "sed e": "SED-E", "sede": "SED-E", "influent sed": "SED-E", "influent settleable solids": "SED-E",
    "raw settleable solids": "SED-E", "settleable solids influent": "SED-E",

    "cond e": "COND-E", "conde": "COND-E", "ec e": "COND-E", "influent cond": "COND-E",
    "influent conductivity": "COND-E", "raw conductivity": "COND-E", "conductivity influent": "COND-E",
    "electrical conductivity influent": "COND-E", "influent electrical conductivity": "COND-E",

    # Stage 2: Primary Settler (-P)
    "ph p": "PH-P", "php": "PH-P", "primary ph": "PH-P", "primary settler ph": "PH-P",
    "settler 1 ph": "PH-P", "ph primary": "PH-P", "primary sedimentation ph": "PH-P",

    "dbo p": "DBO-P", "dbop": "DBO-P", "bod p": "DBO-P", "bodp": "DBO-P", "primary bod": "DBO-P",
    "primary dbo": "DBO-P", "primary settler bod": "DBO-P", "bod primary": "DBO-P",
    "primary bod5": "DBO-P", "settler 1 bod": "DBO-P",

    "ss p": "SS-P", "ssp": "SS-P", "tss p": "SS-P", "primary ss": "SS-P", "primary suspended solids": "SS-P",
    "primary settler ss": "SS-P", "suspended solids primary": "SS-P", "settler 1 ss": "SS-P",

    "ssv p": "SSV-P", "ssvp": "SSV-P", "vss p": "SSV-P", "primary ssv": "SSV-P", "primary volatile solids": "SSV-P",
    "primary volatile suspended solids": "SSV-P", "volatile suspended solids primary": "SSV-P",

    "sed p": "SED-P", "sedp": "SED-P", "primary sed": "SED-P", "primary settleable solids": "SED-P",
    "settleable solids primary": "SED-P", "primary settler settleable solids": "SED-P",

    "cond p": "COND-P", "condp": "COND-P", "ec p": "COND-P", "primary cond": "COND-P",
    "primary conductivity": "COND-P", "conductivity primary": "COND-P", "primary electrical conductivity": "COND-P",

    # Stage 3: Secondary Settler / Aeration Basin (-D)
    "ph d": "PH-D", "phd": "PH-D", "secondary ph": "PH-D", "secondary settler ph": "PH-D",
    "aeration ph": "PH-D", "settler 2 ph": "PH-D", "ph secondary": "PH-D", "biological settler ph": "PH-D",

    "dbo d": "DBO-D", "dbod": "DBO-D", "bod d": "DBO-D", "bodd": "DBO-D", "secondary bod": "DBO-D",
    "secondary dbo": "DBO-D", "secondary settler bod": "DBO-D", "aeration bod": "DBO-D", "bod secondary": "DBO-D",
    "secondary bod5": "DBO-D",

    "dqo d": "DQO-D", "dqod": "DQO-D", "cod d": "DQO-D", "codd": "DQO-D", "secondary cod": "DQO-D",
    "secondary dqo": "DQO-D", "secondary settler cod": "DQO-D", "aeration cod": "DQO-D", "cod secondary": "DQO-D",

    "ss d": "SS-D", "ssd": "SS-D", "tss d": "SS-D", "secondary ss": "SS-D", "secondary suspended solids": "SS-D",
    "secondary settler ss": "SS-D", "aeration ss": "SS-D", "suspended solids secondary": "SS-D",
    "mixed liquor suspended solids": "SS-D",

    "ssv d": "SSV-D", "ssvd": "SSV-D", "vss d": "SSV-D", "secondary ssv": "SSV-D", "secondary volatile solids": "SSV-D",
    "secondary volatile suspended solids": "SSV-D", "volatile suspended solids secondary": "SSV-D",

    "sed d": "SED-D", "sedd": "SED-D", "secondary sed": "SED-D", "secondary settleable solids": "SED-D",
    "settleable solids secondary": "SED-D", "secondary settler settleable solids": "SED-D",

    "cond d": "COND-D", "condd": "COND-D", "ec d": "COND-D", "secondary cond": "COND-D",
    "secondary conductivity": "COND-D", "conductivity secondary": "COND-D", "secondary electrical conductivity": "COND-D",

    # Stage 4: Plant Effluent / Discharge (-S)
    "ph s": "PH-S", "phs": "PH-S", "effluent ph": "PH-S", "discharge ph": "PH-S", "treated ph": "PH-S",
    "final ph": "PH-S", "ph effluent": "PH-S", "ph discharge": "PH-S", "outlet ph": "PH-S",

    "dbo s": "DBO-S", "dbos": "DBO-S", "bod s": "DBO-S", "bods": "DBO-S", "effluent bod": "DBO-S",
    "discharge bod": "DBO-S", "treated bod": "DBO-S", "final bod": "DBO-S", "effluent dbo": "DBO-S",
    "bod effluent": "DBO-S", "effluent bod5": "DBO-S", "outlet bod": "DBO-S",

    "dqo s": "DQO-S", "dqos": "DQO-S", "cod s": "DQO-S", "cods": "DQO-S", "effluent cod": "DQO-S",
    "discharge cod": "DQO-S", "treated cod": "DQO-S", "final cod": "DQO-S", "effluent dqo": "DQO-S",
    "cod effluent": "DQO-S", "outlet cod": "DQO-S",

    "ss s": "SS-S", "sss": "SS-S", "tss s": "SS-S", "effluent ss": "SS-S", "effluent suspended solids": "SS-S",
    "discharge ss": "SS-S", "treated ss": "SS-S", "final ss": "SS-S", "suspended solids effluent": "SS-S",
    "outlet ss": "SS-S", "effluent total suspended solids": "SS-S",

    "ssv s": "SSV-S", "ssvs": "SSV-S", "vss s": "SSV-S", "effluent ssv": "SSV-S", "effluent volatile solids": "SSV-S",
    "effluent volatile suspended solids": "SSV-S", "volatile suspended solids effluent": "SSV-S",

    "sed s": "SED-S", "seds": "SED-S", "effluent sed": "SED-S", "effluent settleable solids": "SED-S",
    "discharge sed": "SED-S", "treated settleable solids": "SED-S", "final sed": "SED-S",
    "settleable solids effluent": "SED-S", "outlet sed": "SED-S",

    "cond s": "COND-S", "conds": "COND-S", "ec s": "COND-S", "effluent cond": "COND-S",
    "effluent conductivity": "COND-S", "discharge cond": "COND-S", "treated conductivity": "COND-S",
    "final cond": "COND-S", "conductivity effluent": "COND-S", "effluent electrical conductivity": "COND-S",

    # Stage 5: Plant Removal Efficiencies (RD-)
    "rd dbo p": "RD-DBO-P", "rddbop": "RD-DBO-P", "rd bod p": "RD-DBO-P", "primary bod removal": "RD-DBO-P",
    "primary bod efficiency": "RD-DBO-P", "primary dbo removal": "RD-DBO-P", "bod removal primary": "RD-DBO-P",

    "rd ss p": "RD-SS-P", "rdssp": "RD-SS-P", "primary solids removal": "RD-SS-P", "primary ss removal": "RD-SS-P",
    "primary suspended solids efficiency": "RD-SS-P", "ss removal primary": "RD-SS-P",

    "rd sed p": "RD-SED-P", "rdsedp": "RD-SED-P", "primary settleable solids removal": "RD-SED-P",
    "primary sed removal": "RD-SED-P", "sed removal primary": "RD-SED-P",

    "rd dbo s": "RD-DBO-S", "rddbos": "RD-DBO-S", "rd bod s": "RD-DBO-S", "secondary bod removal": "RD-DBO-S",
    "secondary bod efficiency": "RD-DBO-S", "secondary dbo removal": "RD-DBO-S", "bod removal secondary": "RD-DBO-S",

    "rd dqo s": "RD-DQO-S", "rddqos": "RD-DQO-S", "rd cod s": "RD-DQO-S", "secondary cod removal": "RD-DQO-S",
    "secondary cod efficiency": "RD-DQO-S", "secondary dqo removal": "RD-DQO-S", "cod removal secondary": "RD-DQO-S",

    "rd dbo g": "RD-DBO-G", "rddbog": "RD-DBO-G", "rd bod g": "RD-DBO-G", "global bod removal": "RD-DBO-G",
    "total bod removal": "RD-DBO-G", "plant bod efficiency": "RD-DBO-G", "overall bod removal": "RD-DBO-G",
    "bod removal global": "RD-DBO-G",

    "rd dqo g": "RD-DQO-G", "rddqog": "RD-DQO-G", "rd cod g": "RD-DQO-G", "global cod removal": "RD-DQO-G",
    "total cod removal": "RD-DQO-G", "plant cod efficiency": "RD-DQO-G", "overall cod removal": "RD-DQO-G",
    "cod removal global": "RD-DQO-G",

    "rd ss g": "RD-SS-G", "rdssg": "RD-SS-G", "global solids removal": "RD-SS-G", "total solids removal": "RD-SS-G",
    "global ss removal": "RD-SS-G", "plant ss efficiency": "RD-SS-G", "overall ss removal": "RD-SS-G",
    "ss removal global": "RD-SS-G",

    "rd sed g": "RD-SED-G", "rdsedg": "RD-SED-G", "global settleable solids removal": "RD-SED-G",
    "total settleable solids removal": "RD-SED-G", "global sed removal": "RD-SED-G",
    "plant sed efficiency": "RD-SED-G", "overall sed removal": "RD-SED-G", "sed removal global": "RD-SED-G"
}

# Ambiguous Term Dictionary: Terms that lack treatment stage clarification
AMBIGUOUS_PATTERNS: Dict[str, List[str]] = {
    "ph": ["PH-E", "PH-P", "PH-D", "PH-S"],
    "ph value": ["PH-E", "PH-P", "PH-D", "PH-S"],
    "bod": ["DBO-E", "DBO-P", "DBO-D", "DBO-S"],
    "dbo": ["DBO-E", "DBO-P", "DBO-D", "DBO-S"],
    "bod5": ["DBO-E", "DBO-P", "DBO-D", "DBO-S"],
    "biological oxygen demand": ["DBO-E", "DBO-P", "DBO-D", "DBO-S"],
    "biochemical oxygen demand": ["DBO-E", "DBO-P", "DBO-D", "DBO-S"],
    "cod": ["DQO-E", "DQO-D", "DQO-S"],
    "dqo": ["DQO-E", "DQO-D", "DQO-S"],
    "chemical oxygen demand": ["DQO-E", "DQO-D", "DQO-S"],
    "ss": ["SS-E", "SS-P", "SS-D", "SS-S"],
    "tss": ["SS-E", "SS-P", "SS-D", "SS-S"],
    "suspended solids": ["SS-E", "SS-P", "SS-D", "SS-S"],
    "total suspended solids": ["SS-E", "SS-P", "SS-D", "SS-S"],
    "ssv": ["SSV-E", "SSV-P", "SSV-D", "SSV-S"],
    "vss": ["SSV-E", "SSV-P", "SSV-D", "SSV-S"],
    "volatile solids": ["SSV-E", "SSV-P", "SSV-D", "SSV-S"],
    "volatile suspended solids": ["SSV-E", "SSV-P", "SSV-D", "SSV-S"],
    "sed": ["SED-E", "SED-P", "SED-D", "SED-S"],
    "settleable solids": ["SED-E", "SED-P", "SED-D", "SED-S"],
    "cond": ["COND-E", "COND-P", "COND-D", "COND-S"],
    "conductivity": ["COND-E", "COND-P", "COND-D", "COND-S"],
    "electrical conductivity": ["COND-E", "COND-P", "COND-D", "COND-S"],
    "removal efficiency": [
        "RD-DBO-P", "RD-SS-P", "RD-SED-P", "RD-DBO-S", "RD-DQO-S",
        "RD-DBO-G", "RD-DQO-G", "RD-SS-G", "RD-SED-G"
    ],
    "bod removal": ["RD-DBO-P", "RD-DBO-S", "RD-DBO-G"],
    "solids removal": ["RD-SS-P", "RD-SS-G"],
    "settleable solids removal": ["RD-SED-P", "RD-SED-G"],
    "cod removal": ["RD-DQO-S", "RD-DQO-G"]
}


# =============================================================================
# 3. UNIT NORMALIZATION & SAFE CONVERSION LOGIC
# =============================================================================

def _normalize_unit(unit_str: Optional[str]) -> str:
    """Normalize unit string representation."""
    if not unit_str or not isinstance(unit_str, str):
        return ""
    u = unit_str.strip().lower()
    u = u.replace("³", "3").replace("^3", "3").replace("µ", "u").replace("μ", "u")
    u = re.sub(r"\s+", "", u)
    return u


def convert_measurement_unit(
    value: float,
    source_unit: Optional[str],
    canonical_feature: str
) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Apply mathematically sound unit conversion to match canonical feature unit.

    Returns:
        Tuple: (converted_value, unit_status, unit_warning_or_info)
        where unit_status is in {'valid', 'converted', 'assumed_default', 'incompatible', 'missing'}
    """
    if pd.isnull(value):
        return np.nan, "valid", None

    canonical_unit = CANONICAL_UNITS.get(canonical_feature, "")
    norm_source = _normalize_unit(source_unit)

    # 1. No unit provided: assume canonical default with explicit notice
    if not norm_source:
        return value, "assumed_default", f"No unit specified for {canonical_feature}. Assumed canonical ({canonical_unit})."

    # 2. Flow Rate (Q-E: canonical = m3/day)
    if canonical_feature == "Q-E":
        if norm_source in {"m3/day", "m3/d", "m3/24h", "cu.m/day"}:
            return value, "valid", None
        elif norm_source in {"m3/h", "m3/hr", "cu.m/hr"}:
            # m3/h * 24 = m3/day
            return value * 24.0, "converted", f"Converted Flow from m³/h to canonical m³/day (× 24)."
        elif norm_source in {"l/s", "liters/sec", "lps"}:
            # 1 L/s = 86.4 m3/day
            return value * 86.4, "converted", f"Converted Flow from L/s to canonical m³/day (× 86.4)."
        elif norm_source in {"mld", "ml/d"}:
            # 1 MLD (Million Litres per Day) = 1,000 m3/day
            return value * 1000.0, "converted", f"Converted Flow from MLD to canonical m³/day (× 1,000)."
        else:
            return None, "incompatible", f"Incompatible flow unit '{source_unit}' for Q-E. Expected volumetric flow (m³/day, m³/h, L/s, MLD)."

    # 3. Mass Concentrations (mg/L: BOD, COD, SS, Zinc)
    if canonical_unit == "mg/L":
        if norm_source in {"mg/l", "ppm", "g/m3", "mg/dm3"}:
            # ppm and g/m3 are physically 1:1 with mg/L in dilute aqueous water
            return value, "valid", None
        elif norm_source in {"g/l", "g/dm3"}:
            # g/L * 1000 = mg/L
            return value * 1000.0, "converted", f"Converted concentration from g/L to canonical mg/L (× 1,000)."
        elif norm_source in {"ug/l", "ug/dm3", "ppb"}:
            # ug/L / 1000 = mg/L
            return value / 1000.0, "converted", f"Converted concentration from µg/L to canonical mg/L (÷ 1,000)."
        else:
            return None, "incompatible", f"Incompatible concentration unit '{source_unit}' for {canonical_feature}. Expected mg/L, ppm, g/m³, g/L, or µg/L."

    # 4. Electrical Conductivity (µS/cm)
    if canonical_unit == "µS/cm":
        if norm_source in {"us/cm", "micros/cm", "umho/cm"}:
            return value, "valid", None
        elif norm_source in {"ms/cm", "millimho/cm"}:
            # 1 mS/cm = 1,000 µS/cm
            return value * 1000.0, "converted", f"Converted conductivity from mS/cm to canonical µS/cm (× 1,000)."
        elif norm_source in {"ms/m"}:
            # 1 mS/m = 10 µS/cm
            return value * 10.0, "converted", f"Converted conductivity from mS/m to canonical µS/cm (× 10)."
        else:
            return None, "incompatible", f"Incompatible conductivity unit '{source_unit}' for {canonical_feature}. Expected µS/cm, mS/cm, or mS/m."

    # 5. Settleable Solids (mL/L)
    if canonical_unit == "mL/L":
        if norm_source in {"ml/l", "ml/dm3", "cm3/l"}:
            return value, "valid", None
        elif norm_source in {"l/m3"}:
            # 1 L/m3 = 1 mL/L
            return value, "valid", None
        else:
            return None, "incompatible", f"Incompatible settleable solids unit '{source_unit}' for {canonical_feature}. Expected mL/L or L/m³."

    # 6. Percentages & Efficiencies (%: SSV-*, RD-*)
    if canonical_unit == "%":
        if norm_source in {"%", "percent", "pct"}:
            return value, "valid", None
        elif norm_source in {"fraction", "ratio"} and value <= 1.0:
            # Fraction to percentage
            return value * 100.0, "converted", f"Converted fractional ratio ({value}) to percentage ({value*100:.1f}%)."
        else:
            return value, "valid", None

    # 7. pH (Dimensionless)
    if canonical_unit == "pH units":
        if norm_source in {"ph", "phunits", "units", "-"}:
            return value, "valid", None
        else:
            return value, "assumed_default", f"Assumed pH units for '{source_unit}'."

    # Default fallback
    return value, "valid", None


# =============================================================================
# 4. INGESTION DATA STRUCTURES & REPORTING
# =============================================================================

@dataclass
class IngestionReport:
    """Comprehensive audit report for uploaded data."""
    file_type: str
    total_rows: int
    detected_canonical_count: int
    coverage_pct: float
    is_complete_38: bool
    is_ml_permitted: bool
    missing_features: List[str] = field(default_factory=list)
    missing_by_stage: Dict[str, List[str]] = field(default_factory=dict)
    mapped_features: Dict[str, str] = field(default_factory=dict) # canonical -> original
    ambiguous_features: List[Dict[str, Any]] = field(default_factory=list)
    unrecognised_features: List[str] = field(default_factory=list)
    unit_warnings: List[str] = field(default_factory=list)
    extraction_warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ExtractedData:
    """Extracted data payload containing raw, standardized data, and audit report."""
    raw_df: pd.DataFrame
    standardized_df: Optional[pd.DataFrame]
    report: IngestionReport


# =============================================================================
# 5. FEATURE MAPPING ENGINE
# =============================================================================

class FeatureMapper:
    """
    Intelligent feature recognition engine mapping arbitrary column headers
    and key-value names to MIZU's 38 canonical sensor features.
    """

    def __init__(self):
        self.canonical_set: Set[str] = set(CANONICAL_FEATURES)
        self.aliases: Dict[str, str] = FEATURE_ALIASES
        self.ambiguous_patterns: Dict[str, List[str]] = AMBIGUOUS_PATTERNS
        self.stage_groups: Dict[str, List[str]] = get_feature_groups()

    def map_columns(
        self,
        column_names: List[str]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], List[str]]:
        """
        Map a list of raw column names to canonical features.

        Returns:
            Tuple of:
              - mapped: Dict[canonical_feature -> original_column_name]
              - ambiguous: List of ambiguous column dictionaries
              - unrecognised: List of column names that could not be mapped
        """
        mapped: Dict[str, str] = {}
        ambiguous: List[Dict[str, Any]] = []
        unrecognised: List[str] = []

        # Track canonical features already assigned to prevent duplicates
        assigned_canonicals: Set[str] = set()

        for col in column_names:
            col_str = str(col).strip()
            # Preserve date columns separately
            clean_param = re.sub(r"[\(\[][^\)\]]*[\)\]]", "", col_str).strip()
            norm_col = _normalize_string(clean_param)
            if norm_col in {"date", "time", "timestamp", "datetime", "day", "date parsed"}:
                continue

            # 1. Exact match with canonical code (case-insensitive)
            upper_clean = clean_param.upper()
            if upper_clean in self.canonical_set:
                if upper_clean not in assigned_canonicals:
                    mapped[upper_clean] = col_str
                    assigned_canonicals.add(upper_clean)
                    continue

            # 2. Check alias dictionary
            if norm_col in self.aliases:
                canonical = self.aliases[norm_col]
                if canonical not in assigned_canonicals:
                    mapped[canonical] = col_str
                    assigned_canonicals.add(canonical)
                    continue

            # 3. Check ambiguous patterns (stage-less names)
            if norm_col in self.ambiguous_patterns:
                ambiguous.append({
                    "original_name": col_str,
                    "normalized_name": norm_col,
                    "candidate_canonicals": self.ambiguous_patterns[norm_col],
                    "reason": "Missing treatment stage specification (Influent, Primary, Secondary, Effluent)."
                })
                continue

            # 4. Partial / fuzzy stage detection fallback
            matched_canonical = self._fuzzy_match(norm_col)
            if matched_canonical:
                if matched_canonical not in assigned_canonicals:
                    mapped[matched_canonical] = col_str
                    assigned_canonicals.add(matched_canonical)
                    continue

            # If none matched, mark unrecognised
            unrecognised.append(col_str)

        return mapped, ambiguous, unrecognised

    def _fuzzy_match(self, norm_col: str) -> Optional[str]:
        """Secondary rule-based matcher for complex phrased column names using tokenized words."""
        words = set(norm_col.split())

        # Detect Stage
        is_influent = bool(words & {"influent", "raw", "intake", "entry", "inflow", "inlet"})
        is_primary = bool(words & {"primary", "settler 1", "sedimentation 1"} or "primary" in norm_col or "settler 1" in norm_col)
        is_secondary = bool(words & {"secondary", "aeration", "settler 2", "biological"} or "secondary" in norm_col or "settler 2" in norm_col)
        is_effluent = bool(words & {"effluent", "discharge", "treated", "final", "outlet"})
        is_global = bool(words & {"global", "total", "plant", "overall"})
        is_removal = bool(words & {"removal", "efficiency", "reduction", "elimination"} or "removal" in norm_col or "efficiency" in norm_col)

        # Detect Parameter using strict token matching
        has_flow = bool(words & {"flow", "flowrate", "inflow", "volume", "qe"} or "flow rate" in norm_col)
        has_zinc = bool(words & {"zinc", "zn", "zne"})
        has_ph = bool(words & {"ph", "phe", "php", "phd", "phs"})
        has_bod = bool(words & {"bod", "dbo", "bod5", "dbo5", "dboe", "dbop", "dbod", "dbos"} or "biochemical oxygen demand" in norm_col or "biological oxygen demand" in norm_col)
        has_cod = bool(words & {"cod", "dqo", "dqoe", "dqod", "dqos"} or "chemical oxygen demand" in norm_col)
        has_ss = bool(words & {"ss", "tss", "sse", "ssp", "ssd", "sss"} or "suspended solids" in norm_col)
        has_ssv = bool(words & {"ssv", "vss", "ssve", "ssvp", "ssvd", "ssvs"} or "volatile suspended solids" in norm_col or "volatile solids" in norm_col)
        has_sed = bool(words & {"sed", "sede", "sedp", "sedd", "seds"} or "settleable solids" in norm_col or "settleable" in words)
        # Whole-word check for conductivity so it does not match inside 'secondary'!
        has_cond = bool(words & {"conductivity", "cond", "ec", "conde", "condp", "condd", "conds"} or "electrical conductivity" in norm_col)

        if has_flow:
            return "Q-E"
        if has_zinc:
            return "ZN-E"

        if is_removal:
            if has_bod:
                if is_primary: return "RD-DBO-P"
                if is_secondary: return "RD-DBO-S"
                return "RD-DBO-G"
            if has_cod:
                if is_secondary: return "RD-DQO-S"
                return "RD-DQO-G"
            if has_ss:
                if is_primary: return "RD-SS-P"
                return "RD-SS-G"
            if has_sed:
                if is_primary: return "RD-SED-P"
                return "RD-SED-G"

        if has_ph:
            if is_influent: return "PH-E"
            if is_primary: return "PH-P"
            if is_secondary: return "PH-D"
            if is_effluent: return "PH-S"

        if has_bod:
            if is_influent: return "DBO-E"
            if is_primary: return "DBO-P"
            if is_secondary: return "DBO-D"
            if is_effluent: return "DBO-S"

        if has_cod:
            if is_influent: return "DQO-E"
            if is_secondary: return "DQO-D"
            if is_effluent: return "DQO-S"

        if has_ssv:
            if is_influent: return "SSV-E"
            if is_primary: return "SSV-P"
            if is_secondary: return "SSV-D"
            if is_effluent: return "SSV-S"

        if has_sed:
            if is_influent: return "SED-E"
            if is_primary: return "SED-P"
            if is_secondary: return "SED-D"
            if is_effluent: return "SED-S"

        if has_cond:
            if is_influent: return "COND-E"
            if is_primary: return "COND-P"
            if is_secondary: return "COND-D"
            if is_effluent: return "COND-S"

        if has_ss:
            if is_influent: return "SS-E"
            if is_primary: return "SS-P"
            if is_secondary: return "SS-D"
            if is_effluent: return "SS-S"

        return None


# =============================================================================
# 6. MULTI-FORMAT PARSERS
# =============================================================================

class MultiFormatIngestionEngine:
    """
    Main entrypoint for ingesting CSV, Excel, and PDF water-treatment data.
    """

    def __init__(self):
        self.mapper = FeatureMapper()
        self.stage_groups = get_feature_groups()

    def ingest_file(
        self,
        file_source: Union[str, Path, io.BytesIO],
        file_name: Optional[str] = None
    ) -> ExtractedData:
        """
        Automatically detect file type and extract data.
        """
        # Determine filename and extension
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            fname = file_name or path.name
            ext = path.suffix.lower()
            with open(path, "rb") as f:
                content_bytes = f.read()
        elif isinstance(file_source, io.BytesIO):
            fname = file_name or "uploaded_file"
            ext = Path(fname).suffix.lower()
            content_bytes = file_source.getvalue()
        elif hasattr(file_source, "read"):
            # Streamlit UploadedFile or file-like object
            fname = getattr(file_source, "name", file_name or "uploaded_file")
            ext = Path(fname).suffix.lower()
            content_bytes = file_source.read()
        else:
            raise ValueError(f"Unsupported file source type: {type(file_source)}")

        # Dispatch based on extension
        if ext in {".csv", ".txt"}:
            return self.parse_csv(content_bytes, fname)
        elif ext in {".xlsx", ".xls"}:
            return self.parse_excel(content_bytes, fname)
        elif ext == ".pdf":
            return self.parse_pdf(content_bytes, fname)
        else:
            report = IngestionReport(
                file_type="unknown",
                total_rows=0,
                detected_canonical_count=0,
                coverage_pct=0.0,
                is_complete_38=False,
                is_ml_permitted=False,
                error_message=f"Unsupported file extension '{ext}'. MIZU supports .csv, .xlsx, and .pdf."
            )
            return ExtractedData(raw_df=pd.DataFrame(), standardized_df=None, report=report)

    # -------------------------------------------------------------------------
    # CSV Parser
    # -------------------------------------------------------------------------
    def parse_csv(self, content_bytes: bytes, filename: str) -> ExtractedData:
        """Parse CSV file with separator sniffing and header cleaning."""
        try:
            # Detect separator
            text_sample = content_bytes[:4096].decode("utf-8", errors="ignore")
            sep = ","
            if "\t" in text_sample and text_sample.count("\t") > text_sample.count(","):
                sep = "\t"
            elif ";" in text_sample and text_sample.count(";") > text_sample.count(","):
                sep = ";"

            df = pd.read_csv(io.BytesIO(content_bytes), sep=sep, na_values=["?", "NA", "null", ""])
            df = df.dropna(how="all").reset_index(drop=True)
            return self._build_standardized_payload(df, "csv", filename)
        except Exception as e:
            report = IngestionReport(
                file_type="csv",
                total_rows=0,
                detected_canonical_count=0,
                coverage_pct=0.0,
                is_complete_38=False,
                is_ml_permitted=False,
                error_message=f"Failed to parse CSV file: {str(e)}"
            )
            return ExtractedData(raw_df=pd.DataFrame(), standardized_df=None, report=report)

    # -------------------------------------------------------------------------
    # Excel Parser
    # -------------------------------------------------------------------------
    def parse_excel(self, content_bytes: bytes, filename: str) -> ExtractedData:
        """Parse Excel (.xlsx, .xls) sheets, supporting horizontal and vertical layouts."""
        try:
            # Read first sheet
            excel_file = io.BytesIO(content_bytes)
            df = pd.read_excel(excel_file, sheet_name=0, na_values=["?", "NA", "null", ""])
            df = df.dropna(how="all").reset_index(drop=True)

            # Check if Excel is vertically formatted:
            # (e.g. Column 0 = Parameter Name, Column 1 = Value, Column 2 = Unit)
            if len(df.columns) in {2, 3} and len(df) > 5:
                col0_sample = [str(x) for x in df.iloc[:5, 0]]
                # If column 0 looks like parameter names rather than dates/IDs
                mapped_sample, _, _ = self.mapper.map_columns(col0_sample)
                if len(mapped_sample) >= 2:
                    df = self._transpose_vertical_table(df)

            return self._build_standardized_payload(df, "excel", filename)
        except Exception as e:
            report = IngestionReport(
                file_type="excel",
                total_rows=0,
                detected_canonical_count=0,
                coverage_pct=0.0,
                is_complete_38=False,
                is_ml_permitted=False,
                error_message=f"Failed to parse Excel file: {str(e)}"
            )
            return ExtractedData(raw_df=pd.DataFrame(), standardized_df=None, report=report)

    def _transpose_vertical_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transpose vertical Key-Value-Unit table into horizontal single-observation row."""
        param_col = df.columns[0]
        val_col = df.columns[1]
        has_unit = len(df.columns) >= 3
        unit_col = df.columns[2] if has_unit else None

        row_dict = {}
        for _, row in df.iterrows():
            param = str(row[param_col]).strip()
            val = row[val_col]
            if has_unit and pd.notnull(row[unit_col]):
                unit_str = str(row[unit_col]).strip()
                if not re.search(r"[\(\[]", param):
                    col_header = f"{param} ({unit_str})"
                else:
                    col_header = param
            else:
                col_header = param
            row_dict[col_header] = [val]

        return pd.DataFrame(row_dict)

    # -------------------------------------------------------------------------
    # PDF Parser (Table & Key-Value Text & Scanned Detection)
    # -------------------------------------------------------------------------
    def parse_pdf(self, content_bytes: bytes, filename: str) -> ExtractedData:
        """
        Parse PDF document:
        1. Attempt table extraction via pdfplumber.
        2. Attempt key-value text extraction via regex.
        3. Detect scanned / raster image PDFs and return graceful degradation message.
        """
        try:
            import pdfplumber
        except ImportError:
            report = IngestionReport(
                file_type="pdf",
                total_rows=0,
                detected_canonical_count=0,
                coverage_pct=0.0,
                is_complete_38=False,
                is_ml_permitted=False,
                error_message="pdfplumber package is required for PDF extraction."
            )
            return ExtractedData(raw_df=pd.DataFrame(), standardized_df=None, report=report)

        try:
            pdf_file = io.BytesIO(content_bytes)
            all_text = ""
            extracted_tables = []
            has_images = False

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    # Check text
                    t = page.extract_text() or ""
                    all_text += t + "\n"

                    # Check tables
                    tables = page.extract_tables()
                    if tables:
                        extracted_tables.extend(tables)

                    # Check images
                    if getattr(page, "images", None):
                        has_images = True

            clean_text = all_text.strip()

            # 1. Scanned PDF Detection (Zero text extracted, but images present)
            if len(clean_text) == 0 and len(extracted_tables) == 0:
                report = IngestionReport(
                    file_type="scanned_pdf",
                    total_rows=0,
                    detected_canonical_count=0,
                    coverage_pct=0.0,
                    is_complete_38=False,
                    is_ml_permitted=False,
                    error_message=(
                        "⚠️ **Scanned PDF Detected (OCR Unavailable)**: "
                        "This PDF contains raster image scans with zero digital text. "
                        "System OCR dependencies (Tesseract) are currently unavailable in this environment. "
                        "Please export or upload a digital text-based PDF, CSV, or Excel file."
                    )
                )
                return ExtractedData(raw_df=pd.DataFrame(), standardized_df=None, report=report)

            # 2. Attempt Table Extraction
            if extracted_tables:
                df_table = self._convert_pdf_tables_to_df(extracted_tables)
                if df_table is not None and not df_table.empty:
                    # Check if table contains canonical features
                    mapped, _, _ = self.mapper.map_columns(list(df_table.columns))
                    if len(mapped) >= 2:
                        return self._build_standardized_payload(df_table, "pdf_table", filename)

            # 3. Attempt Key-Value Text Extraction (Laboratory narrative reports)
            df_kv, kv_warnings = self._extract_key_value_measurements(clean_text)
            if df_kv is not None and not df_kv.empty:
                return self._build_standardized_payload(
                    df_kv, "pdf_key_value", filename, extra_warnings=kv_warnings
                )

            # Fallback if text present but unparseable
            report = IngestionReport(
                file_type="pdf",
                total_rows=0,
                detected_canonical_count=0,
                coverage_pct=0.0,
                is_complete_38=False,
                is_ml_permitted=False,
                error_message=(
                    "PDF extracted digital text, but no structured sensor tables or key-value "
                    "measurements matching wastewater parameters were recognised."
                )
            )
            return ExtractedData(raw_df=pd.DataFrame(), standardized_df=None, report=report)

        except Exception as e:
            report = IngestionReport(
                file_type="pdf",
                total_rows=0,
                detected_canonical_count=0,
                coverage_pct=0.0,
                is_complete_38=False,
                is_ml_permitted=False,
                error_message=f"Failed to process PDF: {str(e)}"
            )
            return ExtractedData(raw_df=pd.DataFrame(), standardized_df=None, report=report)

    def _convert_pdf_tables_to_df(self, tables: List[List[List[Optional[str]]]]) -> Optional[pd.DataFrame]:
        """Convert list of 2D string tables from pdfplumber into clean DataFrame."""
        for t in tables:
            if not t or len(t) < 2:
                continue
            # First row as header
            raw_header = [str(c).strip() if c else f"Col_{i}" for i, c in enumerate(t[0])]
            data_rows = t[1:]
            clean_rows = []
            for r in data_rows:
                # pad or trim
                row = [str(c).strip() if c is not None else "" for c in r]
                if len(row) < len(raw_header):
                    row.extend([""] * (len(raw_header) - len(row)))
                clean_rows.append(row[:len(raw_header)])

            df = pd.DataFrame(clean_rows, columns=raw_header)
            df = df.dropna(how="all").reset_index(drop=True)
            if not df.empty:
                return df
        return None

    def _extract_key_value_measurements(self, text: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Extract parameters, values, and units from narrative PDF reports using regex."""
        warnings = []
        measurements: Dict[str, Any] = {}

        # Look for Date
        date_match = re.search(r"(?:Date|Sample Date|Report Date|Collected)\s*[:=]\s*([A-Za-z0-9\-/]+)", text, re.IGNORECASE)
        if date_match:
            measurements["Date"] = date_match.group(1).strip()

        # Regex matching: Parameter Name : Value [Unit]
        # e.g., "Influent Flow Rate: 42393 m3/day"
        # e.g., "Effluent Settleable Solids = 3.5 mL/L"
        # e.g., "pH (Primary Settler): 7.8"
        pattern = re.compile(
            r"^\s*([A-Za-z0-9\s\-_\./\(\)]+?)\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+)\s*([A-Za-z0-9%/³µμ\^]+)?",
            re.MULTILINE
        )

        for match in pattern.finditer(text):
            param = match.group(1).strip()
            val_str = match.group(2).strip()
            unit_str = match.group(3).strip() if match.group(3) else ""

            norm_param = _normalize_string(param)
            # Skip common non-sensor headers
            if norm_param in {"page", "sample id", "analyst", "method", "tel", "phone", "version"}:
                continue

            try:
                val = float(val_str)
                if unit_str and not re.search(r"[\(\[]", param):
                    col_name = f"{param} ({unit_str})"
                else:
                    col_name = param
                measurements[col_name] = val
            except ValueError:
                continue

        if measurements:
            # Wrap as single row dataframe
            df = pd.DataFrame([measurements])
            return df, warnings

        return None, warnings

    # -------------------------------------------------------------------------
    # Payload Assembly & Strict Abstention Contract
    # -------------------------------------------------------------------------
    def _build_standardized_payload(
        self,
        raw_df: pd.DataFrame,
        file_type: str,
        filename: str,
        extra_warnings: Optional[List[str]] = None,
        manual_overrides: Optional[Dict[str, str]] = None
    ) -> ExtractedData:
        """
        Validate feature mapping, apply unit conversions, compute completeness,
        and enforce strict abstention for ML predictions.
        Supports manual_overrides mapping original_column -> canonical_code.
        """
        warnings = list(extra_warnings or [])
        unit_warnings: List[str] = []

        # Map columns
        col_list = list(raw_df.columns)
        mapped, ambiguous, unrecognised = self.mapper.map_columns(col_list)

        # Apply manual user confirmations/overrides if provided
        if manual_overrides:
            for orig_col, canonical in manual_overrides.items():
                if orig_col in col_list and canonical in CANONICAL_FEATURES:
                    mapped[canonical] = orig_col
                    # Remove from ambiguous if resolved
                    ambiguous = [a for a in ambiguous if a["original_name"] != orig_col]
                    # Remove from unrecognised if resolved
                    unrecognised = [u for u in unrecognised if u != orig_col]

        # Extract units from headers (e.g., "Q-E (m3/h)" -> unit = "m3/h")
        col_units: Dict[str, str] = {}
        for col in col_list:
            unit_match = re.search(r"\(([^)]+)\)|\[([^\]]+)\]", str(col))
            if unit_match:
                extracted_unit = unit_match.group(1) or unit_match.group(2)
                col_units[col] = extracted_unit.strip()

        # Standardize DataFrame
        standardized_records = []
        for idx, row in raw_df.iterrows():
            record = {}
            for canonical, orig_col in mapped.items():
                val = row[orig_col]
                # Coerce to float
                try:
                    val_float = float(val) if pd.notnull(val) else np.nan
                except (ValueError, TypeError):
                    val_float = np.nan

                # Detect unit
                source_unit = col_units.get(orig_col, None)
                converted_val, u_status, u_msg = convert_measurement_unit(val_float, source_unit, canonical)

                if u_msg and idx == 0:
                    unit_warnings.append(u_msg)

                if u_status == "incompatible":
                    record[canonical] = np.nan
                else:
                    record[canonical] = converted_val

            standardized_records.append(record)

        standardized_df = pd.DataFrame(standardized_records)

        # Preserve Date column if present
        date_col = next((c for c in raw_df.columns if _normalize_string(str(c)) in {"date", "time", "timestamp", "datetime", "day"}), None)
        if date_col:
            standardized_df["Date"] = raw_df[date_col].values
        else:
            standardized_df["Date"] = [f"Record_{i+1:04d}" for i in range(len(raw_df))]

        # Completeness Audit
        detected_canonicals = set(mapped.keys())
        missing_canonicals = [c for c in CANONICAL_FEATURES if c not in detected_canonicals]
        detected_count = len(detected_canonicals)
        coverage_pct = round((detected_count / len(CANONICAL_FEATURES)) * 100, 2)

        # Group missing by treatment stage
        missing_by_stage: Dict[str, List[str]] = {}
        for stage_name, stage_cols in self.stage_groups.items():
            missing_in_stage = [c for c in stage_cols if c in missing_canonicals]
            if missing_in_stage:
                missing_by_stage[stage_name] = missing_in_stage

        # Strict Abstention: ML allowed ONLY if exactly all 38 continuous features are present
        is_complete_38 = (detected_count == len(CANONICAL_FEATURES))
        is_ml_permitted = is_complete_38

        # Ensure canonical column order when complete
        if is_complete_38:
            ordered_cols = ["Date"] + CANONICAL_FEATURES
            standardized_df = standardized_df[ordered_cols]

        report = IngestionReport(
            file_type=file_type,
            total_rows=len(raw_df),
            detected_canonical_count=detected_count,
            coverage_pct=coverage_pct,
            is_complete_38=is_complete_38,
            is_ml_permitted=is_ml_permitted,
            missing_features=missing_canonicals,
            missing_by_stage=missing_by_stage,
            mapped_features=mapped,
            ambiguous_features=ambiguous,
            unrecognised_features=unrecognised,
            unit_warnings=unit_warnings,
            extraction_warnings=warnings
        )

        return ExtractedData(raw_df=raw_df, standardized_df=standardized_df, report=report)

    def apply_user_mappings(
        self,
        extracted: ExtractedData,
        user_mappings: Dict[str, str],
        filename: str = "uploaded_file"
    ) -> ExtractedData:
        """
        Re-standardize data using user-confirmed manual mappings for ambiguous or unmapped columns.

        Args:
            extracted: Existing ExtractedData instance.
            user_mappings: Dict mapping original_column_name -> canonical_feature_code.
            filename: Document identifier.

        Returns:
            ExtractedData: Updated payload with recomputed completeness and report.
        """
        return self._build_standardized_payload(
            raw_df=extracted.raw_df,
            file_type=extracted.report.file_type,
            filename=filename,
            extra_warnings=extracted.report.extraction_warnings,
            manual_overrides=user_mappings
        )

