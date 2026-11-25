"""Documentation text taken from the MIZU documentation guide PDF."""

WHAT_MIZU_DOES = (
    "MIZU is a research and decision-support framework for wastewater-treatment telemetry. "
    "It analyses complete sensor records from the treatment train, identifies unusual operating "
    "patterns without requiring anomaly labels, and shows the measurements that contributed to a flag. "
    "It does not create measurements, replace certified testing, or control plant equipment."
)

WORKFLOW = [
    ("1. Read and standardise input",
     "MIZU accepts delimited text, spreadsheets, and supported digital PDF reports. It matches recognised "
     "aliases to the canonical sensor names and applies only documented unit conversions."),
    ("2. Require the complete feature contract",
     "The fitted models require all 38 canonical physicochemical and performance features, spanning influent, "
     "primary, secondary, effluent, and removal-efficiency measurements. Ambiguous or incomplete input is shown "
     "for review but is not scored."),
    ("3. Transform without retraining",
     "For a complete record, the fitted preprocessing pipeline handles numeric missing values within existing "
     "features and applies RobustScaler. Uploading data does not refit the imputer, scaler, thresholds, or models."),
    ("4. Detect anomalies with two models",
     "Isolation Forest identifies readings that isolate quickly across the full sensor space. Novelty Local "
     "Outlier Factor identifies readings with lower local density than nearby baseline operating conditions."),
    ("5. Describe operating context",
     "K-Means assigns an operating regime. Principal Component Analysis projects the 38-dimensional record into a "
     "two-dimensional exploration view and reports the retained variance; it does not make the anomaly decision."),
    ("6. Explain and verify",
     "Feature ablation measures how replacing one selected sensor value with its baseline reference changes the "
     "anomaly score. This supports investigation, which still requires plant records, samples, and qualified "
     "engineering judgement."),
]

HOW_TO_READ = [
    ("Overview", "Shows the latest status, totals, score timelines, and the current plant summary."),
    ("Flagged Readings", "Lists events needing review."),
    ("Patterns", "Shows the operational-regime map."),
    ("Why Flagged", "Shows sensor-level contribution for a selected event."),
    ("Upload Data", "Shows the input audit, confirms whether scoring is permitted, and provides a download of the standardised input as CSV."),
]

HOW_TO_INTERPRET = [
    ("Flagged by both models", "The strongest agreement signal."),
    ("Flagged by one model", "Still worth review, but represents weaker agreement."),
    ("Normal", "Only means the record resembles the training baseline; it is not a regulatory pass/fail decision."),
]

LIMITS = (
    "MIZU is designed for research and operational screening of wastewater-treatment telemetry. It does not certify "
    "drinking-water safety, detect microbiological pathogens directly, establish a causal diagnosis, or send commands "
    "to equipment. A result represents mathematical deviation from the fitted baseline, not a confirmed plant fault. "
    "Any operational action remains under qualified human oversight and should be verified with plant records and "
    "certified laboratory testing."
)
