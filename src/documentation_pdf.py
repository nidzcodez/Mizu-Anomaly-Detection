"""Build the downloadable MIZU documentation guide."""

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# The downloadable guide is intentionally print-friendly: black text, white
# paper, and grey rules only.
INK = colors.black
MUTED = colors.HexColor("#404040")
PAPER = colors.white
BORDER = colors.HexColor("#8A8A8A")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("MizuTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=21, leading=26, textColor=INK, spaceAfter=5),
        "subtitle": ParagraphStyle("MizuSubtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10, leading=15, textColor=MUTED, spaceAfter=18),
        "heading": ParagraphStyle("MizuHeading", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=17, textColor=INK, spaceBefore=14, spaceAfter=7),
        "body": ParagraphStyle("MizuBody", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=14, textColor=INK, spaceAfter=7),
        "small": ParagraphStyle("MizuSmall", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=MUTED),
        "step": ParagraphStyle("MizuStep", parent=base["BodyText"], fontName="Helvetica", fontSize=8.5, leading=12, textColor=INK),
    }


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 8 * mm, "MIZU | Water systems monitoring")
    canvas.drawRightString(192 * mm, 8 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_documentation_pdf() -> bytes:
    """Return a polished, static PDF guide for the in-app download control."""
    stream = BytesIO()
    doc = SimpleDocTemplate(
        stream, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=20 * mm,
        title="MIZU Documentation Guide",
        author="MIZU",
    )
    s = _styles()
    story = [
        Paragraph("MIZU", s["title"]),
        Paragraph("Documentation guide | Explainable unsupervised monitoring for wastewater treatment plants", s["subtitle"]),
        Paragraph("What MIZU does", s["heading"]),
        Paragraph(
            "MIZU is a research and decision-support framework for wastewater-treatment telemetry. It analyses complete sensor records from the treatment train, identifies unusual operating patterns without requiring anomaly labels, and shows the measurements that contributed to a flag. It does not create measurements, replace certified testing, or control plant equipment.",
            s["body"],
        ),
        Paragraph("End-to-end workflow", s["heading"]),
    ]

    steps = [
        ("1. Read and standardise input", "MIZU accepts delimited text, spreadsheets, and supported digital PDF reports. It matches recognised aliases to the canonical sensor names and applies only documented unit conversions."),
        ("2. Require the complete feature contract", "The fitted models require all 38 canonical physicochemical and performance features, spanning influent, primary, secondary, effluent, and removal-efficiency measurements. Ambiguous or incomplete input is shown for review but is not scored."),
        ("3. Transform without retraining", "For a complete record, the fitted preprocessing pipeline handles numeric missing values within existing features and applies RobustScaler. Uploading data does not refit the imputer, scaler, thresholds, or models."),
        ("4. Detect anomalies with two models", "Isolation Forest identifies readings that isolate quickly across the full sensor space. Novelty Local Outlier Factor identifies readings with lower local density than nearby baseline operating conditions."),
        ("5. Describe operating context", "K-Means assigns an operating regime. Principal Component Analysis projects the 38-dimensional record into a two-dimensional exploration view and reports the retained variance; it does not make the anomaly decision."),
        ("6. Explain and verify", "Feature ablation measures how replacing one selected sensor value with its baseline reference changes the anomaly score. This supports investigation, which still requires plant records, samples, and qualified engineering judgement."),
    ]
    rows = [[Paragraph(f"<b>{title}</b><br/>{body}", s["step"])] for title, body in steps]
    flow = Table(rows, colWidths=[156 * mm], hAlign="LEFT")
    flow.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PAPER),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([flow, Spacer(1, 8)])

    story.extend([
        Paragraph("How to read and export results", s["heading"]),
        Paragraph("Overview shows the latest status, totals, score timelines, and the current plant summary. Flagged Readings lists events needing review. Patterns shows the operational-regime map. Why Flagged shows sensor-level contribution for a selected event. Upload Data shows the input audit, confirms whether scoring is permitted, and provides a download of the standardised input as CSV. This guide can be downloaded as a PDF for a concise description of the analysis workflow; it is not a generated incident report.", s["body"]),
        Paragraph("How to interpret a flag", s["heading"]),
        Paragraph("A result flagged by both models is the strongest agreement signal. A result flagged by one model is still worth review, but represents weaker agreement. A normal result only means the record resembles the training baseline; it is not a regulatory pass/fail decision.", s["body"]),
        Paragraph("Operational limits", s["heading"]),
        Paragraph("MIZU is designed for research and operational screening of wastewater-treatment telemetry. It does not certify drinking-water safety, detect microbiological pathogens directly, establish a causal diagnosis, or send commands to equipment. A result represents mathematical deviation from the fitted baseline, not a confirmed plant fault. Any operational action remains under qualified human oversight and should be verified with plant records and certified laboratory testing.", s["body"]),
    ])
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return stream.getvalue()
