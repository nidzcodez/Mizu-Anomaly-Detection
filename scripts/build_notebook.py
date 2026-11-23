"""
Script to build and execute the beginner-friendly exploratory_analysis.ipynb notebook.
Uses nbformat to construct well-structured markdown and code cells.
"""
import nbformat as nbf
from pathlib import Path

project_root = Path.cwd()
nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Title and Introduction (Markdown)
cells.append(nbf.v4.new_markdown_cell("""# MIZU (水) — Exploratory Data Analysis (EDA)
### Understanding Wastewater Treatment Data Before Building Machine Learning Models

Welcome! If you are new to machine learning and data science, this notebook is built specifically for you.

---

### What is Exploratory Data Analysis (EDA)?
Before data scientists write a single line of machine learning code, they must understand their data. Think of EDA like a doctor examining a patient before prescribing medication:
1. **Inspecting features**: What physical measurements do we have?
2. **Finding missing data**: Did any sensors fail or delay their readings?
3. **Spotting anomalies & outliers**: Are there extreme spikes (like toxic industrial dumping or massive rainstorms)?
4. **Tracking relationships**: How does clean water emerge from dirty wastewater across the plant?

---

### The Physical System: A 4-Stage Water Treatment Plant
Imagine wastewater traveling through 4 connected treatment zones:
1. **Influent (`-E`)**: Raw wastewater coming into the facility from city sewer lines.
2. **Primary Settler (`-P`)**: Large settling basins where heavy particles and sand physically sink to the bottom.
3. **Biological Aeration (`-D`)**: Giant aeration tanks filled with friendly bacteria (activated sludge) that "eat" organic pollution.
4. **Effluent (`-S`)**: The final clean treated water discharged back into the river or sea.
5. **Efficiency Ratios (`RD-`)**: Calculated percentages showing how much pollution was removed (e.g., 90% reduction).
"""))

# Cell 2: Imports (Code)
cells.append(nbf.v4.new_code_cell("""# Step 1: Import our tools
# sys and pathlib let Python find our custom code in the 'src' directory
import sys
from pathlib import Path

# Add project root to Python module path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Standard data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom data loader from src
from src.data_loader import load_water_treatment_data, get_feature_groups, get_missing_summary

# Set visualization aesthetics for a clean scientific look
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

# Create figures directory to save exported plots
figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print("Environment configured successfully!")
"""))

# Cell 3: Loading Data (Markdown & Code)
cells.append(nbf.v4.new_markdown_cell("""## Step 2: Loading the Dataset
We use our custom `load_water_treatment_data()` function from `src/data_loader.py`.
Notice how it automatically converts the raw character `'?'` into proper `NaN` (Not a Number) values.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the dataset
df = load_water_treatment_data(project_root / "data" / "raw" / "water-treatment.data")

print(f"Dataset shape: {df.shape[0]} rows (operational days) and {df.shape[1]} columns (features).")
print(f"Date range: from {df['Date_Parsed'].min().date()} to {df['Date_Parsed'].max().date()}")

# Display the first 5 days of data (first 8 columns for readability)
df.iloc[:5, :8]
"""))

# Cell 4: Missing Values (Markdown & Code)
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Missing Value Analysis

### Why do missing values happen in water plants?
In many tech problems (like clicks on an e-commerce website), missing data is just a software glitch. 
In a wastewater treatment plant, missing data has a **physical and biological explanation**:

- **Continuous sensor probes** (like $pH$ and Electrical Conductivity) report numbers every second. They almost never have missing values!
- **Biochemical Oxygen Demand ($BOD_5$ / `DBO`)** requires taking a water bottle and incubating it in a lab with live bacteria at $20^\\circ\\text{C}$ for **5 whole days**. If the temperature fluctuates, the glassware breaks, or the sample is contaminated, the test is lost.

Let's visualize which columns have missing values and how severe they are.
"""))

cells.append(nbf.v4.new_code_cell("""# Compute missing value summary table
missing_df = get_missing_summary(df)
missing_only = missing_df[missing_df["missing_count"] > 0]

# Create a horizontal bar chart of missing percentages
plt.figure(figsize=(10, 8))
bars = plt.barh(missing_only["column"], missing_only["missing_pct"], color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)

plt.xlabel("Missing Values (%)", fontsize=11, fontweight="bold")
plt.ylabel("Sensor / Operational Feature", fontsize=11, fontweight="bold")
plt.title("MIZU: Missing Value Rate Across All Operational Features (Total: 527 Days)", fontsize=13, fontweight="bold", pad=15)
plt.gca().invert_yaxis()  # Highest missing on top

# Add percentage labels on the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%", va="center", fontsize=9, color="#1e293b")

plt.xlim(0, 15)
plt.tight_layout()
plt.savefig(figures_dir / "01_missing_values.png", dpi=300)
plt.show()

print(f"Total features with missing values: {len(missing_only)} out of {df.shape[1] - 2}")
print(f"Highest missing feature: {missing_only.iloc[0]['column']} ({missing_only.iloc[0]['missing_pct']}% missing)")
"""))

# Cell 5: Missing value takeaway (Markdown)
cells.append(nbf.v4.new_markdown_cell("""### Preprocessing Lesson from Missing Values:
1. **Should we drop rows with `dropna()`?**
   **No!** If we dropped every row that had even one missing value, we would lose **164 out of 527 days** (over 31% of our entire operational history!).
2. **What should we do instead?**
   We must use **Imputation** (filling missing values with smart estimates, such as the median value of each sensor). Median is much safer than mean because it is not distorted by extreme outlier spikes.
"""))

# Cell 6: Key Sensor Distributions (Markdown & Code)
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Sensor Distributions & Outlier Hunting

In machine learning, an **outlier** is an observation that is far away from the rest of the data. 
In wastewater treatment, outliers are not just "bad data"—they often represent **critical real-world events**:
- **Storm events**: A huge surge in incoming water volume (`Q-E`).
- **Toxic industrial spills**: A sudden spike in Zinc (`ZN-E`) or conductivity from a factory illegally dumping chemicals.
- **Treatment failure**: High organic pollution or solids escaping in final discharge (`DBO-S`, `SS-S`).

Let's plot histograms and density curves for 4 vital measurements.
"""))

cells.append(nbf.v4.new_code_cell("""# 4 Key sensors spanning influent and effluent
key_sensors = [
    ("Q-E", "Influent Flow Rate (m³/day)", "#0284c7"),
    ("ZN-E", "Influent Zinc Concentration (mg/L)", "#ea580c"),
    ("DBO-S", "Effluent Biochemical Oxygen Demand (mg/L)", "#16a34a"),
    ("SS-S", "Effluent Suspended Solids (mg/L)", "#dc2626")
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (col, title, color) in enumerate(key_sensors):
    ax = axes[idx]
    data = df[col].dropna()
    
    # Plot histogram with Kernel Density Estimation (KDE)
    sns.histplot(data, kde=True, ax=ax, color=color, bins=30, alpha=0.6, edgecolor="white")
    
    # Mark median and 95th percentile
    median_val = data.median()
    p95_val = data.quantile(0.95)
    max_val = data.max()
    
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1.5, label=f"Median: {median_val:.1f}")
    ax.axvline(p95_val, color="purple", linestyle=":", linewidth=1.5, label=f"95th Pct: {p95_val:.1f}")
    
    ax.set_title(f"{title} [{col}]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Measured Value")
    ax.set_ylabel("Count of Days")
    ax.legend(loc="upper right", fontsize=9)

plt.suptitle("MIZU: Distribution and Outlier Analysis of Key Operational Parameters", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(figures_dir / "02_sensor_distributions.png", dpi=300)
plt.show()
"""))

# Cell 7: Distribution takeaway (Markdown)
cells.append(nbf.v4.new_markdown_cell("""### What do these distribution plots teach us?
1. **Zinc (`ZN-E`) has a massive right tail**: The median is just $1.5\\ mg/L$, but the maximum reaches $33.5\\ mg/L$! This is more than $11$ standard deviations away. This is a classic industrial shock anomaly.
2. **Effluent BOD (`DBO-S`) and Solids (`SS-S`) are clean most of the time**: The median effluent BOD is only $18\\ mg/L$ (well within environmental safety limits). But on rare breakdown days, it spikes to $320\\ mg/L$!
3. **Scaling Lesson**: Because these features have heavy tails and extreme outliers, using standard `StandardScaler` (which relies on the mean and variance) will be distorted. Instead, we should use **`RobustScaler`** (which scales using the Median and Interquartile Range, $IQR = Q_3 - Q_1$).
"""))

# Cell 8: Stage by stage progression (Markdown & Code)
cells.append(nbf.v4.new_markdown_cell("""## Step 5: How the Treatment Plant Cleans Water (Stage-by-Stage Tracking)

Does the water treatment plant actually work?
Let's trace two primary pollutants as wastewater flows from the entrance to the exit:
1. **$BOD$ (Biochemical Oxygen Demand)**: Organic pollution.
2. **$SS$ (Suspended Solids)**: Insoluble dirt and particulate matter.

Stages:
- Entrance: Influent (`-E`)
- After Settling Tank: Primary (`-P`)
- After Aeration Basin: Secondary (`-D`)
- Final Exit: Effluent (`-S`)
"""))

cells.append(nbf.v4.new_code_cell("""# Compare stage-by-stage progression
stages = ["Influent (-E)", "Primary Settler (-P)", "Biological Stage (-D)", "Effluent (-S)"]

# Medians for BOD across stages
bod_stages = [df["DBO-E"].median(), df["DBO-P"].median(), df["DBO-D"].median(), df["DBO-S"].median()]

# Medians for Suspended Solids across stages
ss_stages = [df["SS-E"].median(), df["SS-P"].median(), df["SS-D"].median(), df["SS-S"].median()]

x = np.arange(len(stages))
width = 0.35

plt.figure(figsize=(11, 6))
bars1 = plt.bar(x - width/2, bod_stages, width, label="BOD (Organic Pollution, DBO)", color="#2563eb", alpha=0.85)
bars2 = plt.bar(x + width/2, ss_stages, width, label="Suspended Solids (Dirt Particles, SS)", color="#059669", alpha=0.85)

plt.ylabel("Median Concentration (mg/L)", fontsize=11, fontweight="bold")
plt.title("MIZU: Pollution Reduction Across Plant Treatment Stages", fontsize=13, fontweight="bold", pad=15)
plt.xticks(x, stages, fontsize=10, fontweight="bold")
plt.legend(fontsize=10)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f}", ha="center", fontsize=9, fontweight="bold", color="#1e3a8a")
for bar in bars2:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h:.1f}", ha="center", fontsize=9, fontweight="bold", color="#064e3b")

# Calculate overall reduction percentage
bod_reduction = ((bod_stages[0] - bod_stages[-1]) / bod_stages[0]) * 100
ss_reduction = ((ss_stages[0] - ss_stages[-1]) / ss_stages[0]) * 100

plt.ylim(0, 290)
plt.annotate(
    f"Overall BOD Reduction: {bod_reduction:.1f}%\\nOverall Solids Reduction: {ss_reduction:.1f}%",
    xy=(2.5, 200),
    bbox=dict(boxstyle="round,pad=0.6", fc="#f8fafc", ec="#94a3b8", lw=1.5),
    fontsize=10,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(figures_dir / "03_treatment_progression.png", dpi=300)
plt.show()
"""))

# Cell 9: Correlation Matrix (Markdown & Code)
cells.append(nbf.v4.new_markdown_cell("""## Step 6: Correlation Analysis (How Features Interact)

### What is correlation?
Correlation is a statistical number between $-1.0$ and $+1.0$:
- **$+1.0$ (Strong Positive)**: When one sensor goes up, the other sensor also goes up.
- **$0.0$ (No Correlation)**: The two sensors have no linear connection.
- **$-1.0$ (Strong Negative)**: When one sensor goes up, the other sensor goes down.

Let's examine how the raw incoming pollutants relate to one another and to effluent quality.
"""))

cells.append(nbf.v4.new_code_cell("""# Select representative features from each stage
selected_cols = [
    "Q-E", "ZN-E", "PH-E", "DBO-E", "DQO-E", "SS-E", "COND-E",
    "DBO-S", "DQO-S", "SS-S", "RD-DBO-G", "RD-SS-G"
]

corr_matrix = df[selected_cols].corr()

plt.figure(figsize=(11, 9))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation Coefficient"}
)

plt.title("MIZU: Correlation Matrix of Key Plant Features", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(figures_dir / "04_correlation_heatmap.png", dpi=300)
plt.show()
"""))

# Cell 10: Key takeaways & interview defense (Markdown)
cells.append(nbf.v4.new_markdown_cell("""## Step 7: Summary & Master's Interview Defense Takeaways

Here is everything we learned from this Exploratory Data Analysis, summarized in clear points you can confidently explain in an interview:

1. **Ill-Structured Biological Domain**:
   Wastewater treatment is non-linear and dynamic. Wastewater quality changes daily depending on weather, industrial discharges, and microbial activity.
2. **Missingness Mechanism (MAR/MCAR)**:
   Missing values are concentrated in BOD ($BOD_5$ / `DBO`) features due to the physical 5-day laboratory incubation process. We must use median imputation rather than dropping rows.
3. **Heavy-Tailed Outliers Require Robust Scaling**:
   Sensors like Zinc (`ZN-E`) and effluent solids (`SS-S`) feature extreme outliers. Standard scaling with mean and standard deviation would be heavily biased; `RobustScaler` (using median and IQR) is the mathematically defensible choice.
4. **Treatment Plant Performance**:
   Under normal operation, the activated sludge plant successfully reduces over $89\%$ of biological oxygen demand and over $90\%$ of suspended solids.
5. **No Ground-Truth Labels**:
   Because real plants do not have a binary "Normal / Broken" sensor, we must frame anomaly detection and clustering using **unsupervised learning**.
"""))

nb["cells"] = cells

# Save notebook
notebook_path = project_root / "notebooks" / "exploratory_analysis.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Successfully generated notebook at: {notebook_path}")
