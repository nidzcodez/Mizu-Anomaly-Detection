# MIZU (水) — Intelligent Water Systems Monitoring
### Explainable Unsupervised Machine Learning Framework for Urban Wastewater Treatment Plants

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)

---

## 1. Executive Summary & Vision

**MIZU** (derived from the Japanese word for water, 水) is a research and decision-support machine learning framework engineered to monitor, analyze, and diagnose operational anomalies across municipal wastewater treatment plants (WWTP).

Modern wastewater treatment processes rely on complex chemical, biological, and physical sedimentation dynamics across multiple connected stages (influent intake, primary clarifier settler, biological aeration basin, and secondary clarifier settler). Subtle operational drifts—such as filamentous sludge bulking, toxic shock loads, hydraulic storm surges, or equipment failures—can severely degrade effluent quality before conventional 5-day laboratory tests can confirm a breakdown.

MIZU applies **unsupervised anomaly detection** (Isolation Forest, Local Outlier Factor), **unsupervised operational clustering** (K-Means), **dimensionality reduction** (Principal Component Analysis), and **feature-level explainability** to sensor time-series data, enabling operators and environmental engineers to:
1. **Detect early operational deviations** without requiring historical anomaly labels.
2. **Discover canonical operational regimes** across seasons and inflow conditions.
3. **Deconstruct flagged anomalies** into specific physical and chemical sensor contributions.
4. **Interact with findings** via an intuitive, research-grade scientific monitoring dashboard.

> **Important Regulatory & Scope Disclaimer**  
> MIZU is an exploratory decision-support and analytical prototype designed for research and engineering analysis. It **does not certify drinking-water safety**, **does not detect microbiological pathogens or waterborne contaminants directly**, and **does not replace certified chemical or microbiological laboratory testing**.

---

## 2. United Nations Sustainable Development Goals (SDG) Alignment

MIZU is explicitly grounded in global environmental sustainability priorities:

- **UN SDG 6: Clean Water and Sanitation (Target 6.3)**  
  *Improve water quality by reducing pollution, eliminating dumping and minimizing release of hazardous chemicals and materials, halving the proportion of untreated wastewater and substantially increasing recycling and safe reuse globally.*  
  *MIZU Contribution:* By identifying operational failures in primary and secondary settlers early, MIZU helps prevent untreated effluent discharge containing elevated Biochemical Oxygen Demand ($BOD$) and Suspended Solids ($SS$) into downstream freshwater ecosystems.

- **UN SDG 9: Industry, Innovation and Infrastructure (Target 9.4)**  
  *Upgrade infrastructure and retrofit industries to make them sustainable, with increased resource-use efficiency and greater adoption of clean and environmentally sound technologies.*  
  *MIZU Contribution:* Demonstrates how legacy municipal physical infrastructure can be retrofitted with intelligent, explainable sensor analytics without prohibitive sensor overhaul costs.

- **UN SDG 12: Responsible Consumption and Production (Target 12.4 & 12.5)**  
  *Achieve environmentally sound management of chemicals and all wastes throughout their life cycle... substantially reduce waste generation through prevention, reduction, recycling and reuse.*  
  *MIZU Contribution:* Optimizes treatment plant stability, preventing chemical coagulant overdosing during false alarms and minimizing chemical waste.

---

## 3. Dataset Architecture & Operational Domain

The framework is benchmarked against the official **UCI Water Treatment Plant dataset** (UCI Machine Learning Repository ID: 106).

- **Data Provenance:** Daily sensor records collected from an urban activated sludge wastewater treatment plant located in Catalonia, Spain (created by Manel Poch et al., Universitat Autònoma de Barcelona).
- **Time Horizon:** 527 consecutive operational days (1990–1991).
- **Dimensionality:** 38 continuous physicochemical and performance variables + 1 temporal identifier (`Date`).
- **Nature of the Domain:** Characterized as an *ill-structured domain* where biological processes exhibit high non-linearity, time-lagged dynamics, and environmental stochasticity.

### Plant Process Stages & Variable Taxonomy

The 38 continuous attributes span four physical processing stages and derived performance ratios:

```
[Influent: -E] ──> [Primary Settler: -P] ──> [Secondary Settler/Aeration: -D] ──> [Effluent: -S]
       │                         │                               │                       │
       └─────────────────────────┴───────────────┬───────────────┴───────────────────────┘
                                                 ▼
                             [Performance / Removal Efficiencies: RD-]
```

| Prefix / Suffix | Process Stage | Key Monitored Parameters |
|:---|:---|:---|
| **`-E` (Entrada)** | Plant Influent / Input | `Q-E` (Flow rate, $m^3/day$), `ZN-E` (Zinc, $mg/L$), `PH-E` (pH), `DBO-E` (Biological Oxygen Demand, $mg/L$), `DQO-E` (Chemical Oxygen Demand, $mg/L$), `SS-E` (Suspended Solids, $mg/L$), `SSV-E` (Volatile Suspended Solids, %), `SED-E` (Settleable Solids, $mL/L$), `COND-E` (Conductivity, $\mu S/cm$) |
| **`-P` (Primario)** | Primary Clarifier Settler | `PH-P`, `DBO-P`, `SS-P`, `SSV-P`, `SED-P`, `COND-P` |
| **`-D` (Decantador Sec.)**| Biological Aeration & Secondary Clarifier | `PH-D`, `DBO-D`, `DQO-D`, `SS-D`, `SSV-D`, `SED-D`, `COND-D` |
| **`-S` (Salida)** | Plant Effluent / Output | `PH-S`, `DBO-S`, `DQO-S`, `SS-S`, `SSV-S`, `SED-S`, `COND-S` |
| **`RD-` (Rendimientos)** | Removal Efficiencies | `RD-DBO-P`, `RD-SS-P`, `RD-SED-P` (Primary Clarifier Reductions, %)<br>`RD-DBO-S`, `RD-DQO-S` (Secondary Stage Reductions, %)<br>`RD-DBO-G`, `RD-DQO-G`, `RD-SS-G`, `RD-SED-G` (Global Plant Efficiencies, %) |

---

## 4. Machine Learning & Engineering Methodology

### A. Missing Value Dynamics & Preprocessing
- **Domain Reality:** In wastewater laboratories, Biological Oxygen Demand ($BOD_5$ / `DBO`) requires a 5-day biological incubation window. Consequently, `DBO` features suffer from higher rates of missing values (up to 11.8% in `RD-DBO-P`).
- **Missingness Mechanism:** Missing completely at random (MCAR) or missing at random (MAR) due to laboratory batch delays or sensor calibration periods, encoded as `'?'` in raw files.
- **Handling Strategy:** Robust median or iterative multivariate imputation (MICE) evaluated against standard median imputation. Row dropping is prohibited as dropping incomplete operational days would discard nearly 30% of critical monitoring history.
- **Feature Scaling:** RobustScaler or StandardScaler fitted strictly within pipeline transforms to avoid data leakage.

### B. Unsupervised Anomaly Detection
1. **Isolation Forest ($iForest$):**
   - *Core Premise:* Anomalies are "few and different", meaning they require fewer random axis-aligned splits in an ensemble of isolation trees to become isolated in terminal leaf nodes.
   - *Score Calculation:* Path length $h(x)$ normalized by average path length $c(n)$ of unsuccessful searches in a Binary Search Tree (BST). Anomaly score $s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$.
   - *Contamination Assumption:* Configurable contamination hyperparameter $\nu \in [0.01, 0.05]$ reflecting realistic plant fault incidence rates ($<3\%$).
2. **Local Outlier Factor (LOF):**
   - *Core Premise:* Evaluates local density deviations with respect to $k$-nearest neighbors.
   - *Metrics:* Compares the local reachability density ($lrd$) of an observation against that of its neighbors. LOF scores significantly greater than 1 identify low-density outliers relative to surrounding operating clusters.

### C. Operational Pattern Discovery (Clustering & Dimensionality Reduction)
1. **K-Means Clustering:**
   - Discovers recurring operational regimes (e.g., normal low-influent days, heavy industrial load days, high-efficiency summer operation).
   - Cluster count $k$ is determined using multi-criteria validation: Elbow Method (Inertia), Silhouette Score analysis, and the Davies-Bouldin Index.
2. **Principal Component Analysis (PCA):**
   - Projects 38-dimensional sensor vectors onto orthogonal principal components capturing maximum operational variance.
   - Quantifies cumulative explained variance ratio to document dimensional information loss explicitly.

### D. Explainability & Attribution
- Computes standardized z-score deviations ($\Delta z = |x_j - \mu_j| / \sigma_j$) and SHAP (SHapley Additive exPlanations) values where defensible to pinpoint which specific sensors (e.g., spike in influent zinc `ZN-E` or drop in settling efficiency `RD-SED-G`) triggered an anomaly flag.

---

## 5. Evaluation Philosophy (Why Accuracy is Invalid)

In standard supervised machine learning, models are evaluated using classification accuracy:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Why accuracy cannot be used in MIZU:**
1. **Absence of Ground-Truth Labels:** The operational dataset does not have binary ground-truth labels in production. The plant observations represent continuous sensor telemetry.
2. **Extreme Class Imbalance:** In real-world treatment plants, abnormal operational faults represent $< 3\%$ of operating days. A trivial dummy model that predicts "Normal" for 100% of days would achieve $> 97\%$ accuracy while failing completely at its core mission.
3. **Model Concordance vs. Ground Truth:** Instead of naive accuracy, MIZU evaluates:
   - **Cross-model Concordance & Jaccard Similarity:** Quantifying the overlap between Isolation Forest and LOF detections:
     $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
   - **Internal Cluster Validation:** Silhouette coefficient ($s \in [-1, 1]$), Calinski-Harabasz score, and Davies-Bouldin index.
   - **Domain Literature Validation:** Comparing unsupervised clusters and top anomaly flags against historical fault dates documented in the original Catalonia research studies (classes 2, 3, 4, 6, 8, 10, 12, 13).

---

## 6. Project Architecture

The codebase follows professional software engineering standards with modular decoupling:

```
mizu/
├── data/
│   ├── raw/                           # Raw UCI dataset (water-treatment.data, water-treatment.names)
│   └── processed/                     # Cleaned, imputed, and scaled parquet/csv datasets
├── notebooks/
│   └── exploratory_analysis.ipynb     # Reproducible EDA and mathematical investigation
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_loader.py                 # Robust raw data loading and schema validation
│   ├── preprocessing.py               # Imputation, transformation, and scaling pipelines
│   ├── anomaly_detection.py           # Isolation Forest and LOF wrappers with scoring
│   ├── clustering.py                  # K-Means clustering with k-selection heuristics
│   ├── dimensionality_reduction.py    # PCA with explained variance decomposition
│   ├── evaluation.py                  # Unsupervised metrics, concordance, domain validation
│   └── explainability.py              # Sensor deviation analysis and local attribution
├── models/                            # Serialized scikit-learn pipeline artifacts
├── scripts/
│   └── download_data.py               # Automated dataset acquisition script
├── app.py                             # Interactive Streamlit scientific monitoring dashboard
├── requirements.txt                   # Documented pinned dependencies
├── README.md                          # Comprehensive project documentation
└── .gitignore                         # Version control exclusions
```

---

## 7. Installation & Quickstart

### Prerequisites
- macOS, Linux, or Windows with **Python 3.11** installed.
- Git.

### 1. Clone & Navigate
```bash
git clone https://github.com/nidahermein/mizu.git
cd mizu
```

### 2. Set Up Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Fetch the UCI Dataset
```bash
python scripts/download_data.py
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

---

## 8. Master's Interview Defense Guide

When defending this project in an AI/ML interview, emphasize:
1. **Domain Understanding Before Algorithms:** You understood wastewater treatment unit operations (influent, clarifiers, aeration, performance ratios) before writing model code.
2. **Missingness Mechanics:** Why $BOD_5$ is frequently missing due to the physical 5-day incubation protocol, justifying robust imputation over data deletion.
3. **Rigorous Unsupervised Validation:** Why accuracy is statistically deceptive for anomaly detection, how unsupervised concordance metrics work, and how external domain literature acts as a secondary validation anchor.
4. **Computational Complexity & Trade-offs:**
   - *Isolation Forest:* $\mathcal{O}(n \cdot t \cdot \psi \log \psi)$, sub-sampling $\psi = 256$, highly scalable to streaming telemetry.
   - *Local Outlier Factor:* $\mathcal{O}(n^2 \cdot d)$ without indexing trees, local density sensitivity vs. computational cost.
   - *PCA:* $\mathcal{O}(\min(n^3, d^3))$ SVD decomposition, linear assumption limitations vs. non-linear operational dynamics.
5. **Ethics & Limitations:** Clear delineation between research-grade operational decision-support and statutory drinking-water compliance testing.

---

## 9. Author & License

- **Project Lead:** Nida Hermein (BCA Cloud Architecture & Cybersecurity)
- **License:** MIT License — free for educational and research usage.

