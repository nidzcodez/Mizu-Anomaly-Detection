# MIZU Operational Data Dictionary & Feature Specification
### UCI Water Treatment Plant Dataset (ID: 106)

---

## 1. Dataset Overview & Provenance

- **Source:** Urban Waste Water Treatment Plant, Catalonia, Spain (Unitat d'Enginyeria Química, Universitat Autònoma de Barcelona).
- **Timeframe:** Daily sensor measures recorded between March 1, 1990 and August 30, 1991.
- **Instance Count:** 527 operational days.
- **Dimensionality:** 39 columns (1 Temporal identifier + 38 continuous physicochemical measurements).
- **Data Representation:** Numeric continuous attributes with missing values indicated as `'?'` in raw files.

```
                          PHYSICAL PROCESS FLOW
                          ─────────────────────
 [ Influent: -E ] ──────> [ Primary Settler: -P ] ──────> [ Biological / Aeration: -D ] ──────> [ Effluent: -S ]
  Raw Municipal            Physical Sedimentation          Microbiological Digestion             Treated Discharge
  & Industrial Intake      & Coagulation                   by Activated Sludge                   to River/Sea
        │                         │                               │                                 │
        └─────────────────────────┴───────────────┬───────────────┴─────────────────────────────────┘
                                                  ▼
                               [ Performance Efficiencies: RD- ]
```

---

## 2. Complete Column Specification (39 Columns)

| Index | Column Name | Physical / Operational Meaning | Process Stage | Unit | Data Type | Missing Count | Missing % | Min | Mean | Max | Std | Domain Interpretation |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **0** | `Date` | Operational Date Identifier | Metadata | String (`D-D/M/YY`) | `object` | 0 | 0.00% | `D-1/3/90` | N/A | `D-30/8/91` | N/A | Sequential calendar day of measurement. |
| **1** | `Q-E` | Input flow rate to plant | Influent (`-E`) | $m^3/\text{day}$ | `float64` | 18 | 3.42% | 10,050.0 | 37,226.57 | 60,081.0 | 6,636.00 | Total hydraulic volumetric load entering the treatment facility. Surges indicate rainfall/storms. |
| **2** | `ZN-E` | Input Zinc concentration | Influent (`-E`) | $mg/L$ | `float64` | 3 | 0.57% | 0.10 | 2.36 | 33.50 | 2.75 | Heavy metal tracer indicating industrial wastewater discharges. Toxic to biological sludge at high levels. |
| **3** | `PH-E` | Input pH (Acidity / Alkalinity) | Influent (`-E`) | $pH \text{ units}$ | `float64` | 0 | 0.00% | 6.90 | 7.81 | 8.70 | 0.25 | Measure of hydrogen-ion activity. Extreme deviations inhibit microbial flora. |
| **4** | `DBO-E` | Input Biological Oxygen Demand ($BOD_5$) | Influent (`-E`) | $mg/L$ | `float64` | 23 | 4.36% | 31.00 | 188.71 | 438.00 | 62.06 | Oxygen required by microorganisms to decompose biodegradable organic matter over 5 days. |
| **5** | `DQO-E` | Input Chemical Oxygen Demand ($COD$) | Influent (`-E`) | $mg/L$ | `float64` | 6 | 1.14% | 81.00 | 406.90 | 941.00 | 120.36 | Total oxygen required to chemically oxidize all organic compounds (biodegradable + non-biodegradable). |
| **6** | `SS-E` | Input Suspended Solids | Influent (`-E`) | $mg/L$ | `float64` | 1 | 0.19% | 98.00 | 227.44 | 2,008.00 | 135.94 | Dry weight of insoluble particulate matter in incoming wastewater. |
| **7** | `SSV-E` | Input Volatile Suspended Solids | Influent (`-E`) | $\%$ | `float64` | 11 | 2.09% | 13.20 | 61.39 | 85.00 | 12.42 | Ratio of organic/combustible solids relative to total solids. Reflects organic strength. |
| **8** | `SED-E` | Input Settleable Solids | Influent (`-E`) | $mL/L$ | `float64` | 25 | 4.74% | 0.40 | 4.59 | 36.00 | 2.74 | Volume of solids that settle out of a 1-liter cone (Imhoff cone) in 1 hour. |
| **9** | `COND-E` | Input Electrical Conductivity | Influent (`-E`) | $\mu S/cm$ | `int64` | 0 | 0.00% | 651.0 | 1,478.62 | 3,230.0 | 394.90 | Measure of total dissolved inorganic ions and salts. Highly correlated with industrial spills. |
| **10** | `PH-P` | Input pH to primary settler | Primary (`-P`) | $pH \text{ units}$ | `float64` | 0 | 0.00% | 7.30 | 7.83 | 8.50 | 0.23 | pH entering physical clarification stage after screening and grit removal. |
| **11** | `DBO-P` | Input Biological Oxygen Demand to primary settler | Primary (`-P`) | $mg/L$ | `float64` | 40 | 7.59% | 32.00 | 206.21 | 517.00 | 74.82 | Organic load entering primary clarification. |
| **12** | `SS-P` | Input Suspended Solids to primary settler | Primary (`-P`) | $mg/L$ | `int64` | 0 | 0.00% | 104.0 | 253.95 | 1,692.0 | 147.46 | Solid particulate burden confronting primary sedimentation basins. |
| **13** | `SSV-P` | Input Volatile Suspended Solids to primary settler | Primary (`-P`) | $\%$ | `float64` | 11 | 2.09% | 7.10 | 60.37 | 93.50 | 12.39 | Organic fraction entering primary settler. |
| **14** | `SED-P` | Input Settleable Solids to primary settler | Primary (`-P`) | $mL/L$ | `float64` | 24 | 4.55% | 1.00 | 5.03 | 46.00 | 3.35 | Settleable fraction targeting primary clarification. |
| **15** | `COND-P` | Input Electrical Conductivity to primary settler | Primary (`-P`) | $\mu S/cm$ | `int64` | 0 | 0.00% | 646.0 | 1,496.03 | 3,170.0 | 402.59 | Dissolved ionic profile into primary settler. |
| **16** | `PH-D` | Input pH to biological / secondary settler | Secondary (`-D`) | $pH \text{ units}$ | `float64` | 0 | 0.00% | 7.10 | 7.81 | 8.40 | 0.20 | pH entering secondary aeration/biological reactor. Bacteria require $6.8 \le pH \le 8.2$. |
| **17** | `DBO-D` | Input Biological Oxygen Demand to secondary settler | Secondary (`-D`) | $mg/L$ | `float64` | 28 | 5.31% | 26.00 | 122.35 | 285.00 | 37.02 | Soluble biodegradable substrate feeding the activated sludge microorganisms. |
| **18** | `DQO-D` | Input Chemical Oxygen Demand to secondary settler | Secondary (`-D`) | $mg/L$ | `float64` | 9 | 1.71% | 80.00 | 274.05 | 511.00 | 74.12 | Total organic load confronting aeration basin. |
| **19** | `SS-D` | Input Suspended Solids to secondary settler | Secondary (`-D`) | $mg/L$ | `float64` | 2 | 0.38% | 49.00 | 94.22 | 244.00 | 23.99 | Solids escaping primary settler into secondary reactor. |
| **20** | `SSV-D` | Input Volatile Suspended Solids to secondary settler | Secondary (`-D`) | $\%$ | `float64` | 13 | 2.47% | 20.20 | 72.97 | 100.00 | 10.47 | Active organic biological fraction in mixed liquor. |
| **21** | `SED-D` | Input Settleable Solids to secondary settler | Secondary (`-D`) | $mL/L$ | `float64` | 25 | 4.74% | 0.00 | 0.42 | 3.50 | 0.38 | Settleable solids exiting primary treatment. |
| **22** | `COND-D` | Input Electrical Conductivity to secondary settler | Secondary (`-D`) | $\mu S/cm$ | `int64` | 0 | 0.00% | 85.0 | 1,490.57 | 3,690.0 | 399.99 | Salt concentration across biological stage. |
| **23** | `PH-S` | Output pH (Effluent) | Effluent (`-S`) | $pH \text{ units}$ | `float64` | 1 | 0.19% | 7.00 | 7.71 | 9.70 | 0.19 | Acidity of treated discharge. Regulatory limits typically enforce $6.5 \le pH \le 8.5$. |
| **24** | `DBO-S` | Output Biological Oxygen Demand (Effluent) | Effluent (`-S`) | $mg/L$ | `float64` | 23 | 4.36% | 3.00 | 19.99 | 320.00 | 17.59 | Residual organic pollution. Values $>25-30\ mg/L$ indicate severe treatment failure. |
| **25** | `DQO-S` | Output Chemical Oxygen Demand (Effluent) | Effluent (`-S`) | $mg/L$ | `float64` | 18 | 3.42% | 9.00 | 87.29 | 350.00 | 39.03 | Residual chemical pollutants reaching natural receiving waters. |
| **26** | `SS-S` | Output Suspended Solids (Effluent) | Effluent (`-S`) | $mg/L$ | `float64` | 5 | 0.95% | 6.00 | 22.24 | 238.00 | 16.34 | Residual particulate matter. High values indicate sludge bulking or clarifier overflow. |
| **27** | `SSV-S` | Output Volatile Suspended Solids (Effluent) | Effluent (`-S`) | $\%$ | `float64` | 17 | 3.23% | 29.20 | 80.15 | 100.00 | 9.16 | Proportion of effluent solids composed of organic biomass. |
| **28** | `SED-S` | Output Settleable Solids (Effluent) | Effluent (`-S`) | $mL/L$ | `float64` | 28 | 5.31% | 0.00 | 0.04 | 3.50 | 0.20 | Solids settling out of final discharge. Must approach 0 in healthy plant. |
| **29** | `COND-S` | Output Electrical Conductivity (Effluent) | Effluent (`-S`) | $\mu S/cm$ | `float64` | 1 | 0.19% | 683.0 | 1,494.82 | 3,950.0 | 387.91 | Total dissolved ions in discharged effluent. |
| **30** | `RD-DBO-P` | Removal efficiency of DBO in primary settler | Efficiency (`RD-`) | $\%$ | `float64` | 62 | 11.76% | 0.60 | 39.09 | 79.10 | 14.79 | Percentage reduction: $\frac{\text{DBO-E} - \text{DBO-P}}{\text{DBO-E}} \times 100$. |
| **31** | `RD-SS-P` | Removal efficiency of Suspended Solids in primary settler | Efficiency (`RD-`) | $\%$ | `float64` | 4 | 0.76% | 5.30 | 58.52 | 96.10 | 12.80 | Primary clarification settling efficiency for solids. |
| **32** | `RD-SED-P` | Removal efficiency of Settleable Solids in primary settler | Efficiency (`RD-`) | $\%$ | `float64` | 27 | 5.12% | 7.70 | 90.55 | 100.00 | 8.94 | Primary settleable solids removal. Healthy clarifiers remove $>90\%$. |
| **33** | `RD-DBO-S` | Removal efficiency of DBO in secondary settler | Efficiency (`RD-`) | $\%$ | `float64` | 40 | 7.59% | 8.20 | 83.45 | 94.70 | 8.80 | Biological decomposition efficiency across aeration and secondary settling. |
| **34** | `RD-DQO-S` | Removal efficiency of DQO in secondary settler | Efficiency (`RD-`) | $\%$ | `float64` | 26 | 4.93% | 1.40 | 67.82 | 96.80 | 11.62 | Secondary chemical oxidation efficiency. |
| **35** | `RD-DBO-G` | Global plant removal efficiency for DBO | Efficiency (`RD-`) | $\%$ | `float64` | 36 | 6.83% | 19.60 | 89.01 | 97.00 | 7.03 | Overall biological treatment performance across entire facility. |
| **36** | `RD-DQO-G` | Global plant removal efficiency for DQO | Efficiency (`RD-`) | $\%$ | `float64` | 25 | 4.74% | 19.20 | 77.86 | 98.10 | 8.89 | Overall chemical oxygen demand removal efficiency. |
| **37** | `RD-SS-G` | Global plant removal efficiency for Suspended Solids | Efficiency (`RD-`) | $\%$ | `float64` | 8 | 1.52% | 10.30 | 88.96 | 99.40 | 8.22 | Overall solids capture performance. Healthy threshold $\ge 85-90\%$. |
| **38** | `RD-SED-G` | Global plant removal efficiency for Settleable Solids | Efficiency (`RD-`) | $\%$ | `float64` | 31 | 5.88% | 36.40 | 99.09 | 100.00 | 4.46 | Global clarification efficiency. Expected close to $99-100\%$. |

---

## 3. Analysis of Missingness Mechanisms

- **Mechanism:** Missingness is not uniform. Out of 38 numeric features:
  - 8 features have **0 missing values** (`PH-E`, `COND-E`, `PH-P`, `SS-P`, `COND-P`, `PH-D`, `COND-D`, plus `Date`). These represent instantaneous sensor probes.
  - 4 features have **$> 35$ missing values** (`RD-DBO-P`: 62, `RD-DBO-S`: 40, `DBO-P`: 40, `RD-DBO-G`: 36).
- **Physical Justification:** As established, $BOD_5$ is a **5-day microbial bio-assay**. If an incubator malfunctions, the dilution water is contaminated, or glassware breaks, the reading is lost retroactively.
- **Derived Ratios:** Notice that $RD\text{-}DBO\text{-}P$ is a mathematical function of $DBO\text{-}E$ and $DBO\text{-}P$. If either input is missing, the reduction percentage is inherently missing.
- **Handling Strategy:** Row dropping is inadmissible (would eliminate $\approx 30\%$ of all operational days). We implement **Median Imputation** or **Iterative Multivariate Imputation (MICE)** fitted strictly on training slices.

---

## 4. Dataset Limitations & Scope Delineation

| Dimension | Real-World Limitation | Analytical / Operational Implication |
|:---|:---|:---|
| **Temporal Granularity** | Daily aggregated composite samples (1 observation per 24 hours). | Cannot detect transient sub-hourly hydraulic surges or instantaneous equipment trips. |
| **Sample Size** | 527 operational days (~1.5 years). | Adequate for prototype machine learning; insufficient for deep recurrent networks or multi-decade climate trend analysis. |
| **Ground-Truth Labels** | Unsupervised telemetry; no binary fault target in production data. | Standard classification accuracy is statistically invalid. Anomaly models must be evaluated via unsupervised metrics, concordance, and domain literature cross-referencing. |
| **Class Imbalance** | Abnormal operational states represent $<3\%$ of total days in domain literature. | Any supervised metric would be overwhelmed by high true-negative rates (class imbalance trap). |
| **Chemical Scope** | Measures bulk bulk aggregates ($COD, BOD, SS, Zn, pH, COND$). | Does **not** measure microbiological pathogens ($E. coli$, coliforms, viruses) or specific emerging contaminants (PFAS, pharmaceuticals). |

---

## 5. What MIZU Realistically Detects vs What It Does Not

### What MIZU CAN Realistically Do:
1. **Detect Multivariate Operational Anomalies:** Identify simultaneous sensor drifts indicative of hydraulic storm overloads, toxic industrial shocks ($Zn, COND$), or secondary clarification failures (sludge bulking, solids carryover).
2. **Discover Operational Regimes:** Group 527 operating days into meaningful operating clusters (e.g., low-flow weekend operation, heavy organic load cycles, optimal summer performance).
3. **Deconstruct Feature Attribution:** Provide explainability for why an observation was flagged (e.g., "Observation 125 flagged due to a $5.2\sigma$ spike in influent Zinc and a $38\%$ collapse in Secondary Clarifier efficiency").
4. **Assist Plant Operators:** Provide an interactive decision-support interface that prioritizes which operational days warrant manual engineering inspection.

### What MIZU CANNOT Do (Regulatory & Technical Bounds):
1. **Cannot Certify Drinking Water Safety:** This is an **urban wastewater treatment plant**, not a drinking water treatment facility. Effluent is discharged into environmental waterways, not potable water taps.
2. **Cannot Detect Pathogen Contamination:** Sensors measure bulk physical/chemical properties, not microbiological organisms.
3. **Cannot Replace Certified Laboratory Testing:** MIZU is a machine-learning research decision-support tool, not an accredited laboratory instrument.

