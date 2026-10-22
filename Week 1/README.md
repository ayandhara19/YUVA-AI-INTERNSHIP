# Week 1: Python for Machine Learning & Data Preprocessing

This directory contains the complete implementation, verified outputs, technical documentation, and dependency specifications for **Week 1** of the **YUVA AI Internship Program**.

---

## 1. Project Overview

The objective of Week 1 is to implement an end-to-end data preparation and exploratory data analysis (EDA) pipeline using Python. The project uses the authentic **Student Performance Dataset** to clean raw academic records, perform exploratory analysis, handle categorical encoding, apply bounded numerical scaling, and export model-ready data alongside high-resolution visualizations.

---

## 2. Directory Structure

```text
Week 1/
├── preprocessing.py                              # Core data preprocessing and visualization pipeline
├── README.md                                     # Week 1 project documentation and instructions
├── requirements.txt                              # Python environment dependencies
├── Week_1_Python_ML_Data_Preprocessing_Report.docx # Comprehensive technical report with verified results
├── outputs/                                      # Pipeline generated outputs
│   ├── student_performance_preprocessed.csv     # Transformed, encoded, and normalized dataset (395 x 42)
│   ├── final_grade_distribution.png              # High-resolution histogram of final grades (G3)
│   └── average_grade_by_studytime.png            # High-resolution bar chart of average grade by study time
└── student-mat.csv                               # Local raw dataset (UCI Student Performance)
```

---

## 3. Dataset Information

- **Dataset Name:** Student Performance Dataset (Mathematics)
- **Source:** UCI Machine Learning Repository (Cortez and Silva, 2008)
- **Format:** Semicolon-delimited CSV (`sep=";"`)
- **Records (Rows):** 395 student instances
- **Original Features (Columns):** 33 features (16 numerical/integer, 17 categorical/object)
- **Target Variable:** `G3` (Final mathematics grade, integer scale 0–20)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

> **Note on Raw Dataset Distribution:** If `student-mat.csv` is excluded from public source control to minimize repository footprint or adhere to redistribution policies, you can download `student-mat.csv` directly from the [UCI Machine Learning Repository: Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) and place `student-mat.csv` in the `Week 1/` folder.

---

## 4. Preprocessing & Engineering Workflow

`preprocessing.py` executes the following verified pipeline:

1. **Path Resolution & Data Ingestion:**
   - Robustly locates `student-mat.csv` relative to script location using `pathlib.Path`.
   - Parses semicolon-separated fields (`sep=";"`).
2. **Data Integrity Inspection:**
   - Audits missing values (`isnull().sum()`) -> Verified 0 missing values across all 33 columns.
   - Audits duplicate records (`duplicated().sum()`) -> Verified 0 duplicate rows.
   - Includes robust fallback imputation (median for numeric features, mode for categorical features).
3. **Categorical Encoding:**
   - Detects 17 categorical features (`school`, `sex`, `address`, `famsize`, `Pstatus`, `Mjob`, `Fjob`, `reason`, `guardian`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`).
   - Applies one-hot encoding (`pd.get_dummies(..., drop_first=True, dtype=int)`), expanding the feature space from 33 to 42 columns without multicollinearity.
4. **Feature Normalization:**
   - Preserves standard academic evaluation scale for grade columns (`G1`, `G2`, `G3`).
   - Normalizes all remaining 39 numeric and dummy features to a `[0, 1]` bounded range using `MinMaxScaler()`.
5. **Output Generation:**
   - Exports the processed dataset to `outputs/student_performance_preprocessed.csv`.
6. **Exploratory Visualizations:**
   - Generates `outputs/final_grade_distribution.png` (300 DPI): Histogram illustrating student distribution across final grades (G3).
   - Generates `outputs/average_grade_by_studytime.png` (300 DPI): Bar chart displaying mean final grade across weekly study time tiers.

---

## 5. Summary of Verified Results

| Metric / Parameter | Raw Dataset | Preprocessed Dataset |
| :--- | :--- | :--- |
| **Total Rows** | 395 | 395 |
| **Total Columns** | 33 | 42 |
| **Missing Values** | 0 | 0 |
| **Duplicate Rows** | 0 | 0 |
| **Demographic / Encoded Features Scale** | Unnormalized integers / strings | Scaled to `[0.0, 1.0]` |
| **Target Scale (`G3`)** | `0` to `20` (Mean: 10.42, Std: 4.58) | Preserved on `0` to `20` |

---

## 6. Setup and Execution

### Step 1: Set Up Python Environment
Ensure Python 3.8+ is installed on your system:
```bash
python --version
```

### Step 2: Install Dependencies
Install all required libraries specified in [requirements.txt](requirements.txt):
```bash
pip install -r requirements.txt
```

### Step 3: Run the Preprocessing Script
Navigate to the `Week 1` folder or run from repository root:
```bash
python "Week 1/preprocessing.py"
```
Or when inside `Week 1/`:
```bash
python preprocessing.py
```

### Step 4: Verify Outputs
Check the `outputs/` folder for generated artifacts:
- `outputs/student_performance_preprocessed.csv`
- `outputs/final_grade_distribution.png`
- `outputs/average_grade_by_studytime.png`