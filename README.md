# YUVA AI Internship Portfolio

Welcome to the **YUVA AI Internship** project repository (`YUVA-AI-INTERNSHIP`). This repository documents the practical milestones, source code, data pipelines, visual insights, and technical reports developed throughout the internship program.

---

## Portfolio Structure

```text
YUVA AI INTERNSHIP/
├── README.md                                     # Full internship portfolio overview
└── Week 1/                                       # Week 1: Python for ML & Data Preprocessing
    ├── preprocessing.py                          # Data cleaning, encoding, and scaling script
    ├── README.md                                 # Week 1 milestone documentation
    ├── requirements.txt                          # Python dependencies for Week 1
    ├── Week_1_Python_ML_Data_Preprocessing_Report.docx # Verified comprehensive report
    └── outputs/                                  # Verified pipeline outputs
        ├── student_performance_preprocessed.csv # Preprocessed dataset (395 x 42)
        ├── final_grade_distribution.png          # Final grade (G3) distribution graph
        └── average_grade_by_studytime.png        # Study time vs average grade graph
```

---

## Weekly Milestones

### [Week 1: Python for Machine Learning & Data Preprocessing](Week%201/README.md)
- **Focus:** Exploratory Data Analysis (EDA), automated data integrity validation, categorical encoding, feature normalization, and academic performance distribution visualization.
- **Dataset:** UCI Student Performance Dataset (`student-mat.csv`, 395 records, 33 attributes).
- **Key Deliverables:**
  - Automated preprocessing script ([preprocessing.py](Week%201/preprocessing.py)).
  - Processed and normalized dataset ([student_performance_preprocessed.csv](Week%201/outputs/student_performance_preprocessed.csv)).
  - Distribution graphs ([final_grade_distribution.png](Week%201/outputs/final_grade_distribution.png), [average_grade_by_studytime.png](Week%201/outputs/average_grade_by_studytime.png)).
  - Comprehensive technical report ([Week_1_Python_ML_Data_Preprocessing_Report.docx](Week%201/Week_1_Python_ML_Data_Preprocessing_Report.docx)).

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/ayandhara19/YUVA-AI-INTERNSHIP.git
cd YUVA-AI-INTERNSHIP
```

### 2. Set Up the Environment
Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies & Run Week 1
```bash
pip install -r "Week 1/requirements.txt"
python "Week 1/preprocessing.py"
```

---

## Dataset Attribution
The dataset utilized in Week 1 is the **Student Performance Dataset** hosted on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008), licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
