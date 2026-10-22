# Week 2: Supervised Machine Learning Models

## Objective
Train and compare regression and classification models using the UCI Student Performance mathematics dataset.

## Directory Structure
```text
Week 2/
├── supervised_ml.py                            # Supervised learning regression and classification pipeline
├── README.md                                   # Week 2 project documentation
├── requirements.txt                            # Dependencies (pandas, numpy, matplotlib, scikit-learn)
├── REPORT_PLAN.txt                             # Week 2 report outline and guidelines
├── Week_2_Supervised_ML_Project_Report.docx    # Final project report document
└── outputs/                                    # Generated model outputs and visualizations
    ├── regression_model_comparison.csv         # Regression metrics (MAE, RMSE, R2)
    ├── regression_predictions.csv              # Regression predictions on test split
    ├── regression_model_comparison.png         # Bar chart comparing regression models
    ├── regression_actual_vs_predicted.png      # Scatter plot of actual vs predicted G3
    ├── classification_model_comparison.csv     # Classification metrics (Accuracy, Precision, Recall, F1)
    ├── classification_predictions.csv          # Classification test set predictions
    ├── classification_report.txt               # Precision, recall, f1-score report
    ├── classification_model_comparison.png     # Bar chart comparing classification models
    └── classification_confusion_matrix.png     # Confusion matrix visualization
```

## Dataset
Use the original `student-mat.csv` from Week 1. The script expects it at `../Week 1/student-mat.csv` relative to this folder. The CSV is semicolon-delimited.

If your dataset is stored elsewhere, set `DATA_PATH` to its full path before running the script.

### Windows PowerShell example
```powershell
$env:DATA_PATH="C:\full\path\to\student-mat.csv"
python supervised_ml.py
```

## Setup & Execution
```bash
pip install -r requirements.txt
python supervised_ml.py
```

## Models
- Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor, KNN Regressor.
- Classification: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, KNN Classifier.

The classification target is `pass = 1` when `G3 >= 10`, otherwise `pass = 0`. The final grade and earlier grades `G1` and `G2` are excluded from input features. Regression predicts `G3` while also excluding `G1` and `G2`.

## Leakage Prevention
The script splits data before fitting imputers, scalers, and encoders. These preprocessing steps are inside scikit-learn Pipelines and are fitted using training data only. Classification uses stratified splitting.

## Outputs & Deliverables
The script creates CSV metric tables, prediction CSVs, a classification report, and PNG graphs in `outputs/`. Model results depend on the dataset and software versions, so use the generated results when writing conclusions. The comprehensive project analysis and findings are documented in `Week_2_Supervised_ML_Project_Report.docx`.

## Important
Do not claim one model is best until you have run the script and checked the metrics. This is an educational experiment, not a validated system for making real decisions about students.
