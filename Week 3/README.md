# Week 3: Unsupervised Learning & Model Evaluation

This directory contains the complete implementation, synthetic demonstration dataset, evaluation metrics, visual insights, and technical documentation for **Week 3** of the **YUVA AI Internship Program**.

---

## 1. Project Overview & Objective

The objective of Week 3 is to implement, evaluate, and compare unsupervised clustering algorithms for **Customer Segmentation**. Using multi-attribute demographic and behavioral customer features, the pipeline partitions unlabeled records into meaningful customer profiles using:
- **K-Means Clustering** (partition-based centroid algorithm)
- **Hierarchical Clustering** (agglomerative bottom-up clustering with Ward linkage)

### What are Unsupervised Learning and Customer Segmentation?
- **Unsupervised Learning:** A class of machine learning where algorithms discover intrinsic patterns, groupings, and latent structures from unlabeled input data ($X$) without predefined target labels ($y$) or external supervision.
- **Customer Segmentation:** The business and analytical practice of dividing a customer base into homogeneous cohorts based on shared demographic, purchasing, or behavioral attributes. This allows organizations to understand customer diversity, personalize product offerings, and optimize engagement strategies without needing pre-existing customer category labels.

---

## 2. Directory Structure

```text
Week 3/
├── unsupervised_ml.py                      # Unsupervised learning pipeline (K-Means & Hierarchical)
├── README.md                               # Week 3 milestone documentation and execution guide
├── requirements.txt                        # Environment dependencies (pandas, scikit-learn, matplotlib)
├── REPORT_PLAN.txt                         # Detailed technical report structure and checklist
├── customer_segmentation_demo.csv          # Synthetic demonstration customer dataset (201 records)
├── Week_3_Unsupervised_ML_Report.docx      # Comprehensive verified project report document
└── outputs/                                # Generated pipeline outputs, metrics, and figures
    ├── cluster_selection_metrics.csv       # Inertia and silhouette scores across candidate k (2 to 7)
    ├── model_comparison.csv                # Silhouette comparison (K-Means vs Agglomerative)
    ├── customer_segments.csv               # Customer records with assigned cluster labels
    ├── kmeans_cluster_summary.csv          # Mean feature values across identified segments
    ├── elbow_method.png                    # Elbow curve plot (Inertia vs k)
    ├── silhouette_scores.png               # Silhouette score plot across tested cluster counts
    ├── customer_clusters_pca.png           # 2D PCA projection scatter plot of customer clusters
    └── README.txt                          # Output directory usage note
```

---

## 3. Dataset Information

- **Dataset File:** `customer_segmentation_demo.csv`
- **Total Records:** 201 customer instances (rows)
- **Attributes:** `CustomerID`, `Gender`, `Age`, `AnnualIncome_kUSD`, `SpendingScore_1_100`
- **Synthetic Nature Notice:** This dataset is **synthetic demonstration data** created specifically for instructional and demonstration purposes. It does not represent real customer transactions, personally identifiable records, or data collected from real-world consumers.
- **Features Selected for Clustering:**
  - `Age`: Customer age in years
  - `AnnualIncome_kUSD`: Annual income in thousands of USD
  - `SpendingScore_1_100`: Normalized spending score assigned based on customer purchasing behavior (1–100 scale)

Before clustering, all selected numerical features are standardized using `StandardScaler` to ensure zero mean and unit variance, preventing variables with larger numerical scales from disproportionately influencing distance calculations.

---

## 4. Setup and Execution

### Step 1: Install Dependencies
Open a terminal in the `Week 3/` folder and install the required dependencies:
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the Clustering Pipeline
Execute the Python script:
```bash
python unsupervised_ml.py
```

The script automatically executes the data ingestion, feature standardization, cluster selection sweep ($k=2$ through $7$), model training, cluster profile summary generation, and visualization exports to the `outputs/` folder.

---

## 5. Evaluation Methodology & Limitations

### Unsupervised Evaluation Metrics
Because unsupervised learning operates on unlabeled data without ground-truth class labels, evaluation is performed through intrinsic geometric cluster validation:
1. **Inertia (Within-Cluster Sum of Squares):** Evaluated across candidate values of $k \in [2, 7]$ using the Elbow Method to identify the point of diminishing returns in cluster compactness.
2. **Silhouette Coefficient:** Evaluates cluster cohesion (intra-cluster distance) and separation (nearest-cluster distance), bounded in $[-1, 1]$. In this demonstration, $k = 4$ yields the optimal silhouette score ($\approx 0.481$ for K-Means and $\approx 0.462$ for Hierarchical Clustering).

### Note on Classification Metrics
Supervised classification metrics—such as **accuracy**, **precision**, **recall**, **F1-score**, **confusion matrix**, and **ROC-AUC**—require known ground-truth targets ($y$). They **cannot be directly calculated or applied** to ordinary unlabeled clustering problems unless external benchmark labels or an explicit downstream supervised validation task are provided. For this project, evaluation strictly relies on mathematically appropriate unsupervised clustering metrics.
