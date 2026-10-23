# Week 4 - AI Project Deployment & Capstone
## Mall Customer Segmentation (K-Means + Flask)

An end-to-end machine learning project: the Week 3 customer segmentation work
is turned into a **deployed web application and prediction API**.

| Step | What was done |
|------|---------------|
| Preprocessing | Validate columns, drop duplicates / missing / out-of-range rows |
| Training | `StandardScaler` + `KMeans` in one scikit-learn `Pipeline` |
| Model selection | Elbow method + silhouette score for K = 2..10 -> **K = 5** |
| Serialization | Pipeline saved with **Joblib** (`models/customer_segmentation.joblib`) |
| Deployment | **Flask** web page + JSON API, `gunicorn` ready |
| Testing | 17 automated **pytest** tests |
| Experiments | `compare_models.py`: 5 algorithms, 5 feature sets, scaling, seed stability, fairness check |

## Results

| Metric | Value |
|--------|-------|
| Customers | 200 |
| Clusters (K) | 5 |
| Silhouette score | 0.555 |
| Davies-Bouldin index | 0.572 |

| Segment | Customers | Avg income (k$) | Avg spending score |
|---------|-----------|-----------------|--------------------|
| Average Shoppers | 81 | 55.3 | 49.5 |
| Premium Shoppers | 39 | 86.5 | 82.1 |
| Careful Spenders | 35 | 88.2 | 17.1 |
| Budget Customers | 23 | 26.3 | 20.9 |
| Impulsive Spenders | 22 | 25.7 | 79.4 |

## Project structure

```
Week 4/
├── data/
│   └── Mall_Customers.csv            # input dataset
├── models/
│   ├── customer_segmentation.joblib  # serialized model
│   └── metadata.json                 # features, segment names, metrics
├── outputs/
│   ├── cluster_visualization.png     # segment scatter plot
│   ├── elbow_silhouette.png          # how K was chosen
│   ├── customer_segments.csv         # dataset + assigned segment
│   ├── cluster_profiles.csv          # average profile per segment
│   ├── model_comparison.csv/.png     # algorithm comparison
│   ├── feature_comparison.csv        # feature-set comparison
│   ├── segment_fairness_check.csv    # gender / age make-up per segment
│   └── web_app_screenshot.png        # screenshot of the running app
├── templates/
│   └── index.html                    # web interface
├── static/
│   └── style.css
├── tests/
│   └── test_app.py                   # pytest suite
├── train_model.py                    # preprocessing + training + saving
├── compare_models.py                 # algorithm / feature / fairness experiments
├── app.py                            # Flask app + prediction API
├── requirements.txt
├── Procfile                          # for Heroku/Railway style hosts
├── render.yaml                       # for Render
├── README.md
├── PROJECT_PLAN.txt
├── Week_4_AI_Capstone_Report.docx
└── Capstone_Presentation.pptx
```

## Run it locally

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python compare_models.py   # optional experiments
python app.py
```
macOS / Linux:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```
Open **http://127.0.0.1:5000**.

> Run `python train_model.py` once on your machine. It rebuilds the model with
> *your* scikit-learn version, which avoids version-mismatch warnings when the
> `.joblib` file is loaded.

Run the tests:
```bash
python -m pytest -q
```

## API

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Health check |
| GET | `/api/info` | Model details, segment names and profiles |
| POST | `/predict` | Predict the segment of one customer |

**Request**
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"annual_income": 90, "spending_score": 85}'
```
**Response**
```json
{
  "cluster": 1,
  "segment": "Premium Shoppers",
  "recommendation": "High income and high spending. Offer loyalty rewards, VIP events and premium product launches.",
  "input": {"annual_income": 90.0, "spending_score": 85.0}
}
```
Invalid input (missing, non-numeric or out-of-range) returns HTTP `400` with
`{"error": "..."}`. `annual_income` must be 0-1000 (k$) and `spending_score` 1-100.

## Deploy to the cloud (Render - free tier)

1. Push this folder to a GitHub repository.
2. On https://render.com choose **New -> Web Service** and connect the repo.
3. Render reads `render.yaml` automatically
   (build: `pip install -r requirements.txt && python train_model.py`,
   start: `gunicorn app:app`, health check: `/health`).
4. After the build, your live URL is shown on the dashboard.

*Note on Live Deployment:* The application is fully prepared and deployment-ready (`Procfile`, `render.yaml`, environment port binding), but it has **not** been deployed to a live public URL yet.

Railway, Heroku and PythonAnywhere also work through the `Procfile`
(`web: gunicorn app:app`).

## Methodology highlights

| Question | Finding |
|----------|---------|
| Which algorithm? | K-Means, Ward hierarchical and Gaussian Mixture tie (silhouette 0.554-0.555). DBSCAN scores 0.558 but discards 23 customers as noise; single linkage is poor (0.276). K-Means was chosen for fit, simplicity and `predict()` support. |
| Which features? | Income + Spending is best (0.555). Adding Age gives 0.417, adding Gender too gives 0.304. |
| Does scaling matter? | Not here: both features have a std of about 26, so scaled and raw give identical segments. The scaler stays in the pipeline as a safeguard. |
| Is it stable? | Yes: identical segments over 10 random seeds (ARI = 1.0). |

## Bias and ethics (summary - full discussion in the report, Section 6)

- Gender is **not** a model feature; segments were tested and show no evidence of gender dependence (chi-square p = 0.693).
- Age is not a feature either, but it shapes the segments (Impulsive Spenders are 18-35). Segments must **never** be used for pricing, credit or deciding who is served.
- The Spending Score formula is unknown and the 200-customer sample is undocumented, so results should not be generalised.
- The API does not store the values sent to `/predict`.

## Limitations

- Only two features (income, spending score) are used; Age and Gender are kept
  for profiling but not for clustering.
- The dataset has just 200 customers, so segments may shift on larger data.
- K-Means assumes roughly round clusters; this dataset fits that assumption, but
  other data may not. `StandardScaler` is kept in the pipeline for robustness.
- The model is unsupervised, so supervised classification metrics such as **accuracy**, **precision**, **recall**, **F1-score**, **confusion matrix**, and **ROC-AUC** do not apply without ground-truth labels and were not calculated. Clustering quality is evaluated using intrinsic geometric validation (**silhouette score** and **Davies-Bouldin index**) alongside domain-specific business interpretation.
