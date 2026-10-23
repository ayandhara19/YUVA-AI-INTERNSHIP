"""
Week 4 - Customer Segmentation: data preprocessing, model training,
evaluation and model serialization.

Run:  python train_model.py
Produces:
    models/customer_segmentation.joblib   (scaler + K-Means pipeline)
    models/metadata.json                  (features, cluster names, metrics)
    outputs/customer_segments.csv         (dataset with assigned segment)
    outputs/cluster_profiles.csv          (average profile of each segment)
    outputs/cluster_visualization.png     (scatter plot of the segments)
    outputs/elbow_silhouette.png          (how K was selected)
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Mall_Customers.csv"
MODEL_PATH = BASE_DIR / "models" / "customer_segmentation.joblib"
META_PATH = BASE_DIR / "models" / "metadata.json"
OUT_DIR = BASE_DIR / "outputs"

FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42


# ----------------------------------------------------------------- 1. data
def load_and_clean(path=DATA_PATH):
    """Load the CSV, validate it and remove unusable rows."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["CustomerID", "Gender", "Age"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    before = len(df)
    df = df.drop_duplicates(subset="CustomerID").dropna(subset=required)
    df["Gender"] = df["Gender"].str.strip().str.title()
    df = df[df["Annual Income (k$)"].between(0, 1000)]
    df = df[df["Spending Score (1-100)"].between(1, 100)]
    df = df[df["Age"].between(10, 100)].reset_index(drop=True)
    print(f"[data] {before} rows loaded, {len(df)} rows kept after cleaning")
    return df


# ------------------------------------------------------------ 2. model sel.
def select_k(X_scaled, k_range=range(2, 11)):
    """Evaluate K with inertia (elbow) and silhouette score."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, km.labels_),
        })
    return pd.DataFrame(rows)


def plot_k_selection(scores, chosen_k):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(scores["k"], scores["inertia"], marker="o", color="#2563eb")
    ax[0].axvline(chosen_k, ls="--", color="#dc2626")
    ax[0].set(title="Elbow method", xlabel="Number of clusters (K)", ylabel="Inertia")
    ax[1].plot(scores["k"], scores["silhouette"], marker="o", color="#059669")
    ax[1].axvline(chosen_k, ls="--", color="#dc2626")
    ax[1].set(title="Silhouette score", xlabel="Number of clusters (K)", ylabel="Score")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "elbow_silhouette.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ 3. naming
def name_clusters(centers):
    """Give each cluster a business-friendly name from its centroid
    (centers are in original units: income, spending)."""
    inc_mid = centers[:, 0].mean()
    sp_mid = centers[:, 1].mean()
    names = {}
    for i, (inc, sp) in enumerate(centers):
        hi_inc, hi_sp = inc >= inc_mid, sp >= sp_mid
        if hi_inc and hi_sp:
            names[i] = "Premium Shoppers"
        elif hi_inc and not hi_sp:
            names[i] = "Careful Spenders"
        elif not hi_inc and hi_sp:
            names[i] = "Impulsive Spenders"
        else:
            names[i] = "Budget Customers"
    # the 5th group sits near the average of both axes
    for i, (inc, sp) in enumerate(centers):
        if abs(inc - inc_mid) < 0.25 * centers[:, 0].std() * 2 and \
           abs(sp - sp_mid) < 0.25 * centers[:, 1].std() * 2:
            names[i] = "Average Shoppers"
    return names


# ------------------------------------------------------------ 4. plotting
def plot_clusters(df, pipeline, names):
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
              "#0891b2", "#be185d", "#4d7c0f"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(df["Cluster"].unique()):
        sub = df[df["Cluster"] == c]
        ax.scatter(sub[FEATURES[0]], sub[FEATURES[1]], s=45, alpha=.8,
                   color=colors[c % len(colors)], label=f"{c}: {names[c]}")
    centers = pipeline.named_steps["scaler"].inverse_transform(
        pipeline.named_steps["kmeans"].cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=220, color="black",
               edgecolor="white", label="Centroids")
    ax.set(title="Mall Customer Segments (K-Means)",
           xlabel=FEATURES[0], ylabel=FEATURES[1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8, frameon=False)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_visualization.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------ 5. main
def main():
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)

    df = load_and_clean()
    X = df[FEATURES]

    # choose K using the silhouette score on scaled data
    X_scaled = StandardScaler().fit_transform(X)
    scores = select_k(X_scaled)
    print(scores.round(3).to_string(index=False))
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])
    print(f"[model] best silhouette at K={best_k}")
    chosen_k = best_k
    print(f"[model] chosen K = {chosen_k}")
    plot_k_selection(scores, chosen_k)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=chosen_k, n_init=10, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X)
    labels = pipeline.predict(X)
    df["Cluster"] = labels

    sil = float(silhouette_score(pipeline.named_steps["scaler"].transform(X), labels))
    dbi = float(davies_bouldin_score(pipeline.named_steps["scaler"].transform(X), labels))
    print(f"[eval] silhouette={sil:.3f}  davies-bouldin={dbi:.3f}")

    centers = pipeline.named_steps["scaler"].inverse_transform(
        pipeline.named_steps["kmeans"].cluster_centers_)
    names = name_clusters(centers)
    df["Segment"] = df["Cluster"].map(names)

    profiles = (df.groupby(["Cluster", "Segment"])
                  .agg(Customers=("CustomerID", "count"),
                       Avg_Age=("Age", "mean"),
                       Avg_Income_k=(FEATURES[0], "mean"),
                       Avg_Spending_Score=(FEATURES[1], "mean"))
                  .round(1).reset_index())
    print(profiles.to_string(index=False))

    # ---- save outputs
    df.to_csv(OUT_DIR / "customer_segments.csv", index=False)
    profiles.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)
    plot_clusters(df, pipeline, names)

    # ---- serialize model (Joblib) + metadata
    joblib.dump(pipeline, MODEL_PATH)
    meta = {
        "model": "StandardScaler + KMeans",
        "features": FEATURES,
        "n_clusters": chosen_k,
        "n_samples": int(len(df)),
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(dbi, 4),
        "cluster_names": {str(k): v for k, v in names.items()},
        "cluster_profiles": profiles.to_dict(orient="records"),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[save] model -> {MODEL_PATH.relative_to(BASE_DIR)}")
    print(f"[save] metadata -> {META_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
