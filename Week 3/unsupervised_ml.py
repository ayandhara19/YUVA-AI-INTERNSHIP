"""
Week 3: Unsupervised Learning & Model Evaluation
Project: Customer Segmentation using K-Means and Hierarchical Clustering

Run:
    pip install -r requirements.txt
    python unsupervised_ml.py

The included CSV is synthetic demo data created for learning and demonstration.
It is not a real customer dataset.
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

DATA_PATH = ROOT / "customer_segmentation_demo.csv"
FEATURES = ["Age", "AnnualIncome_kUSD", "SpendingScore_1_100"]


def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES].copy()

    # Standardize features so variables with larger numeric scales do not dominate.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow curve and silhouette score for several possible cluster counts.
    inertias, silhouette_scores = [], []
    candidate_k = range(2, 8)
    for k in candidate_k:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    elbow_df = pd.DataFrame({
        "k": list(candidate_k),
        "inertia": inertias,
        "silhouette_score": silhouette_scores
    })
    elbow_df.to_csv(OUTPUTS / "cluster_selection_metrics.csv", index=False)

    # Choose the k with the best silhouette score among tested values.
    # This is a simple demonstration; interpret clusters using domain context too.
    best_k = int(elbow_df.loc[elbow_df["silhouette_score"].idxmax(), "k"])
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    data["KMeansCluster"] = kmeans.fit_predict(X_scaled)

    # Hierarchical clustering with the same number of clusters for a basic comparison.
    hierarchical = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    data["HierarchicalCluster"] = hierarchical.fit_predict(X_scaled)

    kmeans_silhouette = silhouette_score(X_scaled, data["KMeansCluster"])
    hierarchical_silhouette = silhouette_score(X_scaled, data["HierarchicalCluster"])

    comparison = pd.DataFrame([
        {"model": "K-Means", "clusters": best_k, "silhouette_score": kmeans_silhouette},
        {"model": "Hierarchical (Agglomerative)", "clusters": best_k,
         "silhouette_score": hierarchical_silhouette}
    ])
    comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    data.to_csv(OUTPUTS / "customer_segments.csv", index=False)

    # Cluster summaries make the results easier to interpret.
    summary = data.groupby("KMeansCluster")[FEATURES].mean().round(2)
    summary.to_csv(OUTPUTS / "kmeans_cluster_summary.csv")

    # Elbow plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(list(candidate_k), inertias, marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "elbow_method.png", dpi=160)
    plt.close()

    # Silhouette plot by candidate k
    plt.figure(figsize=(7, 4.5))
    plt.plot(list(candidate_k), silhouette_scores, marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "silhouette_scores.png", dpi=160)
    plt.close()

    # PCA projection to 2 dimensions for visualization only.
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                          c=data["KMeansCluster"], cmap="viridis", alpha=0.8)
    plt.title(f"Customer Segments (K-Means, k={best_k})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "customer_clusters_pca.png", dpi=160)
    plt.close()

    print("Finished. Files saved in:", OUTPUTS)
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nK-Means cluster profile (feature averages):")
    print(summary)
    print("\nNote: precision, recall, F1-score, confusion matrix and ROC-AUC are")
    print("classification metrics and are not directly suitable for unlabeled clustering.")
    print("For this project, silhouette score and inertia are used for clustering evaluation.")


if __name__ == "__main__":
    main()
