"""
Week 4 - Model comparison experiments (supports the methodology section of the report).

Part A: which ALGORITHM clusters this data best?   (same features, same scaling)
Part B: which FEATURES should be used?             (K-Means, K=5)
Part C: does SCALING matter?                       (K-Means, K=5)

Run:  python compare_models.py
Writes outputs/model_comparison.csv, outputs/feature_comparison.csv,
       outputs/model_comparison.png, outputs/segment_fairness_check.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from train_model import FEATURES, RANDOM_STATE, load_and_clean

OUT = Path(__file__).resolve().parent / "outputs"


def score(X, labels):
    """Return (silhouette, davies_bouldin, n_clusters, noise_points)."""
    mask = labels != -1                      # DBSCAN marks noise as -1
    n = len(set(labels[mask]))
    if n < 2:
        return np.nan, np.nan, n, int((~mask).sum())
    return (silhouette_score(X[mask], labels[mask]),
            davies_bouldin_score(X[mask], labels[mask]), n, int((~mask).sum()))


def main():
    df = load_and_clean()
    X = StandardScaler().fit_transform(df[FEATURES])

    # ---------------- Part A: algorithms
    models = {
        "K-Means (K=5)": KMeans(5, n_init=10, random_state=RANDOM_STATE),
        "Agglomerative / Ward (K=5)": AgglomerativeClustering(5, linkage="ward"),
        "Agglomerative / Single (K=5)": AgglomerativeClustering(5, linkage="single"),
        "Gaussian Mixture (K=5)": GaussianMixture(5, n_init=10, random_state=RANDOM_STATE),
        "DBSCAN (eps=0.35, min=5)": DBSCAN(eps=0.35, min_samples=5),
    }
    rows = []
    for name, m in models.items():
        labels = m.fit_predict(X)
        sil, dbi, n, noise = score(X, labels)
        rows.append({"Algorithm": name, "Clusters": n, "Noise points": noise,
                     "Silhouette": round(sil, 3), "Davies-Bouldin": round(dbi, 3)})
    algo = pd.DataFrame(rows).sort_values("Silhouette", ascending=False)
    print("\n== A. Algorithms ==\n", algo.to_string(index=False))
    algo.to_csv(OUT / "model_comparison.csv", index=False)

    # ---------------- Part B: feature sets
    sets = {
        "Income + Spending (used)": FEATURES,
        "Income + Spending + Age": FEATURES + ["Age"],
        "Age + Spending": ["Age", FEATURES[1]],
        "Age + Income": ["Age", FEATURES[0]],
        "All + Gender (encoded)": FEATURES + ["Age", "Gender_Male"],
    }
    d2 = df.assign(Gender_Male=(df["Gender"] == "Male").astype(int))
    rows = []
    for name, cols in sets.items():
        Xs = StandardScaler().fit_transform(d2[cols])
        labels = KMeans(5, n_init=10, random_state=RANDOM_STATE).fit_predict(Xs)
        sil, dbi, _, _ = score(Xs, labels)
        rows.append({"Feature set": name, "Silhouette": round(sil, 3), "Davies-Bouldin": round(dbi, 3)})
    feat = pd.DataFrame(rows).sort_values("Silhouette", ascending=False)
    print("\n== B. Feature sets (K-Means, K=5) ==\n", feat.to_string(index=False))
    feat.to_csv(OUT / "feature_comparison.csv", index=False)

    # ---------------- Part C: scaling
    raw = df[FEATURES].values
    lab_raw = KMeans(5, n_init=10, random_state=RANDOM_STATE).fit_predict(raw)
    lab_scl = KMeans(5, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    print("\n== C. Scaling ==")
    print("raw   silhouette (raw space)   :", round(silhouette_score(raw, lab_raw), 3))
    print("scaled silhouette (scaled space):", round(silhouette_score(X, lab_scl), 3))
    from sklearn.metrics import adjusted_rand_score
    print("agreement raw vs scaled (ARI)  :", round(adjusted_rand_score(lab_raw, lab_scl), 3))
    print("std of the two raw features    :", df[FEATURES].std().round(1).to_dict())

    # ---------------- Part D: stability across random seeds
    from sklearn.metrics import adjusted_rand_score as ari
    base = KMeans(5, n_init=10, random_state=0).fit_predict(X)
    aris = [ari(base, KMeans(5, n_init=10, random_state=s).fit_predict(X)) for s in range(1, 11)]
    print("\n== D. Stability over 10 seeds: mean ARI =", round(float(np.mean(aris)), 3))

    # ---------------- Part E: fairness / bias check on the final segments
    seg = pd.read_csv(OUT / "customer_segments.csv")
    g = pd.crosstab(seg["Segment"], seg["Gender"], normalize="index").mul(100).round(1)
    g["Customers"] = seg["Segment"].value_counts()
    g["Avg_Age"] = seg.groupby("Segment")["Age"].mean().round(1)
    g["Age_Min"] = seg.groupby("Segment")["Age"].min()
    g["Age_Max"] = seg.groupby("Segment")["Age"].max()
    print("\n== E. Gender / age make-up of each segment ==\n", g.to_string())
    print("overall gender %:", (seg["Gender"].value_counts(normalize=True) * 100).round(1).to_dict())
    g.reset_index().to_csv(OUT / "segment_fairness_check.csv", index=False)

    # chi-square test: is gender independent of segment?
    from scipy.stats import chi2_contingency
    chi = chi2_contingency(pd.crosstab(seg["Segment"], seg["Gender"]))
    print("chi-square gender vs segment: p =", round(chi[1], 3))

    # ---------------- chart
    fig, ax = plt.subplots(figsize=(8, 3.8))
    a = algo.dropna(subset=["Silhouette"]).sort_values("Silhouette")
    colors = ["#13A89E" if "K-Means" in n else "#94A3B8" for n in a["Algorithm"]]
    ax.barh(a["Algorithm"], a["Silhouette"], color=colors)
    for y, v in enumerate(a["Silhouette"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set(xlabel="Silhouette score (higher is better)", title="Algorithm comparison on the same scaled data")
    ax.set_xlim(0, max(a["Silhouette"]) + 0.08)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
