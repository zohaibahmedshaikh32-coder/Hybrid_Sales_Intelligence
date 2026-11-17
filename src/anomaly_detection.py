"""
anomaly_detection.py

Three unsupervised anomaly-detection models + business classification
+ evaluation + persistence, on the Retail Intelligence dataset.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

from src.config import PROCESSED_DIR, BASE_DIR, RANDOM_STATE

log = logging.getLogger(__name__)

MODELS_DIR  = BASE_DIR / "models" / "anomaly_detection"
REPORTS_DIR = BASE_DIR / "reports" / "anomaly_detection"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "Quantity", "Price", "Revenue",
    "AbsQuantity", "AbsRevenue",
    "qty_zscore", "price_zscore",
    "Hour", "DayOfWeek", "IsReturn",
]

ANOMALY_PCT         = 0.02
SILHOUETTE_SAMPLE   = 20_000
HIERARCHICAL_SAMPLE = 15_000
DENDROGRAM_SAMPLE   = 500


# ---------------------------------------------------------------------------
# Safe silhouette helper — handles degenerate single-cluster cases
# ---------------------------------------------------------------------------
def _safe_silhouette(X: np.ndarray,
                     labels: np.ndarray,
                     sample_size: int = 20_000,
                     random_state: int = RANDOM_STATE) -> float:
    """
    Silhouette with two safeguards:
      1. If < 2 distinct labels → return -1.0.
      2. Stratified per-cluster subsampling (tiny clusters still represented).
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return -1.0

    rng = np.random.default_rng(random_state)

    if len(X) > sample_size:
        clusters, counts = np.unique(labels, return_counts=True)
        per_cluster_min = 2
        base_allocation = per_cluster_min * len(clusters)
        remaining = max(0, sample_size - base_allocation)
        weights = counts / counts.sum()
        alloc = np.maximum(
            per_cluster_min,
            (weights * remaining).astype(int) + per_cluster_min,
        )
        alloc = np.minimum(alloc, counts)

        chosen = []
        for c, n in zip(clusters, alloc):
            idx = np.where(labels == c)[0]
            if len(idx) > n:
                idx = rng.choice(idx, size=int(n), replace=False)
            chosen.append(idx)
        idx = np.concatenate(chosen)
    else:
        idx = np.arange(len(X))

    if len(np.unique(labels[idx])) < 2:
        return -1.0

    return float(silhouette_score(X[idx], labels[idx]))


def _safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin, guarded for the single-cluster case."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(davies_bouldin_score(X, labels))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_anomaly_features() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "anomaly_features.parquet")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in anomaly_features.parquet: {missing}\n"
            f"Re-run run_eda.py or update FEATURE_COLS.")
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    log.info("Loaded anomaly features: %s", df.shape)
    return df


def _standardise(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].astype(float).values
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------------
# Model 1 — K-Means
# ---------------------------------------------------------------------------
@dataclass
class KMeansResult:
    pipeline:   Pipeline
    k:          int
    labels:     np.ndarray
    distances:  np.ndarray
    anomaly:    np.ndarray
    metrics:    dict = field(default_factory=dict)
    elbow:      dict = field(default_factory=dict)


def _kmeans_elbow(X: np.ndarray, k_range: Sequence[int]) -> dict:
    out = {"k": [], "inertia": [], "silhouette": []}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        out["k"].append(int(k))
        out["inertia"].append(float(km.inertia_))
        sil = _safe_silhouette(X, labels)
        out["silhouette"].append(sil)
        log.info("KMeans k=%d  inertia=%.2f  silhouette=%.4f",
                 k, km.inertia_, sil)
    return out


def _kmeans_choose_k(elbow: dict) -> int:
    sils = np.array(elbow["silhouette"])
    ks   = np.array(elbow["k"])
    inertia = np.array(elbow["inertia"])

    # Degenerate silhouette (>0.9) suggests one tiny cluster swallowing
    # the outliers — prefer the elbow heuristic instead.
    if sils.max() > 0.90:
        log.info("Silhouette degenerate (max=%.3f>0.90); "
                 "using elbow heuristic instead.", sils.max())
        drops = -np.diff(inertia)
        return int(ks[np.argmax(drops) + 1])

    valid = sils > -0.5
    if not valid.any():
        return int(ks[len(ks) // 2])
    best_score = sils[valid].max()
    candidates = ks[valid & (sils >= best_score - 0.02)]
    return int(candidates.min())


def train_kmeans(df: pd.DataFrame,
                 k_range: Sequence[int] = (2, 3, 4, 5, 6, 7),
                 anomaly_pct: float = ANOMALY_PCT) -> KMeansResult:
    X = _standardise(df)
    elbow = _kmeans_elbow(X, k_range)
    k = _kmeans_choose_k(elbow)
    log.info("KMeans chosen k = %d", k)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("km", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)),
    ])
    pipe.fit(df[FEATURE_COLS].astype(float).values)

    labels = pipe.predict(df[FEATURE_COLS].astype(float).values)
    scaler = pipe.named_steps["scaler"]
    km     = pipe.named_steps["km"]

    X_scaled  = scaler.transform(df[FEATURE_COLS].astype(float).values)
    centroids = km.cluster_centers_
    distances = np.linalg.norm(X_scaled - centroids[labels], axis=1)

    threshold = np.quantile(distances, 1 - anomaly_pct)
    anomaly   = (distances >= threshold).astype(int)

    metrics = {
        "k":                  k,
        "silhouette":         _safe_silhouette(X_scaled, labels),
        "davies_bouldin":     _safe_davies_bouldin(X_scaled, labels),
        "anomaly_pct":        float(anomaly.mean() * 100),
        "threshold_distance": float(threshold),
    }
    log.info("KMeans metrics: %s", metrics)

    return KMeansResult(pipe, k, labels, distances, anomaly,
                        metrics, elbow)


# ---------------------------------------------------------------------------
# Model 2 — Hierarchical Agglomerative Clustering
# ---------------------------------------------------------------------------
@dataclass
class HierarchicalResult:
    model:         AgglomerativeClustering
    labels:        np.ndarray
    distances:     np.ndarray
    anomaly:       np.ndarray
    metrics:       dict = field(default_factory=dict)
    linkage_array: np.ndarray | None = None


def train_hierarchical(
    df: pd.DataFrame,
    n_clusters: int = 4,
    linkage_method: str = "ward",
    anomaly_pct: float = ANOMALY_PCT,
) -> HierarchicalResult:
    X = _standardise(df)

    rng = np.random.default_rng(RANDOM_STATE)
    if len(df) > HIERARCHICAL_SAMPLE:
        sub_idx = rng.choice(len(df), HIERARCHICAL_SAMPLE, replace=False)
    else:
        sub_idx = np.arange(len(df))
    X_sub = X[sub_idx]

    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    linkage=linkage_method)
    model.fit(X_sub)
    sub_labels = model.labels_

    centroids = np.vstack([
        X_sub[sub_labels == c].mean(axis=0)
        for c in range(n_clusters)
    ])

    dists_to_centroids = np.linalg.norm(
        X[:, None, :] - centroids[None, :, :], axis=2)
    labels    = dists_to_centroids.argmin(axis=1)
    distances = dists_to_centroids.min(axis=1)

    threshold = np.quantile(distances, 1 - anomaly_pct)
    anomaly   = (distances >= threshold).astype(int)

    if len(X_sub) > DENDROGRAM_SAMPLE:
        di = rng.choice(len(X_sub), DENDROGRAM_SAMPLE, replace=False)
        X_dendro = X_sub[di]
    else:
        X_dendro = X_sub
    Z = linkage(X_dendro, method=linkage_method)

    metrics = {
        "n_clusters":         n_clusters,
        "linkage":            linkage_method,
        "silhouette":         _safe_silhouette(X_sub, sub_labels),
        "davies_bouldin":     _safe_davies_bouldin(X_sub, sub_labels),
        "anomaly_pct":        float(anomaly.mean() * 100),
        "threshold_distance": float(threshold),
    }
    log.info("Hierarchical metrics: %s", metrics)

    return HierarchicalResult(model, labels, distances, anomaly,
                              metrics, Z)


# ---------------------------------------------------------------------------
# Model 3 — Isolation Forest
# ---------------------------------------------------------------------------
@dataclass
class IsolationForestResult:
    model:   IsolationForest
    scores:  np.ndarray
    anomaly: np.ndarray
    metrics: dict = field(default_factory=dict)


def train_isolation_forest(
    df: pd.DataFrame,
    contamination: float = ANOMALY_PCT,
) -> IsolationForestResult:
    X = _standardise(df)
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X)
    scores  = -model.score_samples(X)
    labels  = model.predict(X)
    anomaly = (labels == -1).astype(int)

    metrics = {
        "contamination": float(contamination),
        "anomaly_pct":   float(anomaly.mean() * 100),
        "score_p50":     float(np.percentile(scores, 50)),
        "score_p99":     float(np.percentile(scores, 99)),
    }
    log.info("IsolationForest metrics: %s", metrics)

    return IsolationForestResult(model, scores, anomaly, metrics)


# ---------------------------------------------------------------------------
# Business-rule classification
# ---------------------------------------------------------------------------
def classify_anomaly(row: pd.Series) -> str:
    if row.get("IsReturn", 0) == 1 and row.get("AbsRevenue", 0) > 500:
        return "Unusual return / cancellation value"
    if row.get("qty_zscore", 0) > 3:
        return "Unusually high quantity"
    if row.get("qty_zscore", 0) < -3:
        return "Unusually low quantity"
    if row.get("price_zscore", 0) > 3:
        return "Unusually high unit price"
    if row.get("price_zscore", 0) < -3:
        return "Unusually low unit price"
    if row.get("AbsRevenue", 0) > 1_000:
        return "Unusual revenue"
    return "Unclassified (flagged by model only)"


def attach_classifications(df: pd.DataFrame,
                           anomaly_mask: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["anomaly_flag"] = anomaly_mask
    out["anomaly_category"] = np.where(
        anomaly_mask == 1,
        out.apply(classify_anomaly, axis=1),
        "normal",
    )
    return out


# ---------------------------------------------------------------------------
# Inter-model agreement
# ---------------------------------------------------------------------------
def agreement_matrix(*masks: np.ndarray) -> pd.DataFrame:
    names = ["KMeans", "Hierarchical", "IsolationForest"][:len(masks)]
    n = len(masks)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            a, b = masks[i], masks[j]
            inter = np.logical_and(a == 1, b == 1).sum()
            union = np.logical_or(a == 1, b == 1).sum()
            m[i, j] = inter / union if union else 0.0
    return pd.DataFrame(m, index=names, columns=names).round(3)


def agreement_counts(*masks: np.ndarray) -> pd.Series:
    stacked = np.vstack(masks).sum(axis=0)
    return pd.Series(stacked).value_counts().sort_index()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def write_metrics(results: dict) -> pd.DataFrame:
    rows = []
    for name in ("KMeans", "Hierarchical", "IsolationForest"):
        m = results[name]["metrics"].copy()
        m = {"model": name, **m}
        rows.append(m)
    df = pd.DataFrame(rows)
    path = REPORTS_DIR / "metrics.csv"
    df.to_csv(path, index=False)
    log.info("Saved metrics -> %s", path)
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = REPORTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure -> %s", path)


def plot_elbow_and_silhouette(km: KMeansResult) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(km.elbow["k"], km.elbow["inertia"], marker="o")
    axes[0].set_title("K-Means — Elbow (inertia vs k)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("inertia (SSE)")
    axes[0].axvline(km.k, color="red", ls="--",
                    label=f"chosen k = {km.k}")
    axes[0].legend()

    axes[1].plot(km.elbow["k"], km.elbow["silhouette"], marker="o",
                 color="#10B981")
    axes[1].set_title("K-Means — Silhouette vs k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("silhouette score")
    axes[1].axvline(km.k, color="red", ls="--")
    fig.tight_layout()
    _save(fig, "01_elbow_silhouette")


def plot_kmeans_clusters_pca(df: pd.DataFrame, km: KMeansResult) -> None:
    X = _standardise(df)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(Z), size=min(10_000, len(Z)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].scatter(Z[idx, 0], Z[idx, 1],
                    c=km.labels[idx], cmap="tab10",
                    s=4, alpha=0.4)
    axes[0].set_title(f"K-Means clusters (k={km.k}) — PCA 2D")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    normal = idx[km.anomaly[idx] == 0]
    anomal = idx[km.anomaly[idx] == 1]
    axes[1].scatter(Z[normal, 0], Z[normal, 1],
                    s=4, alpha=0.25, color="#94A3B8", label="normal")
    axes[1].scatter(Z[anomal, 0], Z[anomal, 1],
                    s=6, alpha=0.7, color="#EF4444", label="anomaly")
    axes[1].set_title(f"K-Means anomalies ({len(anomal)} shown)")
    axes[1].legend()
    fig.tight_layout()
    _save(fig, "02_kmeans_clusters_pca")


def plot_dendrogram(hc: HierarchicalResult) -> None:
    if hc.linkage_array is None:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(hc.linkage_array, truncate_mode="lastp", p=30,
               ax=ax, no_labels=True, color_threshold=0)
    ax.set_title("Hierarchical clustering — Ward dendrogram "
                 "(subsample, truncated to 30 leaves)")
    ax.set_ylabel("Ward linkage distance")
    fig.tight_layout()
    _save(fig, "03_dendrogram")


def plot_isolation_scores(if_res: IsolationForestResult) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    scores = if_res.scores
    anomaly = if_res.anomaly
    ax.hist(scores[anomaly == 0], bins=100, alpha=0.7,
            label="normal", color="#2563EB")
    ax.hist(scores[anomaly == 1], bins=100, alpha=0.9,
            label="anomaly", color="#EF4444")
    ax.set_title("Isolation Forest — anomaly score distribution")
    ax.set_xlabel("anomaly score (higher = more anomalous)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "04_isolation_scores")


def plot_agreement(matrix: pd.DataFrame, counts: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Blues",
                vmin=0, vmax=1, ax=axes[0], square=True,
                cbar_kws={"shrink": 0.7})
    axes[0].set_title("Jaccard agreement between models")

    sns.barplot(x=counts.index, y=counts.values, ax=axes[1],
                palette="Reds")
    axes[1].set_title("How many models flagged each row")
    axes[1].set_xlabel("# models flagging")
    axes[1].set_ylabel("# transactions")
    fig.tight_layout()
    _save(fig, "05_agreement_matrix")


def plot_categories(df: pd.DataFrame) -> None:
    flagged = df[df["anomaly_flag"] == 1]
    if flagged.empty:
        return
    counts = flagged["anomaly_category"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax,
                palette="viridis")
    ax.set_title("Business categories of flagged anomalies")
    ax.set_xlabel("# flagged transactions")
    fig.tight_layout()
    _save(fig, "06_anomaly_categories")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_models(km: KMeansResult,
                hc: HierarchicalResult,
                if_res: IsolationForestResult,
                metrics_df: pd.DataFrame) -> None:
    joblib.dump(km.pipeline, MODELS_DIR / "kmeans.joblib")
    joblib.dump({"model": hc.model, "metrics": hc.metrics},
                MODELS_DIR / "hierarchical.joblib")
    joblib.dump(if_res.model, MODELS_DIR / "isolation_forest.joblib")

    metadata = {
        "feature_cols":       FEATURE_COLS,
        "schema_version":     1,
        "chosen_kmeans_k":    km.k,
        "anomaly_pct_target": ANOMALY_PCT,
        "metrics":            metrics_df.to_dict(orient="records"),
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    log.info("Saved anomaly models -> %s", MODELS_DIR)


def load_models() -> dict:
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    return {
        "feature_cols":     meta["feature_cols"],
        "kmeans":           joblib.load(MODELS_DIR / "kmeans.joblib"),
        "hierarchical":     joblib.load(MODELS_DIR / "hierarchical.joblib"),
        "isolation_forest": joblib.load(MODELS_DIR / "isolation_forest.joblib"),
        "metadata":         meta,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_training_pipeline(verbose: bool = True) -> dict:
    df = load_anomaly_features()

    log.info("Training K-Means ...")
    km = train_kmeans(df)

    log.info("Training Hierarchical ...")
    hc = train_hierarchical(df)

    log.info("Training Isolation Forest ...")
    if_res = train_isolation_forest(df)

    matrix = agreement_matrix(km.anomaly, hc.anomaly, if_res.anomaly)
    counts = agreement_counts(km.anomaly, hc.anomaly, if_res.anomaly)
    print("\nINTER-MODEL AGREEMENT (Jaccard)")
    print(matrix.to_string())
    print("\nFlag counts (0/1/2/3 models):")
    print(counts.to_string())

    # Attach per-row anomaly score BEFORE filtering, so the assignment
    # length matches the full DataFrame (1,046,446 rows), not the
    # filtered subset (20,929 rows).
    classified = attach_classifications(df, km.anomaly)
    classified = classified.assign(_score=km.distances)

    top = (classified[classified["anomaly_flag"] == 1]
           .sort_values("_score", ascending=False)
           .head(100))
    top.to_csv(REPORTS_DIR / "top_anomalies_sample.csv", index=False)

    metrics_df = write_metrics({
        "KMeans":          {"metrics": km.metrics},
        "Hierarchical":    {"metrics": hc.metrics},
        "IsolationForest": {"metrics": if_res.metrics},
    })
    plot_elbow_and_silhouette(km)
    plot_kmeans_clusters_pca(df, km)
    plot_dendrogram(hc)
    plot_isolation_scores(if_res)
    plot_agreement(matrix, counts)
    plot_categories(classified)
    save_models(km, hc, if_res, metrics_df)

    if verbose:
        print("\n" + "=" * 72)
        print("ANOMALY DETECTION SUMMARY")
        print("=" * 72)
        for r in (km, hc, if_res):
            name = type(r).__name__.replace("Result", "")
            print(f"{name:20s}  anomaly%={r.metrics['anomaly_pct']:.2f}")
        print("=" * 72)

    return {
        "kmeans":           km,
        "hierarchical":     hc,
        "isolation_forest": if_res,
        "agreement_matrix": matrix,
        "agreement_counts": counts,
        "metrics":          metrics_df,
        "classified":       classified,
    }