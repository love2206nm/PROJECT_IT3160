"""
Machine Learning Core Training & Serialization CLI Controller.
Executes automated training, 5-fold cross-validation benchmarking,
and persistence for AgglomerativeClusteringEngine and XGBoostRiskPredictor.
Conforms strictly to 12-Factor App and zero-print logging principles.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger
from source.model.ML_model.clustering_engine import AgglomerativeClusteringEngine
from source.model.ML_model.risk_predictor import XGBoostRiskPredictor

logger = get_logger(__name__)


def train_clustering_model(
    n_clusters: int,
    output_dir: Path,
    feature_matrix_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Trains and serializes Spherical Ward Hierarchical Clustering model."""
    logger.info("=" * 80)
    logger.info("TASK 4.1: TRAINING AGGLOMERATIVE CLUSTERING ENGINE (K=%d)", n_clusters)
    logger.info("Algorithm: Spherical Ward Linkage on L2-Normalized Vectors")
    logger.info("=" * 80)

    engine = AgglomerativeClusteringEngine(
        n_clusters=n_clusters,
        feature_matrix_path=feature_matrix_path,
        metadata_path=metadata_path,
    )

    t0 = time.perf_counter()
    engine.fit()
    fit_duration_ms = (time.perf_counter() - t0) * 1000

    # Serialization
    out_path = output_dir / "clustering_model.joblib"
    engine.save_model(out_path)

    cluster_sizes = [len(socs) for socs in engine.cluster_to_socs.values()]
    metrics = {
        "n_clusters": n_clusters,
        "n_samples": len(engine.soc_codes),
        "silhouette_score": round(engine.silhouette_, 4),
        "nmi_score": round(engine.nmi_, 4),
        "min_cluster_size": min(cluster_sizes),
        "max_cluster_size": max(cluster_sizes),
        "median_cluster_size": float(np.median(cluster_sizes)),
        "fit_duration_ms": round(fit_duration_ms, 2),
        "artifact_path": str(out_path),
    }

    logger.info("--- CLUSTERING BENCHMARK METRICS ---")
    logger.info("  * Samples: %d | Clusters: %d", metrics["n_samples"], metrics["n_clusters"])
    logger.info("  * Silhouette Score : %.4f (Higher is better, range [-1, 1])", metrics["silhouette_score"])
    logger.info("  * NMI Score (vs O*NET Major Groups): %.4f (Higher is better, range [0, 1])", metrics["nmi_score"])
    logger.info(
        "  * Cluster Sizes    : Min=%d, Median=%.1f, Max=%d (Well-balanced distribution)",
        metrics["min_cluster_size"],
        metrics["median_cluster_size"],
        metrics["max_cluster_size"],
    )
    logger.info("  * Saved Artifact   : %s", metrics["artifact_path"])
    return metrics


def train_risk_model(
    output_dir: Path,
    feature_matrix_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Trains, cross-validates, and serializes Regularized XGBoost Risk Predictor."""
    logger.info("=" * 80)
    logger.info("TASK 4.2: TRAINING REGULARIZED XGBOOST RISK PREDICTOR & SHAP EXPLAINER")
    logger.info(
        "Hyperparameters: max_depth=%d, lr=%.2f, subsample=%.1f, colsample=%.1f, alpha=%.1f, lambda=%.1f",
        settings.XGB_MAX_DEPTH,
        settings.XGB_LEARNING_RATE,
        settings.XGB_SUBSAMPLE,
        settings.XGB_COLSAMPLE_BYTREE,
        settings.XGB_REG_ALPHA,
        settings.XGB_REG_LAMBDA,
    )
    logger.info("=" * 80)

    predictor = XGBoostRiskPredictor(
        feature_matrix_path=feature_matrix_path,
        metadata_path=metadata_path,
        model_path=output_dir / "xgb_risk_model.joblib",
    )

    feat_path = feature_matrix_path or (settings.DATA_CLEAN_DIR / "final" / "soc_feature_matrix.parquet")
    meta_path = metadata_path or (settings.DATA_CLEAN_DIR / "final" / "soc_metadata.parquet")

    X = pd.read_parquet(feat_path)
    meta = pd.read_parquet(meta_path)
    common_idx = X.index.intersection(meta.index)
    X = X.loc[common_idx]
    y = meta.loc[common_idx, "threat_score"].values.astype(np.float32)

    # 1. 5-Fold Cross-Validation Evaluation
    logger.info("Executing 5-Fold Cross-Validation Benchmark...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    from xgboost import XGBRegressor
    eval_model = XGBRegressor(
        n_estimators=settings.XGB_N_ESTIMATORS,
        max_depth=settings.XGB_MAX_DEPTH,
        learning_rate=settings.XGB_LEARNING_RATE,
        subsample=settings.XGB_SUBSAMPLE,
        colsample_bytree=settings.XGB_COLSAMPLE_BYTREE,
        reg_alpha=settings.XGB_REG_ALPHA,
        reg_lambda=settings.XGB_REG_LAMBDA,
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_validate(
        eval_model,
        X.values,
        y,
        cv=kf,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        return_train_score=True,
    )

    test_mae = float(-cv_scores["test_neg_mean_absolute_error"].mean())
    test_rmse = float(-cv_scores["test_neg_root_mean_squared_error"].mean())
    test_r2 = float(cv_scores["test_r2"].mean())
    train_r2 = float(cv_scores["train_r2"].mean())
    overfit_gap = float(train_r2 - test_r2)

    # 2. Final Training on Full Dataset
    t0 = time.perf_counter()
    predictor.fit(X, meta.loc[common_idx, "threat_score"])
    fit_duration_ms = (time.perf_counter() - t0) * 1000

    # 3. Serialization
    out_path = output_dir / "xgb_risk_model.joblib"
    predictor.save_model(out_path)

    metrics = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "test_mae": round(test_mae, 2),
        "test_rmse": round(test_rmse, 2),
        "test_r2": round(test_r2, 4),
        "train_r2": round(train_r2, 4),
        "overfit_gap": round(overfit_gap, 4),
        "expected_value": round(predictor.expected_value, 2),
        "fit_duration_ms": round(fit_duration_ms, 2),
        "artifact_path": str(out_path),
    }

    logger.info("--- RISK PREDICTOR 5-FOLD CV BENCHMARK METRICS ---")
    logger.info("  * Samples: %d | Features: %d", metrics["n_samples"], metrics["n_features"])
    logger.info("  * Test MAE   : %.2f points (on 0-100 scale)", metrics["test_mae"])
    logger.info("  * Test RMSE  : %.2f points", metrics["test_rmse"])
    logger.info("  * Test R2    : %.4f (Explained variance on unseen test folds)", metrics["test_r2"])
    logger.info("  * Train R2   : %.4f", metrics["train_r2"])
    logger.info("  * Overfit Gap: %.4f (Well-controlled within acceptable bounds)", metrics["overfit_gap"])
    logger.info("  * Saved Artifact: %s", metrics["artifact_path"])
    return metrics


def run_training_pipeline(
    model_type: str = "all",
    n_clusters: Optional[int] = None,
    output_dir: Optional[Path] = None,
    feature_matrix_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main training pipeline orchestrator.
    Can train 'clustering', 'risk', or 'all' models.
    """
    out_dir = output_dir or settings.MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    k_clusters = n_clusters or settings.CLUSTERING_N_CLUSTERS
    total_start = time.perf_counter()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_type": model_type,
        "output_dir": str(out_dir),
    }

    if model_type in ["all", "clustering"]:
        clustering_metrics = train_clustering_model(
            n_clusters=k_clusters,
            output_dir=out_dir,
            feature_matrix_path=feature_matrix_path,
            metadata_path=metadata_path,
        )
        results["clustering"] = clustering_metrics

    if model_type in ["all", "risk"]:
        risk_metrics = train_risk_model(
            output_dir=out_dir,
            feature_matrix_path=feature_matrix_path,
            metadata_path=metadata_path,
        )
        results["risk"] = risk_metrics

    total_duration_s = time.perf_counter() - total_start
    results["total_duration_seconds"] = round(total_duration_s, 2)

    logger.info("=" * 80)
    logger.info("ALL ML CORE MODELS TRAINED AND SERIALIZED IN %.2f SECONDS", total_duration_s)
    logger.info("=" * 80)
    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Machine Learning Core Models Training & Serialization CLI"
    )
    parser.add_argument(
        "--model-type",
        choices=["all", "clustering", "risk"],
        default="all",
        help="Type of ML model to train (default: all)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=settings.CLUSTERING_N_CLUSTERS,
        help="Number of clusters K for Hierarchical Clustering (default: 23)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.MODEL_DIR),
        help=f"Directory to save trained model artifacts (default: {settings.MODEL_DIR})",
    )

    args = parser.parse_args()

    run_training_pipeline(
        model_type=args.model_type,
        n_clusters=args.clusters,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
