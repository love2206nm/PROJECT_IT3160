"""
XGBoost Risk Predictor & SHAP Explainer Adapter.
Implements IRiskPredictor protocol using Regularized XGBRegressor,
Deterministic Boundary Mapping, and TreeSHAP explainability for custom/hybrid roles.
"""

from pathlib import Path
import time
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
import shap

from source.domain.interfaces.ml_models import IRiskPredictor
from source.domain.entities.candidate import CandidateProfile
from source.domain.entities.risk import RiskProfile, RiskLevel, SubRiskScores
from source.domain.exceptions import ModelInferenceError
from source.infrastructure.config import settings
from source.infrastructure.logger import get_logger

logger = get_logger(__name__)


class XGBoostRiskPredictor(IRiskPredictor):
    """
    Production-grade AI Automation Risk Predictor.
    Uses Regularized Gradient Boosting (max_depth=3, L1/L2 penalties)
    to predict threat scores (0-100) and extract Top 3 defensive and vulnerable skills via TreeSHAP.
    """

    def __init__(
        self,
        feature_matrix_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
    ):
        self.feature_matrix_path = (
            feature_matrix_path
            or (settings.DATA_CLEAN_DIR / "final" / "soc_feature_matrix.parquet")
        )
        self.metadata_path = (
            metadata_path or (settings.DATA_CLEAN_DIR / "final" / "soc_metadata.parquet")
        )
        self.model_path = model_path or (settings.MODEL_DIR / "xgb_risk_model.joblib")

        self.model: Optional[XGBRegressor] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.feature_means: Dict[str, float] = {}
        self.expected_value: float = 23.06
        self.is_fitted: bool = False

    def fit(
        self,
        X_df: Optional[pd.DataFrame] = None,
        y_series: Optional[pd.Series] = None,
    ) -> "XGBoostRiskPredictor":
        """
        Trains Regularized XGBRegressor on occupational capability matrix against threat_score.
        """
        logger.info("Training Regularized XGBoost Risk Predictor...")
        start_time = time.perf_counter()

        # 1. Load Data if not provided
        if X_df is not None and y_series is not None:
            X = X_df.copy()
            y = y_series.values.astype(np.float32)
        else:
            if not self.feature_matrix_path.exists():
                raise ModelInferenceError(f"Feature matrix not found: '{self.feature_matrix_path}'")
            if not self.metadata_path.exists():
                raise ModelInferenceError(f"Metadata catalog not found: '{self.metadata_path}'")

            X = pd.read_parquet(self.feature_matrix_path)
            meta = pd.read_parquet(self.metadata_path)

            common_idx = X.index.intersection(meta.index)
            X = X.loc[common_idx]
            meta = meta.loc[common_idx]
            y = meta["threat_score"].values.astype(np.float32)

        self.feature_names = list(X.columns)
        self.feature_means = {col: float(X[col].mean()) for col in self.feature_names}
        self.expected_value = float(np.mean(y))

        # 2. Configure Regularized XGBRegressor
        self.model = XGBRegressor(
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

        logger.info(
            "Fitting XGBRegressor (depth=%d, lr=%.2f, alpha=%.1f, lambda=%.1f) on %d samples, %d features...",
            settings.XGB_MAX_DEPTH,
            settings.XGB_LEARNING_RATE,
            settings.XGB_REG_ALPHA,
            settings.XGB_REG_LAMBDA,
            len(X),
            len(self.feature_names),
        )
        self.model.fit(X.values, y)

        # 3. Initialize SHAP TreeExplainer
        logger.info("Initializing SHAP TreeExplainer on fitted XGBoost model...")
        self.explainer = shap.TreeExplainer(self.model)

        self.is_fitted = True
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Calculate Training R2
        train_preds = self.model.predict(X.values)
        train_r2 = float(1.0 - (np.sum((y - train_preds) ** 2) / np.sum((y - np.mean(y)) ** 2)))
        train_mae = float(np.mean(np.abs(y - train_preds)))

        logger.info(
            "XGBoost Risk Predictor trained in %.2f ms | Train R2: %.4f | Train MAE: %.2f | Mean Threat: %.2f",
            elapsed_ms,
            train_r2,
            train_mae,
            self.expected_value,
        )
        return self

    def predict_risk(self, profile: CandidateProfile) -> RiskProfile:
        """
        Infers multi-dimensional AI automation risk profile for custom/non-standard roles.
        Uses TreeSHAP to attribute feature contributions into defensive vs vulnerable skills.
        """
        if not self.is_fitted:
            # Attempt to load model from disk or train fresh
            if self.model_path.exists():
                self.load_model(self.model_path)
            else:
                logger.info("Model artifact not found. Fitting fresh XGBoost model...")
                self.fit()

        start_time = time.perf_counter()

        # 1. Vectorize CandidateProfile into 76D Capability Space
        vector = self.vectorize_profile(profile)

        # 2. Predict Threat Score (continuous 0-100)
        raw_pred = float(self.model.predict(vector.reshape(1, -1))[0])
        threat_score = round(float(np.clip(raw_pred, 0.0, 100.0)), 2)

        # 3. Compute SHAP Values for Explainability
        shap_values = self.explainer(vector.reshape(1, -1)).values[0]

        # Sort feature indices by SHAP values
        # Positive SHAP: Increases threat score (Vulnerability)
        # Negative SHAP: Decreases threat score (Defense / Resilience)
        sorted_indices = np.argsort(shap_values)

        defensive_indices = sorted_indices[:3]  # most negative
        vulnerable_indices = sorted_indices[::-1][:3]  # most positive

        defense_skills = [self.feature_names[i] for i in defensive_indices]
        vulnerable_skills = [self.feature_names[i] for i in vulnerable_indices]

        # 4. Map Threat Score to RiskLevel (Deterministic Boundary Mapping)
        risk_level = self.map_score_to_risk_level(threat_score)

        # 5. Compute Component Sub-Scores
        # Align with domain sub-metrics: task automation, tool maturity, agentic exposure
        sub_task_automation = round(float(np.clip(threat_score * 0.95, 0.0, 100.0)), 2)
        sub_tool_maturity = round(float(np.clip(threat_score * 1.05, 0.0, 100.0)), 2)
        sub_agentic_exposure = round(float(np.clip(threat_score * 0.90, 0.0, 100.0)), 2)

        # 6. Generate Contextual AI Automation Insight
        insight = (
            f"Role '{profile.current_title or 'Candidate'}' exhibits a {risk_level.value} threat score of {threat_score}/100. "
            f"Strongest defensive capabilities: {', '.join(defense_skills)}. "
            f"Primary automation vulnerabilities: {', '.join(vulnerable_skills)}."
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Risk prediction for '%s' completed in %.2f ms (Score: %.2f, Level: %s)",
            profile.full_name or "Candidate",
            elapsed_ms,
            threat_score,
            risk_level.value,
        )

        return RiskProfile(
            soc_code=None,
            role_title=profile.current_title or "Unspecified Candidate Role",
            threat_score=threat_score,
            risk_level=risk_level,
            sub_scores=SubRiskScores(
                task_automation=sub_task_automation,
                tool_maturity=sub_tool_maturity,
                agentic_exposure=sub_agentic_exposure,
            ),
            defense_skills=defense_skills,
            insight=insight,
            is_deterministic=False,
        )

    def vectorize_profile(self, profile: CandidateProfile) -> np.ndarray:
        """
        Maps unstructured CandidateProfile skills, tools, and experiences
        into the 76-dimensional O*NET capability feature vector.
        """
        # Combine all candidate skills, tools, experience text, projects, and certifications
        text_tokens = set()
        for s in profile.hard_skills:
            text_tokens.update(re.split(r"[\s,/_\\-]+", s.lower()))
        for t in profile.tools_and_technologies:
            text_tokens.update(re.split(r"[\s,/_\\-]+", t.lower()))
        for exp in profile.experiences:
            text_tokens.update(re.split(r"[\s,/_\\-]+", exp.role_title.lower()))
            if exp.description:
                text_tokens.update(re.split(r"[\s,/_\\-]+", exp.description.lower()))
            for ach in exp.key_achievements:
                text_tokens.update(re.split(r"[\s,/_\\-]+", ach.lower()))
        for proj in getattr(profile, "projects", []):
            text_tokens.update(re.split(r"[\s,/_\\-]+", proj.lower()))
        for cert in getattr(profile, "certifications", []):
            text_tokens.update(re.split(r"[\s,/_\\-]+", cert.lower()))
        for dom in getattr(profile, "domain_expertise", []):
            text_tokens.update(re.split(r"[\s,/_\\-]+", dom.lower()))

        full_profile_text = (
            " ".join(profile.hard_skills)
            + " "
            + " ".join(profile.tools_and_technologies)
            + " "
            + " ".join(exp.role_title for exp in profile.experiences)
            + " "
            + " ".join(getattr(profile, "projects", []))
            + " "
            + " ".join(getattr(profile, "certifications", []))
            + " "
            + " ".join(getattr(profile, "domain_expertise", []))
            + (" " + profile.raw_text[:2000].lower() if getattr(profile, "raw_text", None) else "")
        ).lower()

        # Experience scale multiplier
        exp_years = profile.years_of_experience
        exp_factor = min(1.2, 0.8 + (exp_years * 0.05))  # Boost for experienced professionals

        vector = np.zeros(len(self.feature_names), dtype=np.float32)

        for idx, feat_name in enumerate(self.feature_names):
            clean_feat = feat_name.lower().replace("_", " ")
            feat_tokens = [t for t in re.split(r"[\s,/_\\-]+", clean_feat) if len(t) > 2]

            baseline = self.feature_means.get(feat_name, 0.25)

            # Check direct match or token intersection
            match_score = 0.0
            if clean_feat in full_profile_text:
                match_score = 0.85
            else:
                matched_tokens = sum(1 for t in feat_tokens if t in text_tokens)
                if feat_tokens and matched_tokens > 0:
                    match_score = 0.50 + (0.35 * (matched_tokens / len(feat_tokens)))

            if match_score > 0.0:
                vector[idx] = min(1.0, max(baseline, match_score * exp_factor))
            else:
                # Default to baseline market expectation
                vector[idx] = max(0.10, baseline * 0.80)

        return vector

    @staticmethod
    def map_score_to_risk_level(score: float) -> RiskLevel:
        """
        Deterministic Boundary Mapping conforming to O*NET empirical percentiles.
        """
        if score < 30.0:
            return RiskLevel.LOW
        elif score < 60.0:
            return RiskLevel.MODERATE
        elif score < 80.0:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def save_model(self, file_path: Optional[Path] = None) -> Path:
        """Serializes trained XGBoost model and explainer metadata to disk."""
        if not self.is_fitted:
            raise ModelInferenceError("Cannot save unfitted XGBoost Risk Predictor.")

        path = file_path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "expected_value": self.expected_value,
        }
        joblib.dump(payload, path)
        logger.info("XGBoost Risk Predictor serialized successfully to '%s'.", path)
        return path

    def load_model(self, file_path: Optional[Path] = None) -> "XGBoostRiskPredictor":
        """Deserializes trained XGBoost model and explainer metadata from disk."""
        path = file_path or self.model_path
        if not path.exists():
            raise ModelInferenceError(f"Model artifact not found at '{path}'.")

        payload = joblib.load(path)
        self.model = payload["model"]
        self.feature_names = payload["feature_names"]
        self.feature_means = payload["feature_means"]
        self.expected_value = payload.get("expected_value", 23.06)

        # Re-initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        self.is_fitted = True

        logger.info(
            "XGBoost Risk Predictor loaded from '%s' (%d features, base value: %.2f).",
            path.name,
            len(self.feature_names),
            self.expected_value,
        )
        return self
