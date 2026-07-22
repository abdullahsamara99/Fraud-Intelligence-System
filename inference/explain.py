import joblib
import requests
import shap
import pandas as pd
from utils.logger import logger


class FraudExplainer:
    """
    Explains Isolation Forest predictions using SHAP and formats
    the output into business reasoning using an LLM layer.

    Sign convention (IsolationForest.decision_function):
        positive output  -> more NORMAL  -> a positive SHAP value lowers risk
        negative output  -> more ANOMALOUS -> a negative SHAP value raises risk
    """

    # Contributions below this absolute magnitude are treated as noise.
    # Tune to your score scale: a good starting point is ~10-20% of the
    # typical top-feature impact you observe on clearly anomalous samples.
    SIGNAL_FLOOR = 0.03

    def __init__(
        self,
        model_path="models/isolation_forest.pkl",
        preprocessor_path="models/scaler.pkl",
        llm_url="http://localhost:11434/api/generate",
        llm_model="qwen2.5:3b",
    ):
        logger.info("Loading SHAP explainer resources...")
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.explainer = None
        self.llm_url = llm_url
        self.llm_model = llm_model

    def build_explainer(self, background_data: pd.DataFrame):
        """
        Builds the SHAP explainer using a K-Means background summary.
        Extracts the raw NumPy array to avoid 'DenseData object is not callable'.
        """
        try:
            logger.info(f"Building SHAP explainer on {len(background_data)} samples...")
            X_background = self.preprocessor.transform(background_data)

            kmeans_res = shap.kmeans(X_background, 20)
            background_summary = (
                kmeans_res.data if hasattr(kmeans_res, "data") else kmeans_res
            )

            self.explainer = shap.KernelExplainer(
                self.model.decision_function,
                background_summary,
            )
            logger.info("SHAP explainer pre-built successfully.")
        except Exception:
            logger.exception("Failed to build SHAP explainer.")
            raise

    def explain(self, transaction: pd.DataFrame, feature_names: list, top_n: int = 5):
        """
        Compute SHAP attributions for a single transaction.

        Returns a list of dicts sorted by impact. Each dict has:
            feature, impact (abs), direction in
            {"Increase Risk", "Decrease Risk", "Negligible"}.

        A contribution whose magnitude is below SIGNAL_FLOOR is labeled
        "Negligible" so tiny, noisy attributions on near-baseline (normal)
        transactions are not presented as real risk drivers.
        """
        try:
            if self.explainer is None:
                raise RuntimeError("Explainer is not built. Call build_explainer first.")

            X = self.preprocessor.transform(transaction)
            shap_output = self.explainer.shap_values(X, nsamples=200)

            if isinstance(shap_output, list):
                shap_values = shap_output[0]
            elif hasattr(shap_output, "values"):
                shap_values = shap_output.values[0]
            else:
                shap_values = shap_output

            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            importance = []
            for feature, value in zip(feature_names, shap_values):
                val_float = float(value)
                magnitude = round(abs(val_float), 6)

                if magnitude < self.SIGNAL_FLOOR:
                    direction = "Negligible"
                elif val_float > 0:
                    direction = "Decrease Risk"
                else:
                    direction = "Increase Risk"

                importance.append({
                    "feature": feature,
                    "impact": magnitude,
                    "direction": direction,
                })

            importance.sort(key=lambda x: x["impact"], reverse=True)
            return importance[:top_n]

        except Exception:
            logger.exception("SHAP attribution computation failed.")
            raise

    def _risk_drivers(self, top_features: list) -> list:
        """Feature names that meaningfully push risk UP (above the floor)."""
        return [
            f["feature"]
            for f in top_features
            if f["direction"] == "Increase Risk" and f["impact"] >= self.SIGNAL_FLOOR
        ]

    def _fallback_summary(self, top_features: list, risk_level: str, action: str) -> str:
        """
        Deterministic explanation used when the LLM is unavailable.
        Branches on risk level so a Low-risk result never reads as 'abnormal'.
        """
        drivers = self._risk_drivers(top_features)

        if risk_level in ("Critical", "High") and drivers:
            return (
                f"Transaction flagged as {risk_level} risk, driven mainly by "
                f"{', '.join(drivers)}. Recommended action: {action}."
            )
        if risk_level == "Medium" and drivers:
            return (
                f"Transaction shows Medium risk with mild deviations in "
                f"{', '.join(drivers)}. Manual review is suggested."
            )
        # Low risk, or no feature cleared the signal floor
        return (
            f"Transaction assessed as {risk_level} risk. Behavior is consistent "
            f"with the customer's normal profile; no significant anomalies detected."
        )

    def generate_llm_summary(self, top_features: list, risk_level: str, action: str) -> str:
        """
        Translate SHAP outputs into an executive summary via the LLM,
        with a grounded rule-based fallback.
        """
        drivers = self._risk_drivers(top_features)
        has_signal = bool(drivers)

        # If nothing is anomalous, don't even ask the LLM to invent reasons.
        if not has_signal and risk_level in ("Low",):
            return self._fallback_summary(top_features, risk_level, action)

        features_text = ", ".join(
            f"{f['feature']} ({f['direction']}, impact={f['impact']})"
            for f in top_features
            if f["direction"] != "Negligible"
        ) or "no individually significant features"

        prompt = f"""
You are an expert Anti-Money Laundering (AML) and Fraud Risk Analyst at a bank.
Translate these SHAP model outputs into a professional, clear 2-sentence summary
for the fraud operations team.

Risk Level: {risk_level}
Recommended Action: {action}
Top Contributing SHAP Features: {features_text}

Rules:
- Base your response STRICTLY on the provided features. Do not invent external facts.
- If the risk level is Low, describe the transaction as consistent with normal behavior.
- Output ONLY the 2-sentence summary.
"""
        try:
            response = requests.post(
                self.llm_url,
                json={"model": self.llm_model, "prompt": prompt, "stream": False},
                timeout=4,
            )
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                if text:
                    return text
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"LLM generation failed or timed out: {e}. Using rule-based summary."
            )

        return self._fallback_summary(top_features, risk_level, action)