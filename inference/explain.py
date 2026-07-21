import joblib
import requests
import shap
import pandas as pd
from utils.logger import logger


class FraudExplainer:
    """
    Explains Isolation Forest predictions using SHAP and formats
    the output into business reasoning using an LLM Layer.
    """

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
        Builds the SHAP explainer using K-Means background summary.
        Extracts raw NumPy data array to prevent 'DenseData object is not callable' error.
        """
        try:
            logger.info(f"Building SHAP explainer on {len(background_data)} samples...")
            X_background = self.preprocessor.transform(background_data)

            # K-Means clustering reduces background size for performance
            kmeans_res = shap.kmeans(X_background, 20)
            
            # Extract raw NumPy data array from DenseData wrapper
            background_summary = (
                kmeans_res.data if hasattr(kmeans_res, "data") else kmeans_res
            )

            # Explicitly use KernelExplainer for stable model decision_function evaluation
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
        Computes SHAP feature attributions for a single transaction safely.
        """
        try:
            if self.explainer is None:
                raise RuntimeError("Explainer is not built. Call build_explainer first.")

            X = self.preprocessor.transform(transaction)

            # Directly compute SHAP values via shap_values method
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
                # In Isolation Forest decision_function: Positive = Normal, Negative = Anomaly
                importance.append({
                    "feature": feature,
                    "impact": round(abs(val_float), 6),
                    "direction": "Decrease Risk" if val_float > 0 else "Increase Risk",
                })

            importance.sort(key=lambda x: x["impact"], reverse=True)
            return importance[:top_n]

        except Exception:
            logger.exception("SHAP attribution computation failed.")
            raise

    def generate_llm_summary(self, top_features: list, risk_level: str, action: str) -> str:
        """
        Translates SHAP outputs into executive business text via LLM.
        """
        features_text = ", ".join([
            f"{f['feature']} ({f['direction']}, impact={f['impact']})"
            for f in top_features
        ])

        prompt = f"""
You are an expert Anti-Money Laundering (AML) and Fraud Risk Analyst at a bank.
Translate these Machine Learning model SHAP outputs into a professional, clear 2-sentence summary for the fraud operations team.

Risk Level: {risk_level}
Recommended Action: {action}
Top Contributing SHAP Features: {features_text}

Rules:
- Base your response STRICTLY on the provided features. Do not hallucinate external facts.
- Output ONLY the 2-sentence summary.
"""
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=4,
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.warning(f"LLM generation failed or timed out: {e}. Falling back to rule-based summary.")

        # Fallback summary if LLM server is unreachable
        risk_features = [f['feature'] for f in top_features if f['direction'] == 'Increase Risk']
        if risk_features:
            return f"Transaction flagged as {risk_level} risk primarily due to abnormal patterns in: {', '.join(risk_features)}."
        return f"Transaction evaluated as {risk_level} risk with standard behavior across key parameters."