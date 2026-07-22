import joblib
import requests
import shap
import pandas as pd
from utils.logger import logger


# ==========================================================
# Feature -> business language translation
# Keeps explanations grounded: every phrase maps to a real
# SHAP-attributed feature, never an invented fact.
# ==========================================================
def _fmt(value):
    """Human-friendly value formatting."""
    try:
        f = float(value)
        if f.is_integer():
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def humanize(feature: str, value) -> str:
    """Translate a feature (and its value) into an analyst-style phrase."""
    v = _fmt(value)
    mapping = {
        "transaction_amount": f"an unusually high transaction amount ({v})",
        "amount_to_income_ratio": "a transaction value that is large relative to the customer's income",
        "customer_avg_amount": "a clear deviation from the customer's typical spending amount",
        "is_night": "activity during unusual night-time hours",
        "is_weekend": "activity occurring over the weekend",
        "hour": "an atypical transaction hour",
        "is_home_country": "a location outside the customer's home country",
        "is_cross_border": "a cross-border transaction",
        "is_home_city": "a location outside the customer's usual city",
        "transaction_country": f"an uncommon transaction country ({v})",
        "transaction_city": f"an uncommon transaction city ({v})",
        "merchant_category": f"a rare or atypical merchant category ({v})",
        "merchant_category_frequency": "a rarely used merchant category for this customer",
        "transaction_type": f"an unusual transaction type ({v})",
        "channel": f"an unusual payment channel ({v})",
        "card_type": f"an unusual card type ({v})",
        "currency": f"an uncommon currency ({v})",
        "transactions_last_24h": "a spike in transaction frequency over the last 24 hours",
        "transactions_last_7d": "elevated transaction frequency over the past week",
        "transaction_count": "an abnormal number of recent transactions",
    }
    return mapping.get(feature, f"an abnormal value for {feature} ({v})")


class FraudExplainer:
    """
    Explains Isolation Forest predictions with SHAP, then formats the
    grounded attributions into business reasoning via a local LLM,
    with a narrative rule-based fallback.

    Sign convention (IsolationForest.decision_function):
        positive SHAP -> pushes toward NORMAL (lowers risk)
        negative SHAP -> pushes toward ANOMALOUS (raises risk)
    """

    SIGNAL_FLOOR = 0.03  # contributions below this magnitude are noise

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
        Compute SHAP attributions for a single transaction and attach the
        raw feature value to each entry so downstream summaries can cite it.
        """
        try:
            if self.explainer is None:
                raise RuntimeError("Explainer is not built. Call build_explainer first.")

            raw_values = transaction.iloc[0].to_dict()
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
                    "value": raw_values.get(feature),
                })

            importance.sort(key=lambda x: x["impact"], reverse=True)
            return importance[:top_n]

        except Exception:
            logger.exception("SHAP attribution computation failed.")
            raise

    # ------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------
    def _risk_evidence(self, top_features: list) -> list:
        """Grounded business phrases for the features that raise risk."""
        return [
            humanize(f["feature"], f.get("value"))
            for f in top_features
            if f["direction"] == "Increase Risk" and f["impact"] >= self.SIGNAL_FLOOR
        ]

    @staticmethod
    def _join(phrases: list) -> str:
        if not phrases:
            return ""
        if len(phrases) == 1:
            return phrases[0]
        return ", ".join(phrases[:-1]) + ", and " + phrases[-1]

    def _fallback_summary(self, top_features: list, risk_level: str, action: str) -> str:
        """Deterministic, narrative, risk-aware explanation grounded in SHAP."""
        evidence = self._risk_evidence(top_features)

        if risk_level in ("Critical", "High") and evidence:
            return (
                f"This transaction significantly deviates from the customer's "
                f"historical behavior, showing {self._join(evidence)}. Based on "
                f"these combined signals it is assessed as {risk_level} risk, and "
                f"the recommended action is to {action.lower()}."
            )
        if risk_level == "Medium" and evidence:
            return (
                f"This transaction shows moderate deviation from normal behavior, "
                f"including {self._join(evidence)}. It is assessed as Medium risk "
                f"and is recommended for manual review before approval."
            )
        return (
            f"This transaction is consistent with the customer's established behavior "
            f"profile, with no individual feature deviating significantly from normal "
            f"patterns. It is assessed as {risk_level} risk and can be approved."
        )

    def generate_llm_summary(self, top_features: list, risk_level: str, action: str) -> str:
        """
        Reformat grounded SHAP evidence into a 2-3 sentence analyst summary
        via the LLM, falling back to the deterministic narrative.
        """
        evidence = self._risk_evidence(top_features)

        # Nothing anomalous -> don't ask the LLM to invent reasons.
        if not evidence:
            return self._fallback_summary(top_features, risk_level, action)

        evidence_block = "\n".join(f"- {p}" for p in evidence)

        prompt = f"""You are a senior fraud risk analyst at a bank writing a short case note for the fraud operations team.

The anomaly-detection model flagged a transaction. These are the ONLY grounded findings from the model's SHAP attribution:
{evidence_block}

Model assessment:
- Risk level: {risk_level}
- Recommended action: {action}

Write a clear, professional 2-3 sentence summary explaining why this transaction is suspicious.

Strict rules:
- Use ONLY the findings listed above. Do NOT invent any facts, numbers, names, or details not present in the findings.
- Weave the findings into fluent business reasoning; do not just list them.
- End by stating the risk level and recommended action.
- Output ONLY the summary text, no preamble."""

        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 220},
                },
                timeout=15,
            )
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                if text:
                    return text
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM generation failed or timed out: {e}. Using fallback.")

        return self._fallback_summary(top_features, risk_level, action)