import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.preprocess import load_data
from features.feature_engineering import create_features
from inference.explain import FraudExplainer

print("Loading data...")

df = load_data(
    "data/raw/customers.csv",
    "data/raw/transactions.csv",
)

print("Creating features...")

df = create_features(df)

drop_columns = [
    "transaction_id",
    "customer_id",
    "account_number",
    "timestamp",
    "transaction_date",
    "transaction_time",
    "merchant_name",
    "transaction_country_iso",
]

df = df.drop(columns=drop_columns)

print("Building SHAP explainer...")

explainer = FraudExplainer()

explainer.build_explainer(
    df.sample(1000, random_state=42)
)

print("Explaining transaction...")

result = explainer.explain_transaction(
    transaction=df.head(1),
    feature_names=df.columns.tolist(),
)

print("\nTop Features:\n")

for feature in result["top_features"]:
    print(feature)

print("\nSummary:\n")
print(result["summary"])