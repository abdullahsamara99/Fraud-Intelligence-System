import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.preprocess import load_data
from inference.predictor import FraudPredictor


print("Loading data...")

df = load_data(
    "data/raw/customers.csv",
    "data/raw/transactions.csv",
)

print("Loading model...")

predictor = FraudPredictor()

print("Predicting...")

sample = df.head(10)

results = predictor.predict(sample)

print(
    results[
        [
            "transaction_amount",
            "anomaly_score",
            "prediction",
        ]
    ]
)