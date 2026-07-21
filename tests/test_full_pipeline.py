import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.preprocess import load_data
from inference.pipeline import FraudPipeline

print("Loading Data...")

df = load_data(
    "data/raw/customers.csv",
    "data/raw/transactions.csv",
)

pipeline = FraudPipeline()

print("Running Pipeline...")

results = pipeline.analyze(
    df.head(3)
)

for i, result in enumerate(results):

    print("\n========================")
    print(f"Transaction {i+1}")
    print("========================")

    for key, value in result.items():
        print(f"{key}:")
        print(value)