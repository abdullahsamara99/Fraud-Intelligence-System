import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.preprocess import load_data
from inference.pipeline import FraudPipeline


df = load_data(
    "data/raw/customers.csv",
    "data/raw/transactions.csv"
)

sample = df.head(5)

pipeline = FraudPipeline()

results = pipeline.analyze(sample)

for result in results:
    print(result)