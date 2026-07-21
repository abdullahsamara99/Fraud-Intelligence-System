from features.preprocess import load_data
from features.feature_engineering import create_features

df = load_data(
    "data/raw/customers.csv",
    "data/raw/transactions.csv"
)

df = create_features(df)

print(df.shape)

print(df[
    [
        "transaction_amount",
        "customer_avg_amount",
        "amount_deviation",
        "amount_to_income_ratio",
        "hour",
        "country_changed"
    ]
].head())