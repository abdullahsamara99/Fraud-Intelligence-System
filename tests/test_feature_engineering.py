from features.preprocess import load_data
from features.feature_engineering import create_features


def test_feature_engineering_creates_columns():

    df = load_data(
        "data/raw/customers.csv",
        "data/raw/transactions.csv",
    )

    df = create_features(df.head(20))

    expected_columns = [
          "is_weekend",
        "is_night",
        "hour",
        "day_of_week",
        "transactions_last_24h",
        "transactions_last_7d",
        "merchant_category_frequency",
        "amount_to_income_ratio",
    ]

    for column in expected_columns:
        assert column in df.columns