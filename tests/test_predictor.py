from features.preprocess import load_data
from features.feature_engineering import create_features
from inference.predictor import FraudPredictor


def test_predictor_returns_expected_columns():

    df = load_data(
        "data/raw/customers.csv",
        "data/raw/transactions.csv",
    )

    feature_df = create_features(df.head(5))

    predictor = FraudPredictor()

    result = predictor.predict(feature_df)

    expected_columns = [
        "prediction",
        "anomaly_score",
        "risk_score",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_risk_score_between_zero_and_one():

    df = load_data(
        "data/raw/customers.csv",
        "data/raw/transactions.csv",
    )

    feature_df = create_features(df.head(5))

    predictor = FraudPredictor()

    result = predictor.predict(feature_df)

    assert result["risk_score"].between(0, 1).all()