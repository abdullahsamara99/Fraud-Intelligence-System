import pandas as pd

from utils.logger import logger


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fraud detection features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features.
    """

    try:
        logger.info("Starting feature engineering...")

        df = df.copy()

        # --------------------------------------------------
        # Convert timestamp
        # --------------------------------------------------

        logger.info("Converting timestamp column...")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort for rolling-window calculations
        df = df.sort_values(
            ["customer_id", "timestamp"]
        ).reset_index(drop=True)

        # --------------------------------------------------
        # Time Features
        # --------------------------------------------------

        logger.info("Creating time-based features...")

        df["hour"] = df["timestamp"].dt.hour

        df["day_of_week"] = df["timestamp"].dt.dayofweek

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        df["is_night"] = (
            (df["hour"] < 6) |
            (df["hour"] >= 22)
        ).astype(int)

        # --------------------------------------------------
        # Geographic Inconsistency Indicators
        # --------------------------------------------------

        logger.info("Creating geographic features...")

        df["is_home_country"] = (
            df["transaction_country"]
            == df["home_country"]
        ).astype(int)

        df["is_home_city"] = (
            df["transaction_city"]
            == df["home_city"]
        ).astype(int)

        # --------------------------------------------------
        # Customer Statistics
        # --------------------------------------------------

        logger.info("Creating customer statistics...")

        df["customer_avg_amount"] = (
            df.groupby("customer_id")["transaction_amount"]
              .expanding()
              .mean()
              .reset_index(level=0, drop=True)
        )

        df["customer_transaction_count"] = (
            df.groupby("customer_id")["transaction_amount"]
              .cumcount() + 1
        )

        # --------------------------------------------------
        # Customer Behavioral Aggregates (Baseline Features)
        # Group-by aggregates that capture each customer's typical
        # behavior across day, merchant category, country, and channel.
        # --------------------------------------------------

        logger.info("Creating customer behavioral aggregates...")

        # Daily key (derived from timestamp) used for velocity aggregates.
        df["transaction_day"] = df["timestamp"].dt.normalize()

        # 1. Customer + Day : daily spending velocity & activity frequency
        daily_group = df.groupby(
            ["customer_id", "transaction_day"]
        )["transaction_amount"]

        df["cust_daily_total_amount"] = daily_group.transform("sum")
        df["cust_daily_txn_count"] = daily_group.transform("count")

        # 2. Customer + Merchant Category : sector purchasing behavior
        cat_group = df.groupby(
            ["customer_id", "merchant_category"]
        )["transaction_amount"]

        df["cust_cat_total_amount"] = cat_group.transform("sum")
        df["cust_cat_txn_count"] = cat_group.transform("count")

        # 3. Customer + Country : geographic exposure
        df["cust_country_txn_count"] = (
            df.groupby(
                ["customer_id", "transaction_country"]
            )["transaction_amount"]
            .transform("count")
        )

        # 4. Customer + Channel : payment-method anomalies
        df["cust_channel_txn_count"] = (
            df.groupby(
                ["customer_id", "channel"]
            )["transaction_amount"]
            .transform("count")
        )

        # Helper key is not a model feature — drop it.
        df = df.drop(columns=["transaction_day"])

        # --------------------------------------------------
        # Transaction Frequency Over Time Windows
        # --------------------------------------------------

        logger.info(
            "Creating transaction frequency features..."
        )

        # Sort before applying rolling
        df = df.sort_values(
            ["customer_id", "timestamp"]
        ).reset_index(drop=True)

        transactions_last_24h = (
            df.groupby("customer_id")
            .rolling(
                "1D",
                on="timestamp"
            )["transaction_amount"]
            .count()
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

        df["transactions_last_24h"] = transactions_last_24h

        df["transactions_last_7d"] = (
            df.groupby("customer_id")
            .rolling(
                "7D",
                on="timestamp"
            )["transaction_amount"]
            .count()
            .reset_index(level=0, drop=True)
            .to_numpy()
        )

        # --------------------------------------------------
        # Merchant Statistics
        # --------------------------------------------------

        logger.info("Creating merchant statistics...")

        df["merchant_category_frequency"] = (
            df.groupby("merchant_category")["merchant_category"]
            .transform("count")
        )

        # --------------------------------------------------
        # Financial Features
        # --------------------------------------------------

        logger.info("Creating financial features...")

        df["amount_to_income_ratio"] = (
            df["transaction_amount"]
            / (df["avg_monthly_income"] + 1)
        )

        logger.info(
            f"Feature engineering completed successfully. Shape: {df.shape}"
        )

        return df

    except KeyError as e:
        logger.exception(
            f"Missing required column: {e}"
        )
        raise

    except Exception:
        logger.exception(
            "Feature engineering failed."
        )
        raise


if __name__ == "__main__":
    from features.preprocess import load_data

    logger.info("=" * 60)
    logger.info("Testing Feature Engineering")
    logger.info("=" * 60)

    df = load_data(
        "data/raw/customers.csv",
        "data/raw/transactions.csv",
    )

    df = create_features(df)

    print(df.head())

    print("\nNew Features Added:")

    print(
        [
            "is_home_country",
            "is_home_city",
            "transactions_last_24h",
            "transactions_last_7d",
            "cust_daily_total_amount",
            "cust_daily_txn_count",
            "cust_cat_total_amount",
            "cust_cat_txn_count",
            "cust_country_txn_count",
            "cust_channel_txn_count",
        ]
    )

    logger.info(
        "Feature engineering test completed successfully."
    )