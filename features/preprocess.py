import pandas as pd

from utils.logger import logger


def load_data(customers_path: str, transactions_path: str) -> pd.DataFrame:
    """
    Load customer and transaction datasets, merge them,
    and return a clean dataframe.

    Parameters
    ----------
    customers_path : str
        Path to customers.csv

    transactions_path : str
        Path to transactions.csv

    Returns
    -------
    pd.DataFrame
        Merged dataframe
    """

    try:

        # ----------------------------------------
        # Load Customers
        # ----------------------------------------

        logger.info("Loading customers dataset...")

        customers = pd.read_csv(customers_path)

        logger.info(
            f"Customers loaded successfully. Shape: {customers.shape}"
        )

        # ----------------------------------------
        # Load Transactions
        # ----------------------------------------

        logger.info("Loading transactions dataset...")

        transactions = pd.read_csv(transactions_path)

        logger.info(
            f"Transactions loaded successfully. Shape: {transactions.shape}"
        )

        # ----------------------------------------
        # Merge
        # ----------------------------------------

        logger.info("Merging datasets...")

        df = transactions.merge(
            customers,
            on="customer_id",
            how="left",
            suffixes=("", "_customer"),
        )

        logger.info(
            f"Datasets merged successfully. Shape: {df.shape}"
        )

        # ----------------------------------------
        # Remove duplicate columns
        # ----------------------------------------

        duplicate_columns = [
            "account_number_customer",
            "card_type_customer",
        ]

        existing_duplicates = [
            col
            for col in duplicate_columns
            if col in df.columns
        ]

        if existing_duplicates:

            logger.info(
                f"Removing duplicate columns: {existing_duplicates}"
            )

            df.drop(
                columns=existing_duplicates,
                inplace=True,
            )

        logger.info(
            f"Final dataframe shape: {df.shape}"
        )

        logger.info(
            f"Final columns: {list(df.columns)}"
        )

        return df

    except FileNotFoundError:

        logger.exception("Dataset file not found.")

        raise

    except pd.errors.EmptyDataError:

        logger.exception("Dataset is empty.")

        raise

    except Exception:

        logger.exception("Failed to load datasets.")

        raise


if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("Testing Data Loader")
    logger.info("=" * 60)

    df = load_data(
        "data/raw/customers.csv",
        "data/raw/transactions.csv",
    )

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    logger.info("Data loader test completed successfully.")