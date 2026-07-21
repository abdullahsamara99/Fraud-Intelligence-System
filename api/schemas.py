from pydantic import BaseModel


class TransactionRequest(BaseModel):

    transaction_id: str
    customer_id: str
    account_number: str

    timestamp: str
    transaction_date: str
    transaction_time: str

    transaction_amount: float

    currency: str

    transaction_country: str
    transaction_country_iso: str
    transaction_city: str

    is_cross_border: int

    merchant_name: str
    merchant_category: str

    channel: str

    card_type: str

    transaction_type: str

    home_country: str
    home_city: str

    avg_monthly_income: float
    account_age_days: int