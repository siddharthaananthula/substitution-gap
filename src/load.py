import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_orders():
    return pd.read_csv(
        DATA / "orders.csv",
        dtype={
            "order_id": "int32",
            "user_id": "int32",
            "eval_set": "category",
            "order_number": "int16",
            "order_dow": "int8",
            "order_hour_of_day": "int8",
            "days_since_prior_order": "float32",
        },
    )


def load_order_products():
    return pd.read_csv(
        DATA / "order_products__prior.csv",
        dtype={
            "order_id": "int32",
            "product_id": "int32",
            "add_to_cart_order": "int16",
            "reordered": "int8",
        },
    )


def load_products():
    return pd.read_csv(
        DATA / "products.csv",
        dtype={
            "product_id": "int32",
            "product_name": "string",
            "aisle_id": "int16",
            "department_id": "int8",
        },
    )


def load_prior_lines():
    """Every prior order line, with the customer and order number attached."""
    orders = load_orders()
    prior = orders.loc[
        orders["eval_set"] == "prior", ["order_id", "user_id", "order_number"]
    ]
    return load_order_products().merge(prior, on="order_id", how="inner")