"""Evaluation events from each customer's final (train) order. See DECISIONS.md D011.

An event is a regular product A (found from prior orders only) that is missing
from the customer's final order. The truth for that event is the set of
first-time buys (reordered == 0) in that final order.

Run with: python -m src.evaluate
"""
import gc

from src.load import load_prior_lines, load_products, load_train_lines
from src.pairs import find_regulars, orders_per_user


def prior_regulars():
    """Regulars from prior lines only, exactly as in build_substitutes."""
    lines = load_prior_lines()
    reg = find_regulars(lines, orders_per_user(lines))
    del lines
    gc.collect()
    return reg


def build_events(reg, train_lines):
    """One row per (customer, final order, regular A) where A is missing from that order."""
    final = train_lines[["user_id", "order_id"]].drop_duplicates()
    assert final["user_id"].is_unique, "a customer has more than one train order"

    expected = reg.merge(final, on="user_id", how="inner")  # drops test customers
    assert len(expected) <= len(reg)

    bought = train_lines[["order_id", "product_id"]].assign(in_order=True)
    expected = expected.merge(bought, on=["order_id", "product_id"], how="left")
    assert len(expected) == len(reg.merge(final, on="user_id")), "join added rows"

    events = expected.loc[expected["in_order"].isna(), ["user_id", "order_id", "product_id"]]
    return events.rename(columns={"product_id": "a"}).reset_index(drop=True)


def first_time_buys(train_lines):
    """The truth set: (order_id, b) for every first-time buy in a final order."""
    new = train_lines.loc[train_lines["reordered"] == 0, ["order_id", "product_id"]]
    return new.rename(columns={"product_id": "b"}).reset_index(drop=True)


def main():
    reg = prior_regulars()
    train_lines = load_train_lines()[["order_id", "user_id", "product_id", "reordered"]]
    print(f"regular pairs (prior): {len(reg):,}, customers: {reg['user_id'].nunique():,}")
    print(f"train lines: {len(train_lines):,}, train customers: "
          f"{train_lines['user_id'].nunique():,}")

    events = build_events(reg, train_lines)
    truth = first_time_buys(train_lines)

    eval_users = reg.loc[reg["user_id"].isin(train_lines["user_id"]), "user_id"].unique()
    per_customer = events.groupby("user_id").size()
    has_truth = events["order_id"].isin(truth["order_id"])

    print(f"\ncustomers with regulars and a train order: {len(eval_users):,}")
    print(f"events (regular missing from final order): {len(events):,}")
    print(f"customers with at least one event: {events['user_id'].nunique():,}")
    print(f"median events per customer (with >= 1 event): {per_customer.median():.0f}")
    print(f"median events per customer (incl. 0, of {len(eval_users):,}): "
          f"{per_customer.reindex(eval_users, fill_value=0).median():.0f}")
    print(f"first-time buys in final orders: {len(truth):,}")
    print(f"share of events with >= 1 first-time buy: {has_truth.mean():.1%}")

    names = load_products().set_index("product_id")["product_name"]
    top = events["a"].value_counts().head(10).rename("events").to_frame()
    top.insert(0, "product_name", top.index.map(names))
    print("\nTop 10 regular products by events:")
    print(top.to_string())


if __name__ == "__main__":
    main()
