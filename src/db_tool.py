import sqlite3
import pandas as pd

from src.config import DB_PATH
from src.init_db import init_database


def ensure_database():
    if not DB_PATH.exists():
        init_database()


def run_query(sql: str) -> pd.DataFrame:
    ensure_database()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    finally:
        conn.close()


def get_product_sales_trend(product_name: str) -> pd.DataFrame:
    sql = f"""
    SELECT date, product_name, region, sales_amount, order_count, refund_rate
    FROM sales_data
    WHERE product_name = '{product_name}'
    ORDER BY date ASC
    """
    return run_query(sql)


def get_negative_reviews(product_name: str) -> pd.DataFrame:
    sql = f"""
    SELECT review_id, product_name, review_text, rating, review_date
    FROM reviews
    WHERE product_name = '{product_name}' AND rating <= 3
    ORDER BY review_date ASC
    """
    return run_query(sql)


def get_all_reviews(product_name: str) -> pd.DataFrame:
    sql = f"""
    SELECT review_id, product_name, review_text, rating, review_date
    FROM reviews
    WHERE product_name = '{product_name}'
    ORDER BY review_date ASC
    """
    return run_query(sql)


def get_top_refund_products() -> pd.DataFrame:
    sql = """
    SELECT product_name,
           AVG(refund_rate) AS avg_refund_rate,
           SUM(sales_amount) AS total_sales,
           SUM(order_count) AS total_orders
    FROM sales_data
    GROUP BY product_name
    ORDER BY avg_refund_rate DESC
    """
    return run_query(sql)


def get_all_products() -> list:
    sql = """
    SELECT DISTINCT product_name
    FROM sales_data
    """
    df = run_query(sql)
    return df["product_name"].tolist()
