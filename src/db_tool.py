import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "business.db"


def run_query(sql: str) -> pd.DataFrame:
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


if __name__ == "__main__":
    print("可用产品：", get_all_products())

    print("\n=== AI营销助手销量趋势 ===")
    print(get_product_sales_trend("AI营销助手"))

    print("\n=== AI营销助手差评 ===")
    print(get_negative_reviews("AI营销助手"))

    print("\n=== 退款率最高产品 ===")
    print(get_top_refund_products())