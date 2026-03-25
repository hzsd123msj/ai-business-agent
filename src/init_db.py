import sqlite3
import pandas as pd

from src.config import DATA_DIR, DB_PATH


def init_database():
    sales_path = DATA_DIR / "sales_data.csv"
    reviews_path = DATA_DIR / "reviews.csv"

    if not sales_path.exists():
        raise FileNotFoundError(f"未找到文件: {sales_path}")
    if not reviews_path.exists():
        raise FileNotFoundError(f"未找到文件: {reviews_path}")

    conn = sqlite3.connect(DB_PATH)

    sales_df = pd.read_csv(sales_path)
    reviews_df = pd.read_csv(reviews_path)

    sales_df.to_sql("sales_data", conn, if_exists="replace", index=False)
    reviews_df.to_sql("reviews", conn, if_exists="replace", index=False)

    conn.close()
    print(f"数据库初始化完成: {DB_PATH}")


if __name__ == "__main__":
    init_database()
