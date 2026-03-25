import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "business.db"

def main():
    sales_path = DATA_DIR / "sales_data.csv"
    reviews_path = DATA_DIR / "reviews.csv"

    if not sales_path.exists():
        print(f"未找到文件: {sales_path}")
        return
    if not reviews_path.exists():
        print(f"未找到文件: {reviews_path}")
        return

    conn = sqlite3.connect(DB_PATH)

    sales_df = pd.read_csv(sales_path)
    reviews_df = pd.read_csv(reviews_path)

    sales_df.to_sql("sales_data", conn, if_exists="replace", index=False)
    reviews_df.to_sql("reviews", conn, if_exists="replace", index=False)

    conn.close()
    print(f"数据库初始化完成: {DB_PATH}")

if __name__ == "__main__":
    main()