from langchain_openai import ChatOpenAI
from src.config import OPENAI_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.db_tool import get_product_sales_trend, get_negative_reviews


def create_llm():
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0
    )


def analyze_product(product_name: str):
    sales_df = get_product_sales_trend(product_name)
    reviews_df = get_negative_reviews(product_name)

    sales_text = sales_df.to_string(index=False)
    reviews_text = reviews_df.to_string(index=False)

    prompt = f"""
你是企业业务分析助手。请根据下面的销售数据和用户差评，分析产品表现并给出建议。

【产品名称】
{product_name}

【销售数据】
{sales_text}

【差评数据】
{reviews_text}

请输出：
1. 销量趋势判断
2. 可能原因分析
3. 改进建议
"""

    llm = create_llm()
    response = llm.invoke(prompt)
    return response.content


def main():
    product_name = input("请输入产品名称：").strip()
    result = analyze_product(product_name)
    print("\n分析结果：")
    print(result)


if __name__ == "__main__":
    main()
