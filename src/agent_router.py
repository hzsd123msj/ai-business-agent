from langchain_openai import ChatOpenAI

from src.config import OPENAI_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from src.db_tool import (
    get_product_sales_trend,
    get_negative_reviews,
    get_all_reviews,
    get_top_refund_products,
    get_all_products
)


def create_llm():
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0
    )


def extract_product_name(user_query: str, products: list) -> str | None:
    for product in products:
        if product in user_query:
            return product
    return None


def detect_intent(user_query: str) -> str:
    query = user_query.lower()

    if "退款率" in user_query or "退款" in user_query:
        return "refund_ranking"

    if "差评" in user_query or "评论" in user_query or "反馈" in user_query:
        return "review_analysis"

    if "销量" in user_query or "销售" in user_query or "表现" in user_query:
        return "sales_analysis"

    return "general_analysis"


def build_prompt(user_query: str, product_name: str | None, intent: str, data_text: str) -> str:
    return f"""
你是企业业务分析助手，请根据用户问题和提供的数据进行分析。

【用户问题】
{user_query}

【识别出的产品】
{product_name}

【识别出的意图】
{intent}

【参考数据】
{data_text}

请根据问题输出：
1. 直接回答用户问题
2. 给出关键依据
3. 如有必要，给出改进建议

要求：
- 回答要简洁、清晰、有逻辑
- 只能根据提供的数据分析
- 如果数据不足，请明确说明
"""


def handle_sales_analysis(user_query: str, product_name: str) -> str:
    sales_df = get_product_sales_trend(product_name)
    reviews_df = get_negative_reviews(product_name)

    data_text = f"""
【销售趋势数据】
{sales_df.to_string(index=False)}

【差评数据】
{reviews_df.to_string(index=False)}
"""

    llm = create_llm()
    prompt = build_prompt(user_query, product_name, "sales_analysis", data_text)
    response = llm.invoke(prompt)
    return response.content


def handle_review_analysis(user_query: str, product_name: str) -> str:
    reviews_df = get_all_reviews(product_name)

    data_text = f"""
【评论数据】
{reviews_df.to_string(index=False)}
"""

    llm = create_llm()
    prompt = build_prompt(user_query, product_name, "review_analysis", data_text)
    response = llm.invoke(prompt)
    return response.content


def handle_refund_ranking(user_query: str) -> str:
    refund_df = get_top_refund_products()

    data_text = f"""
【退款率排名数据】
{refund_df.to_string(index=False)}
"""

    llm = create_llm()
    prompt = build_prompt(user_query, None, "refund_ranking", data_text)
    response = llm.invoke(prompt)
    return response.content


def answer_question(user_query: str) -> str:
    products = get_all_products()
    product_name = extract_product_name(user_query, products)
    intent = detect_intent(user_query)

    print(f"识别到的产品: {product_name}")
    print(f"识别到的意图: {intent}")

    if intent == "refund_ranking":
        return handle_refund_ranking(user_query)

    if product_name is None:
        return "未识别到具体产品名称，请在问题中明确提到产品名称，例如：AI营销助手、智能客服系统、数据分析平台。"

    if intent == "review_analysis":
        return handle_review_analysis(user_query, product_name)

    if intent in ["sales_analysis", "general_analysis"]:
        return handle_sales_analysis(user_query, product_name)

    return "暂时无法处理该问题。"


def main():
    print("业务分析 Agent 启动（输入 quit 退出）")

    while True:
        user_query = input("\n请输入问题：").strip()

        if user_query.lower() in ["quit", "exit"]:
            print("系统退出")
            break

        if not user_query:
            print("问题不能为空，请重新输入。")
            continue

        try:
            result = answer_question(user_query)
            print("\n分析结果：")
            print(result)
        except Exception as e:
            print("\n程序报错：")
            print(e)


if __name__ == "__main__":
    main()
