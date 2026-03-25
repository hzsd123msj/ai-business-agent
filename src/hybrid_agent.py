from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    OPENAI_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    EMBEDDING_MODEL_NAME
)

from src.db_tool import (
    get_product_sales_trend,
    get_negative_reviews,
    get_all_reviews,
    get_top_refund_products,
    get_all_products
)

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "chroma_db"


def create_llm():
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0
    )


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    db = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings
    )
    return db


def extract_product_name(user_query: str, products: list) -> str | None:
    for product in products:
        if product in user_query:
            return product
    return None


def detect_intent(user_query: str) -> str:
    if "退款率" in user_query or "退款" in user_query:
        return "refund_ranking"

    if "差评" in user_query or "评论" in user_query or "反馈" in user_query:
        return "review_analysis"

    if "销量" in user_query or "销售" in user_query or "表现" in user_query:
        return "sales_analysis"

    return "general_analysis"


def retrieve_knowledge(user_query: str, product_name: str | None, top_k: int = 3):
    db = load_vector_store()

    if product_name:
        search_query = f"{product_name} {user_query}"
    else:
        search_query = user_query

    retriever = db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(search_query)
    return docs


def format_docs(docs) -> str:
    if not docs:
        return "未检索到相关文档。"

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        parts.append(f"【文档{i}】\n来源: {source}\n内容:\n{content}")

    return "\n\n".join(parts)


def build_prompt(user_query: str, product_name: str | None, intent: str, structured_data: str,
                 knowledge_data: str) -> str:
    return f"""
你是企业级业务分析助手，请结合结构化数据和企业知识文档，回答用户问题。

【用户问题】
{user_query}

【识别出的产品】
{product_name}

【识别出的意图】
{intent}

【结构化数据】
{structured_data}

【知识库检索结果】
{knowledge_data}

请按以下要求输出：
1. 直接回答用户问题
2. 说明关键依据（区分数据依据和文档依据）
3. 给出清晰的业务建议
4. 如果信息不足，请明确指出不足之处

要求：
- 回答要有逻辑、有层次
- 只能基于已提供的数据和文档内容
- 不要编造不存在的信息
"""


def handle_sales_analysis(user_query: str, product_name: str) -> str:
    sales_df = get_product_sales_trend(product_name)
    reviews_df = get_negative_reviews(product_name)
    docs = retrieve_knowledge(user_query, product_name, top_k=3)

    structured_data = f"""
【销售趋势数据】
{sales_df.to_string(index=False)}

【差评数据】
{reviews_df.to_string(index=False)}
"""

    knowledge_data = format_docs(docs)

    llm = create_llm()
    prompt = build_prompt(
        user_query=user_query,
        product_name=product_name,
        intent="sales_analysis",
        structured_data=structured_data,
        knowledge_data=knowledge_data
    )
    response = llm.invoke(prompt)
    return response.content


def handle_review_analysis(user_query: str, product_name: str) -> str:
    reviews_df = get_all_reviews(product_name)
    docs = retrieve_knowledge(user_query, product_name, top_k=3)

    structured_data = f"""
【评论数据】
{reviews_df.to_string(index=False)}
"""

    knowledge_data = format_docs(docs)

    llm = create_llm()
    prompt = build_prompt(
        user_query=user_query,
        product_name=product_name,
        intent="review_analysis",
        structured_data=structured_data,
        knowledge_data=knowledge_data
    )
    response = llm.invoke(prompt)
    return response.content


def handle_refund_ranking(user_query: str) -> str:
    refund_df = get_top_refund_products()
    docs = retrieve_knowledge(user_query, None, top_k=3)

    structured_data = f"""
【退款率排名数据】
{refund_df.to_string(index=False)}
"""

    knowledge_data = format_docs(docs)

    llm = create_llm()
    prompt = build_prompt(
        user_query=user_query,
        product_name=None,
        intent="refund_ranking",
        structured_data=structured_data,
        knowledge_data=knowledge_data
    )
    response = llm.invoke(prompt)
    return response.content


def answer_question(user_query: str):
    products = get_all_products()
    product_name = extract_product_name(user_query, products)
    intent = detect_intent(user_query)

    if intent == "refund_ranking":
        refund_df = get_top_refund_products()
        docs = retrieve_knowledge(user_query, None, top_k=3)

        result = handle_refund_ranking(user_query)

        return {
            "answer": result,
            "data": refund_df,
            "docs": docs
        }

    if product_name is None:
        docs = retrieve_knowledge(user_query, None, top_k=3)

        result = "未识别到产品，以下为相关知识："

        return {
            "answer": result,
            "data": None,
            "docs": docs
        }

    if intent == "review_analysis":
        reviews_df = get_all_reviews(product_name)
        docs = retrieve_knowledge(user_query, product_name, top_k=3)

        result = handle_review_analysis(user_query, product_name)

        return {
            "answer": result,
            "data": reviews_df,
            "docs": docs
        }

    if intent in ["sales_analysis", "general_analysis"]:
        sales_df = get_product_sales_trend(product_name)
        docs = retrieve_knowledge(user_query, product_name, top_k=3)

        result = handle_sales_analysis(user_query, product_name)

        return {
            "answer": result,
            "data": sales_df,
            "docs": docs
        }

    return {
        "answer": "暂时无法处理该问题。",
        "data": None,
        "docs": []
    }
def main():
    print("Hybrid Agent 启动（输入 quit 退出）")

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
