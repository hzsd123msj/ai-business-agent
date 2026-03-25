from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    EMBEDDING_MODEL_NAME
)

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = BASE_DIR / "chroma_db"


def load_vector_store():
    print("1. 正在加载向量库...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    db = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings
    )

    print("2. 向量库加载完成")
    return db


def create_llm():
    print("3. 正在初始化 DeepSeek...")
    print("OPENAI_API_KEY exists in rag_engine =", bool(OPENAI_API_KEY))
    print("DEEPSEEK_API_KEY exists in rag_engine =", bool(DEEPSEEK_API_KEY))

    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0
    )
    print("4. DeepSeek 初始化完成")
    return llm


def rag_query(query):
    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 2})

    print("5. 开始检索相关文档...")
    docs = retriever.invoke(query)
    print(f"6. 检索完成，命中文档数: {len(docs)}")

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
你是一个企业知识助手，请严格根据参考内容回答问题。

【参考内容】
{context}

【问题】
{query}

要求：
1. 只根据参考内容回答
2. 如果参考内容中没有明确答案，请回答“未找到相关信息”
3. 回答尽量简洁准确
"""

    llm = create_llm()

    print("7. 正在调用 DeepSeek 生成回答...")
    response = llm.invoke(prompt)
    print("8. 回答生成完成")

    return response.content, docs


def main():
    print("RAG 问答系统启动（输入 quit 退出）")

    while True:
        query = input("\n请输入问题：").strip()

        if query.lower() in ["quit", "exit"]:
            print("系统退出")
            break

        if not query:
            print("问题不能为空，请重新输入。")
            continue

        try:
            answer, docs = rag_query(query)

            print("\n回答：")
            print(answer)

            print("\n参考来源：")
            for doc in docs:
                print("-", doc.metadata)

        except Exception as e:
            print("\n程序报错：")
            print(e)


if __name__ == "__main__":
    main()