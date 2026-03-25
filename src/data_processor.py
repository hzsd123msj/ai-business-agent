import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, EMBEDDING_MODEL_NAME

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "chroma_db"


def load_documents(data_dir=DATA_DIR):
    documents = []

    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return documents

    for file_path in data_dir.iterdir():
        filename = file_path.name

        if filename.endswith(".pdf"):
            try:
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
                print(f"已加载 PDF: {filename}")
            except Exception as e:
                print(f"加载 PDF 失败 {filename}: {e}")

        elif filename.endswith(".md") or filename.endswith(".txt"):
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
                print(f"已加载文本: {filename}")
            except Exception as e:
                print(f"加载文本失败 {filename}: {e}")

        else:
            print(f"跳过不支持的文件类型: {filename}")

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def build_vector_store(chunks, persist_directory=VECTOR_DIR):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    db.persist()
    return db


def main():
    print("开始加载文档...")
    documents = load_documents()
    print(f"加载文档数: {len(documents)}")

    if not documents:
        print("没有可处理的文档，请检查 data 文件夹。")
        return

    print("开始切分文档...")
    chunks = split_documents(documents)
    print(f"切分后 chunk 数: {len(chunks)}")

    print("开始构建向量库...")
    build_vector_store(chunks)

    print(f"向量库构建完成，已保存到: {VECTOR_DIR}")


if __name__ == "__main__":
    main()