from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DIR, EMBEDDING_MODEL_NAME, DATA_DIR


def load_documents(data_dir=DATA_DIR):
    documents = []

    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    for file_path in data_dir.iterdir():
        filename = file_path.name.lower()

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

        elif filename.endswith(".md") or filename.endswith(".txt"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def build_vector_store(chunks, persist_directory=CHROMA_DIR):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    return db


def init_vector_store():
    documents = load_documents()
    chunks = split_documents(documents)
    build_vector_store(chunks)
    print(f"向量库构建完成，已保存到: {CHROMA_DIR}")


def ensure_vector_store():
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        init_vector_store()
        return

    # Chroma 目录存在但为空时也重建
    if not any(chroma_path.iterdir()):
        init_vector_store()


if __name__ == "__main__":
    init_vector_store()
