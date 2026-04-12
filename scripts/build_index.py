import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from app.config import DATA_DIR, INDEX_DIR, validate_config
from app.loaders import load_docs_from_dir
from app.splitter import split_documents_with_metadata
from app.vectorstore import build_faiss_from_documents, save_faiss


def main():
    validate_config()

    print("开始读取文档...")
    raw_docs = load_docs_from_dir(DATA_DIR)
    if not raw_docs:
        raise ValueError(f"未在 {DATA_DIR} 中发现可读取的 txt/md/pdf 文件。")

    print(f"读取完成，共 {len(raw_docs)} 个原始文档页/段。")

    print("开始切块...")
    split_docs = split_documents_with_metadata(raw_docs, chunk_size=500, chunk_overlap=100)
    print(f"切块完成，共 {len(split_docs)} 个 chunk。")

    print("开始构建 FAISS 索引...")
    vectorstore = build_faiss_from_documents(split_docs)

    os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)
    save_faiss(vectorstore, INDEX_DIR)
    print(f"索引构建完成，已保存到：{INDEX_DIR}")


if __name__ == "__main__":
    main()