import os
import sys
import shutil
from datetime import datetime

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
    split_docs = split_documents_with_metadata(
        raw_docs,
        chunk_size=350,
        chunk_overlap=80,
    )
    print(f"切块完成，共 {len(split_docs)} 个 chunk。")

    print("开始构建 FAISS 索引...")
    vectorstore = build_faiss_from_documents(split_docs)

    index_parent = os.path.dirname(INDEX_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_index_dir = f"{INDEX_DIR}.tmp_{timestamp}"
    backup_index_dir = f"{INDEX_DIR}.bak_{timestamp}"

    os.makedirs(index_parent, exist_ok=True)
    print(f"先保存新索引到临时目录：{temp_index_dir}")
    save_faiss(vectorstore, temp_index_dir)

    if os.path.exists(INDEX_DIR):
        print(f"发现旧索引，先备份到：{backup_index_dir}")
        if os.path.exists(backup_index_dir):
            shutil.rmtree(backup_index_dir)
        os.replace(INDEX_DIR, backup_index_dir)

    print(f"切换新索引到：{INDEX_DIR}")
    os.replace(temp_index_dir, INDEX_DIR)

    if os.path.exists(backup_index_dir):
        print(f"删除旧索引备份：{backup_index_dir}")
        shutil.rmtree(backup_index_dir)

    print(f"索引构建完成，已保存到：{INDEX_DIR}")


if __name__ == "__main__":
    main()
