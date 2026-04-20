from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader


def _add_common_metadata(docs: List[Document], path: Path) -> List[Document]:
    suffix = path.suffix.lower().lstrip(".")
    for doc in docs:
        doc.metadata["source"] = path.name
        doc.metadata["file_path"] = str(path)
        doc.metadata["file_type"] = suffix
    return docs


def load_txt_md(path: Path) -> List[Document]:
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    return _add_common_metadata(docs, path)


def is_noise_page(text: str) -> bool:
    t = text.strip()

    if len(t) < 40:
        return True

    chinese_count = sum('\u4e00' <= ch <= '\u9fff' for ch in t)
    alpha_count = sum(ch.isalpha() for ch in t)
    if chinese_count + alpha_count < 20:
        return True

    if "目录" in t and t.count("...") >= 2:
        return True

    lines = [line.strip() for line in t.splitlines() if line.strip()]
    if len(lines) <= 2:
        joined = " ".join(lines)
        if joined.isdigit():
            return True

    return False


def load_pdf(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()

    filtered_docs = []
    for doc in docs:
        text = doc.page_content.strip()
        if is_noise_page(text):
            continue

        doc.metadata["source"] = path.name
        doc.metadata["file_path"] = str(path)
        doc.metadata["file_type"] = "pdf"
        filtered_docs.append(doc)

    return filtered_docs


def infer_subject_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()

    if "psychology" in parts or "心理" in name:
        return "psychology"
    if "physics" in parts or "物理" in name:
        return "physics"
    if "math" in parts or "数学" in name:
        return "math"
    if "finance" in parts or "金融" in name or "货币" in name:
        return "finance"
    return "mixed"


def load_docs_from_dir(data_dir: str) -> List[Document]:
    all_docs: List[Document] = []
    skipped_files: List[str] = []

    for path in Path(data_dir).rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        try:
            if suffix in [".txt", ".md"]:
                docs = load_txt_md(path)
            elif suffix == ".pdf":
                docs = load_pdf(path)
            else:
                continue
        except Exception as exc:
            skipped_files.append(f"{path} ({type(exc).__name__}: {exc})")
            continue

        subject = infer_subject_from_path(path)
        for doc in docs:
            doc.metadata["subject"] = subject

        all_docs.extend(docs)

    if skipped_files:
        print("以下文件读取失败，已跳过：")
        for item in skipped_files:
            print(f"- {item}")

    return all_docs
