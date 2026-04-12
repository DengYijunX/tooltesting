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


def load_pdf(path: Path) -> List[Document]:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    return _add_common_metadata(docs, path)


def infer_subject_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "psychology" in parts or "心理" in path.name.lower():
        return "psychology"
    if "physics" in parts or "物理" in path.name.lower():
        return "physics"
    return "mixed"


def load_docs_from_dir(data_dir: str) -> List[Document]:
    all_docs: List[Document] = []

    for path in Path(data_dir).rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in [".txt", ".md"]:
            docs = load_txt_md(path)
        elif suffix == ".pdf":
            docs = load_pdf(path)
        else:
            continue

        subject = infer_subject_from_path(path)
        for doc in docs:
            doc.metadata["subject"] = subject

        all_docs.extend(docs)

    return all_docs