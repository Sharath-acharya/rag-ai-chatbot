import json
import os
from pathlib import Path
from typing import List

from pypdf import PdfReader


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def load_documents(docs_dir: str) -> List[str]:
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        return []

    all_texts: List[str] = []
    for file_path in sorted(docs_path.iterdir()):
        if file_path.is_dir():
            continue
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            try:
                text = read_text_file(file_path)
                if text.strip():
                    all_texts.append(text)
            except Exception:
                continue
        elif suffix == ".pdf":
            try:
                text = read_pdf_file(file_path)
                if text.strip():
                    all_texts.append(text)
            except Exception:
                continue
    return all_texts


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    if not paragraphs:
        return [text]

    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 1 <= chunk_size:
            current = (current + "\n" + paragraph).strip()
        else:
            if current:
                chunks.append(current)
            if len(paragraph) > chunk_size:
                words = paragraph.split()
                segment = ""
                for word in words:
                    if len(segment) + len(word) + 1 <= chunk_size:
                        segment = (segment + " " + word).strip()
                    else:
                        if segment:
                            chunks.append(segment)
                        segment = word
                if segment:
                    current = segment
            else:
                current = paragraph

    if current:
        chunks.append(current)

    merged: List[str] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            merged.append(chunk)
            continue

        prev = merged[-1]
        if len(prev) + len(chunk) <= chunk_size:
            merged[-1] = prev + "\n" + chunk
        else:
            merged.append(chunk)

    final_chunks: List[str] = []
    for idx, chunk in enumerate(merged):
        if idx == 0:
            final_chunks.append(chunk)
            continue
        prev = final_chunks[-1]
        if len(prev) + len(chunk) <= chunk_size + chunk_overlap:
            final_chunks[-1] = prev + "\n" + chunk
        else:
            final_chunks.append(chunk)

    return final_chunks


def save_chunks(chunks: List[str], output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
