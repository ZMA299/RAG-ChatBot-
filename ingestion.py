from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from database import SupabaseGateway


class KnowledgeIngestionService:
    def __init__(self, settings: Settings, db: SupabaseGateway, embeddings: Any):
        self.settings = settings
        self.db = db
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "؟", "!", "؛", "،", " ", ""],
        )

    def ingest_faq_csv(self, file, batch_size: int = 64) -> int:
        if not file:
            return 0

        faq_rows: list[dict[str, str]] = []
        with open(file.name, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                category = (row.get("category") or "").strip()
                if question and answer:
                    faq_rows.append({"question": question, "answer": answer, "category": category})

        total_inserted = 0
        for start_i in range(0, len(faq_rows), batch_size):
            batch_rows = faq_rows[start_i:start_i + batch_size]
            vectors = self.embeddings.embed_documents([row["question"] for row in batch_rows])
            rows_to_insert = [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "category": row["category"],
                    "embedding": vector,
                }
                for row, vector in zip(batch_rows, vectors)
            ]
            total_inserted += self.db.insert_faq_items(rows_to_insert)

        return total_inserted

    def ingest_pdfs(self, files, batch_size: int = 64) -> tuple[int, int]:
        if not files:
            return 0, 0

        total_chunks = 0
        for file in files:
            filename = Path(file.name).name
            doc_id = self.db.create_document(filename)

            docs = PyPDFLoader(file.name).load()
            splits = self.splitter.split_documents(docs)

            chunk_texts: list[str] = []
            chunk_pages: list[int | None] = []
            for doc in splits:
                text = doc.page_content.strip()
                if not text:
                    continue
                chunk_texts.append(text)
                chunk_pages.append(doc.metadata.get("page"))

            for start_i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[start_i:start_i + batch_size]
                batch_pages = chunk_pages[start_i:start_i + batch_size]
                vectors = self.embeddings.embed_documents(batch_texts)
                rows = [
                    {
                        "document_id": doc_id,
                        "chunk_index": start_i + j,
                        "page": page,
                        "content": text,
                        "embedding": vector,
                    }
                    for j, (text, page, vector) in enumerate(zip(batch_texts, batch_pages, vectors))
                ]
                self.db.insert_chunks(rows)

            total_chunks += len(chunk_texts)

        return len(files), total_chunks
