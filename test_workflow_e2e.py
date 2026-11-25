
import os

# Disable LangSmith tracing for this isolated test
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from config import settings
from database.db import Database
from rag.embeddings import get_embeddings
from rag.ingest import ingest_file
from rag.retriever import Retriever

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.llm import LLMService
from agents.workflow import build_graph


def main():
    print("\n===== EDUPILOT WORKFLOW TEST =====")

    with TemporaryDirectory() as temp:
        root = Path(temp)
        client = None

        old_data = settings.DATA_DIR
        old_upload = settings.UPLOAD_DIR
        old_export = settings.EXPORT_DIR

        try:
            # 1. Temporary project directories
            settings.DATA_DIR = root / "data"
            settings.UPLOAD_DIR = settings.DATA_DIR / "uploads"
            settings.EXPORT_DIR = settings.DATA_DIR / "exports"

            # 2. Create sample PDF
            pdf_path = root / "sample.pdf"

            pdf = pymupdf.open()

            page1 = pdf.new_page()
            page1.insert_textbox(
                pymupdf.Rect(72, 72, 520, 300),
                "Chapter 1: Introduction\n"
                "Artificial Intelligence is a branch of computer science.\n"
                "AI systems can solve intelligent tasks."
            )

            page2 = pdf.new_page()
            page2.insert_textbox(
                pymupdf.Rect(72, 72, 520, 300),
                "Machine Learning\n"
                "Machine learning is a branch of artificial intelligence.\n"
                "It enables computers to learn patterns from data."
            )

            pdf.save(pdf_path)
            pdf.close()

            print("\n[1] Sample PDF: CREATED")

            # 3. Temporary SQLite database
            db = Database(root / "test.sqlite")
            user_id = "workflow_test_user"
            subject_id = db.add_subject(user_id, "CSE")

            print("[2] SQLite database: INITIALIZED")

            # 4. Temporary Qdrant
            client = QdrantClient(
                path=str(root / "qdrant")
            )

            embedding = get_embeddings()
            dimension = len(
                embedding.embed_query("dimension probe")
            )

            client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE
                )
            )

            store = QdrantVectorStore(
                client=client,
                collection_name=settings.COLLECTION_NAME,
                embedding=embedding
            )

            print("[3] Qdrant: INITIALIZED")

            # 5. Ingest the PDF
            ingestion = ingest_file(
                src_path=pdf_path,
                user_id=user_id,
                subject_id=subject_id,
                kind="material",
                db=db,
                store=store
            )

            print("\n[4] PDF INGESTION:")
            print(ingestion)

            # 6. Create Retriever
            retriever = Retriever(store=store)

            # 7. Use Groq as the primary model for this test
            groq = ChatGroq(
                model=settings.FALLBACK_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0,
                timeout=30,
                max_retries=1
            )

            llm = LLMService(primary=groq)

            print("\n[5] Groq LLM: INITIALIZED")

            # 8. Build complete LangGraph workflow
            graph = build_graph(
                db=db,
                llm=llm,
                retriever=retriever
            )

            print("[6] LangGraph: COMPILED")

            # 9. Student question
            question = "What is Machine Learning?"

            initial_state = {
                "user_id": user_id,
                "subject_id": subject_id,
                "question": question,
                "messages": [
                    HumanMessage(content=question)
                ]
            }

            print("\n[7] STUDENT QUESTION:")
            print(question)

            # 10. Execute complete workflow
            result = graph.invoke(initial_state)

            print("\n===== WORKFLOW RESULT =====")

            print("\nIntent:")
            print(result.get("intent"))

            print("\nRewritten question:")
            print(result.get("rewritten"))

            print("\nRetrieved chunks:")
            chunks = result.get("context", [])
            print("Total:", len(chunks))

            for chunk in chunks:
                print({
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "topic": chunk.get("topic"),
                    "score": chunk.get("score")
                })

            print("\nGenerated answer:")
            print(result.get("answer"))

            # 11. Validate result
            answer = result.get("answer", "")

            checks = {
                "PDF ingestion": ingestion["chunks"] > 0,
                "Intent classification": result.get("intent") == "study",
                "Qdrant retrieval": len(chunks) > 0,
                "Answer generation": bool(answer),
                "PDF source citation": "sample.pdf" in answer,
                "Page citation": "p." in answer
            }

            print("\n===== FINAL VALIDATION =====")

            for name, passed in checks.items():
                print(
                    name + ":",
                    "PASS" if passed else "FAIL"
                )

            if all(checks.values()):
                print("\nEDUPILOT STUDY WORKFLOW: SUCCESS")
            else:
                print("\nEDUPILOT STUDY WORKFLOW: CHECK FAILED ITEMS")

        finally:
            if client is not None:
                client.close()

            settings.DATA_DIR = old_data
            settings.UPLOAD_DIR = old_upload
            settings.EXPORT_DIR = old_export


if __name__ == "__main__":
    main()
