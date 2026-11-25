
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import pymupdf

from config import settings
from database.db import Database
from rag.ingest import ingest_file
from rag.retriever import Retriever
from rag.embeddings import get_embeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from core.llm import LLMService
from core.schemas import Quiz
from agents.workflow import build_graph
from agents.tools import build_tools


def main():
    print("\n===== EDUPILOT QUIZ WORKFLOW TEST =====")

    with TemporaryDirectory() as temp:
        root = Path(temp)
        client = None

        old_data = settings.DATA_DIR
        old_upload = settings.UPLOAD_DIR
        old_export = settings.EXPORT_DIR

        try:
            # 1. Temporary directories
            settings.DATA_DIR = root / "data"
            settings.UPLOAD_DIR = settings.DATA_DIR / "uploads"
            settings.EXPORT_DIR = settings.DATA_DIR / "exports"

            # 2. Create sample study PDF
            pdf_path = root / "machine_learning.pdf"
            pdf = pymupdf.open()

            contents = [
                (
                    "Chapter 1: Machine Learning\n"
                    "Machine Learning is a branch of artificial intelligence. "
                    "It enables computers to learn patterns from data without "
                    "being explicitly programmed for every task. "
                    "A machine learning model learns from examples and "
                    "uses the learned patterns to make predictions."
                ),
                (
                    "Supervised Learning\n"
                    "Supervised learning uses labeled training data. "
                    "Classification predicts discrete categories, such as "
                    "spam or not spam. Regression predicts continuous values, "
                    "such as house prices. "
                    "The model learns the relationship between input features "
                    "and known target labels."
                ),
                (
                    "Unsupervised Learning\n"
                    "Unsupervised learning works with unlabeled data. "
                    "Clustering groups similar data points together. "
                    "Dimensionality reduction decreases the number of features "
                    "while preserving useful information. "
                    "These methods help discover hidden structures in data."
                ),
                (
                    "Reinforcement Learning\n"
                    "Reinforcement learning involves an agent interacting "
                    "with an environment. The agent takes actions and receives "
                    "rewards or penalties. "
                    "The goal is to learn a policy that maximizes cumulative "
                    "reward over time."
                ),
                (
                    "Model Evaluation\n"
                    "A dataset can be divided into training, validation and "
                    "test sets. Training data is used to learn model parameters. "
                    "Validation data helps select model settings. "
                    "Test data evaluates performance on unseen examples. "
                    "Overfitting occurs when a model performs well on training "
                    "data but poorly on unseen data."
                )
            ]

            for content in contents:
                page = pdf.new_page()
                page.insert_textbox(
                    pymupdf.Rect(60, 60, 535, 760),
                    content,
                    fontsize=11
                )

            pdf.save(pdf_path)
            pdf.close()

            print("[1] Sample PDF: CREATED")

            # 3. Temporary SQLite database
            db = Database(root / "quiz_test.sqlite")

            user_id = "quiz_test_user"
            subject_id = db.add_subject(user_id, "CSE")

            print("[2] SQLite: INITIALIZED")

            # 4. Temporary Qdrant database
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

            # 5. Ingest PDF
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

            # 6. Initialize Groq
            groq = ChatGroq(
                model=settings.FALLBACK_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0,
                timeout=30,
                max_retries=1
            )

            llm = LLMService(primary=groq)
            retriever = Retriever(store=store)

            # 7. Build LangGraph
            graph = build_graph(
                db=db,
                llm=llm,
                retriever=retriever
            )

            print("\n[5] LangGraph: COMPILED")

            # 8. Student requests a quiz
            question = (
                "Create a 5-question quiz about "
                "Machine Learning from my uploaded material."
            )

            initial_state = {
                "user_id": user_id,
                "subject_id": subject_id,
                "question": question,
                "messages": [
                    HumanMessage(content=question)
                ]
            }

            print("\n[6] STUDENT REQUEST:")
            print(question)

            # 9. Execute Quiz branch
            result = graph.invoke(initial_state)

            print("\n===== QUIZ WORKFLOW RESULT =====")
            print("Intent:", result.get("intent"))
            print("Answer:", result.get("answer"))

            if result.get("intent") != "quiz":
                raise RuntimeError(
                    "Expected quiz intent, received: "
                    + str(result.get("intent"))
                )

            # 10. Extract generated quiz ID
            match = re.search(
                r"Quiz #(\d+)",
                result.get("answer", "")
            )

            if not match:
                raise RuntimeError(
                    "Quiz was not generated: "
                    + str(result.get("answer"))
                )

            quiz_id = int(match.group(1))

            # 11. Verify SQLite quiz storage
            quiz_data = db.get_quiz(user_id, quiz_id)

            if quiz_data is None:
                raise RuntimeError("Quiz was not saved in SQLite.")

            quiz = Quiz.model_validate(quiz_data)

            print("\n===== GENERATED QUIZ =====")
            print("Quiz ID:", quiz_id)
            print("Topic:", quiz.topic)
            print("Question count:", len(quiz.questions))

            for index, q in enumerate(quiz.questions, start=1):
                print(f"\nQuestion {index}: {q.question}")
                print("Type:", q.qtype)
                print("Options:", q.options)
                print("Answer:", q.answer)

            # 12. Create tools for answer evaluation
            tools = build_tools(db, retriever, llm)

            # Simulate a learner submitting all correct answers.
            answers = [q.answer for q in quiz.questions]

            evaluation = tools["record_quiz_result"].invoke({
                "quiz_id": quiz_id,
                "answers": answers,
                "user_id": user_id,
                "subject_id": subject_id
            })

            print("\n===== QUIZ EVALUATION =====")
            print("Score:", evaluation["score"])
            print("Evaluated questions:", len(evaluation["rows"]))

            # 13. Verify results saved in SQLite
            scores = db.topic_scores(user_id, subject_id)

            print("\n===== SAVED QUIZ RESULTS =====")
            print(scores)

            # 14. Final validation
            checks = {
                "PDF ingestion": ingestion["chunks"] > 0,
                "Quiz intent": result.get("intent") == "quiz",
                "Quiz generated": len(quiz.questions) > 0,
                "Quiz saved in SQLite": quiz_data is not None,
                "All answers evaluated": (
                    len(evaluation["rows"]) == len(quiz.questions)
                ),
                "Quiz results saved": len(scores) > 0
            }

            # The prompt requests exactly five questions.
            checks["Five questions generated"] = (
                len(quiz.questions) == 5
            )

            print("\n===== FINAL VALIDATION =====")

            for name, passed in checks.items():
                print(
                    name + ":",
                    "PASS" if passed else "FAIL"
                )

            if all(checks.values()):
                print("\nEDUPILOT QUIZ WORKFLOW: SUCCESS")
            else:
                print("\nEDUPILOT QUIZ WORKFLOW: CHECK FAILED ITEMS")

        finally:
            if client is not None:
                client.close()

            settings.DATA_DIR = old_data
            settings.UPLOAD_DIR = old_upload
            settings.EXPORT_DIR = old_export


if __name__ == "__main__":
    main()
