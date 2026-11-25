
import os
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from pathlib import Path
from tempfile import TemporaryDirectory

from database.db import Database
from config import settings
from core.llm import LLMService
from agents.workflow import build_graph

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


def main():
    print("\n===== EDUPILOT PERFORMANCE TEST =====")

    with TemporaryDirectory() as temp:
        db = Database(Path(temp) / "performance.sqlite")

        user_id = "performance_test_user"
        subject_id = db.add_subject(user_id, "CSE")

        # 1. Create topics
        topic_names = [
            "Machine Learning",
            "Deep Learning",
            "Computer Vision"
        ]

        db.add_topics(user_id, subject_id, topic_names)

        # 2. Insert sample quiz results
        test_scores = {
            "Machine Learning": [0.2, 0.3, 0.4],
            "Deep Learning": [0.8, 0.9, 0.85],
            "Computer Vision": [0.6, 0.65, 0.7]
        }

        for topic, scores in test_scores.items():
            quiz_id = db.save_quiz(
                user_id,
                subject_id,
                topic,
                {"topic": topic, "questions": []}
            )

            rows = []

            for i, score in enumerate(scores, start=1):
                rows.append({
                    "topic": topic,
                    "question": f"Sample question {i}",
                    "correct_answer": "Correct answer",
                    "user_answer": "Student answer",
                    "score": score
                })

            db.save_results(user_id, subject_id, quiz_id, rows)

        print("[1] Sample performance data: CREATED")
        print("[2] SQLite results:", db.topic_scores(user_id, subject_id))

        # 3. Initialize Groq
        groq = ChatGroq(
            model=settings.FALLBACK_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            timeout=30,
            max_retries=1
        )

        llm = LLMService(primary=groq)

        # Retrieval is not needed for these three branches.
        graph = build_graph(
            db=db,
            llm=llm,
            retriever=None
        )

        print("[3] LangGraph: COMPILED")

        def run_question(question):
            state = {
                "user_id": user_id,
                "subject_id": subject_id,
                "question": question,
                "messages": [HumanMessage(content=question)]
            }
            return graph.invoke(state)

        # 4. Performance analysis
        print("\n===== PERFORMANCE BRANCH =====")

        performance = run_question(
            "Analyze my performance in this subject."
        )

        print("Intent:", performance.get("intent"))
        print("Answer:")
        print(performance.get("answer"))

        # 5. Next topic recommendation
        print("\n===== NEXT TOPIC BRANCH =====")

        recommendation = run_question(
            "What should I study next?"
        )

        print("Intent:", recommendation.get("intent"))
        print("Answer:")
        print(recommendation.get("answer"))

        # 6. Study plan
        print("\n===== STUDY PLAN BRANCH =====")

        plan_result = run_question(
            "Create a study plan for me."
        )

        print("Intent:", plan_result.get("intent"))
        print("Answer:")
        print(plan_result.get("answer"))

        # 7. Verify saved plan
        saved_plan = db.latest_plan(user_id, subject_id)

        print("\nSaved study plan exists:", saved_plan is not None)

        if saved_plan:
            print("Scheduled sessions:", len(saved_plan.get("days", [])))

        # 8. Verify persistent memory
        profile = db.get_profile(user_id)

        print("\n===== LEARNING PROFILE =====")
        print(profile)

        # 9. Final validation
        checks = {
            "Performance branch": (
                performance.get("intent") == "performance"
                and "Machine Learning" in performance.get("answer", "")
            ),
            "Next topic branch": (
                recommendation.get("intent") == "next_topic"
                and "Machine Learning" in recommendation.get("answer", "")
            ),
            "Study plan branch": (
                plan_result.get("intent") == "plan"
                and "7-day plan" in plan_result.get("answer", "")
            ),
            "Study plan saved": saved_plan is not None,
            "Learning profile saved": profile is not None,
            "Weak topic identified": (
                "Machine Learning" in profile.get("weak_topics", [])
            ),
            "Strong topic identified": (
                "Deep Learning" in profile.get("strong_topics", [])
            )
        }

        print("\n===== FINAL VALIDATION =====")

        for name, passed in checks.items():
            print(name + ":", "PASS" if passed else "FAIL")

        if all(checks.values()):
            print("\nEDUPILOT LEARNING WORKFLOW: SUCCESS")
        else:
            print("\nEDUPILOT LEARNING WORKFLOW: CHECK FAILED ITEMS")


if __name__ == "__main__":
    main()
