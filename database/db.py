"""Thin SQLite wrapper. Every read/write of learner data is scoped by user_id."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

from database.models import SCHEMA


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        with closing(self._conn()) as c:
            c.executescript(SCHEMA)
            c.commit()

            columns = {
                row["name"]
                for row in c.execute("PRAGMA table_info(documents)").fetchall()
            }

            if "file_hash" not in columns:
                c.execute("ALTER TABLE documents ADD COLUMN file_hash TEXT")
                c.commit()

    # -- plumbing -----------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        return c

    def _exec(self, sql: str, params: tuple = ()) -> int:
        with closing(self._conn()) as c:
            cur = c.execute(sql, params)
            c.commit()
            return cur.lastrowid or 0

    def _all(self, sql: str, params: tuple = ()) -> list[dict]:
        with closing(self._conn()) as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]

    # -- users & subjects ---------------------------------------------------
    def ensure_user(self, user_id: str) -> None:
        self._exec("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))

    def add_subject(self, user_id: str, name: str) -> int:
        self.ensure_user(user_id)
        self._exec("INSERT OR IGNORE INTO subjects(user_id, name) VALUES (?, ?)", (user_id, name))
        row = self._all("SELECT id FROM subjects WHERE user_id=? AND name=?", (user_id, name))
        return row[0]["id"]

    def list_subjects(self, user_id: str) -> list[dict]:
        return self._all("SELECT id, name FROM subjects WHERE user_id=? ORDER BY name", (user_id,))

    def subject_belongs_to(self, user_id: str, subject_id: int) -> bool:
        return bool(self._all("SELECT 1 FROM subjects WHERE id=? AND user_id=?", (subject_id, user_id)))

    def delete_subject(self, user_id: str, subject_id: int) -> list[dict]:
        """Delete a subject and all its related database records."""
        with closing(self._conn()) as c:
            with c:
                subject = c.execute(
                    "SELECT id FROM subjects WHERE id=? AND user_id=?",
                    (subject_id, user_id),
                ).fetchone()

                if not subject:
                    raise ValueError("Subject not found.")

                documents = [
                    dict(row) for row in c.execute(
                        "SELECT * FROM documents WHERE user_id=? AND subject_id=?",
                        (user_id, subject_id),
                    ).fetchall()
                ]

                c.execute(
                    "DELETE FROM question_patterns WHERE user_id=? AND subject_id=?",
                    (user_id, subject_id),
                )
                c.execute(
                    "DELETE FROM quiz_results WHERE user_id=? AND subject_id=?",
                    (user_id, subject_id),
                )

                for table in ("documents", "topics", "quizzes", "study_plans"):
                    c.execute(
                        f"DELETE FROM {table} WHERE user_id=? AND subject_id=?",
                        (user_id, subject_id),
                    )

                c.execute(
                    "DELETE FROM subjects WHERE id=? AND user_id=?",
                    (subject_id, user_id),
                )

            return documents

    # -- documents & topics -------------------------------------------------
    def find_document_by_hash(
        self, user_id: str, subject_id: int, file_hash: str
    ) -> Optional[dict]:
        rows = self._all(
            """
            SELECT * FROM documents
            WHERE user_id=? AND subject_id=? AND file_hash=?
            LIMIT 1
            """,
            (user_id, subject_id, file_hash),
        )
        return rows[0] if rows else None
    def add_document(
        self, user_id: str, subject_id: int, filename: str,
        path: str, kind: str, n_chunks: int = 0,
        flagged: int = 0, file_hash: Optional[str] = None
    ) -> int:

        if not self.subject_belongs_to(user_id, subject_id):
            raise ValueError("Subject does not belong to this user.")

        return self._exec(
            """
            INSERT INTO documents(
                user_id, subject_id, filename, path,
                kind, n_chunks, flagged_chunks, file_hash
            )
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (user_id, subject_id, filename, path, kind,
             n_chunks, flagged, file_hash),
        )
    def list_documents(self, user_id: str, subject_id: int, kind: Optional[str] = None) -> list[dict]:
        sql = "SELECT * FROM documents WHERE user_id=? AND subject_id=?"
        params: list[Any] = [user_id, subject_id]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        return self._all(sql + " ORDER BY id", tuple(params))

    def mark_document_processed(self, user_id: str, document_id: int) -> None:
        self._exec("UPDATE documents SET processed=1 WHERE id=? AND user_id=?", (document_id, user_id))

    def add_topics(self, user_id: str, subject_id: int, names: list[str]) -> None:
        existing = self._all("SELECT COUNT(*) AS n FROM topics WHERE user_id=? AND subject_id=?",
                             (user_id, subject_id))[0]["n"]
        for name in names:
            before = self._all("SELECT 1 FROM topics WHERE user_id=? AND subject_id=? AND name=?",
                               (user_id, subject_id, name))
            if not before:
                self._exec("INSERT INTO topics(user_id, subject_id, name, position) VALUES (?,?,?,?)",
                           (user_id, subject_id, name, existing))
                existing += 1

    def list_topics(self, user_id: str, subject_id: Optional[int] = None) -> list[dict]:
        if subject_id is None:
            return self._all("SELECT * FROM topics WHERE user_id=? ORDER BY subject_id, position", (user_id,))
        return self._all("SELECT * FROM topics WHERE user_id=? AND subject_id=? ORDER BY position",
                         (user_id, subject_id))

    def set_topic_completed(self, user_id: str, name: str, completed: bool) -> None:
        self._exec("UPDATE topics SET completed=? WHERE user_id=? AND name=?",
                   (1 if completed else 0, user_id, name))

    # -- quizzes ------------------------------------------------------------
    def save_quiz(self, user_id: str, subject_id: int, topic: str, payload: dict) -> int:
        return self._exec("INSERT INTO quizzes(user_id, subject_id, topic, payload) VALUES (?,?,?,?)",
                          (user_id, subject_id, topic, json.dumps(payload)))

    def get_quiz(self, user_id: str, quiz_id: int) -> Optional[dict]:
        rows = self._all("SELECT * FROM quizzes WHERE id=? AND user_id=?", (quiz_id, user_id))
        return json.loads(rows[0]["payload"]) if rows else None

    def save_results(self, user_id: str, subject_id: int, quiz_id: int, rows: list[dict]) -> None:
        for r in rows:
            self._exec(
                "INSERT INTO quiz_results(quiz_id, user_id, subject_id, topic, question, correct_answer,"
                " user_answer, score) VALUES (?,?,?,?,?,?,?,?)",
                (quiz_id, user_id, subject_id, r["topic"], r["question"], r["correct_answer"],
                 r["user_answer"], r["score"]),
            )

    def topic_scores(self, user_id: str, subject_id: Optional[int] = None,
                     last_n: int = 20) -> dict[str, list[float]]:
        """topic -> latest `last_n` answer scores (newest first)."""
        sql = "SELECT topic, score FROM quiz_results WHERE user_id=?"
        params: list[Any] = [user_id]
        if subject_id is not None:
            sql += " AND subject_id=?"
            params.append(subject_id)
        out: dict[str, list[float]] = {}
        for row in self._all(sql + " ORDER BY id DESC", tuple(params)):
            bucket = out.setdefault(row["topic"], [])
            if len(bucket) < last_n:
                bucket.append(row["score"])
        return out

    # -- plans & patterns ---------------------------------------------------
    def save_plan(self, user_id: str, subject_id: int, payload: dict) -> int:
        return self._exec("INSERT INTO study_plans(user_id, subject_id, payload) VALUES (?,?,?)",
                          (user_id, subject_id, json.dumps(payload)))

    def latest_plan(self, user_id: str, subject_id: int) -> Optional[dict]:
        rows = self._all("SELECT payload FROM study_plans WHERE user_id=? AND subject_id=? ORDER BY id DESC LIMIT 1",
                         (user_id, subject_id))
        return json.loads(rows[0]["payload"]) if rows else None

    def save_patterns(self, user_id: str, subject_id: int, document_id: int, patterns: list[dict]) -> None:
        for p in patterns:
            self._exec(
                "INSERT INTO question_patterns(user_id, subject_id, document_id, text, year, marks, qtype, topic)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (user_id, subject_id, document_id, p["text"], p.get("year"), p.get("marks"),
                 p.get("qtype"), p.get("topic")),
            )

    def list_patterns(self, user_id: str, subject_id: int) -> list[dict]:
        return self._all("SELECT * FROM question_patterns WHERE user_id=? AND subject_id=?",
                         (user_id, subject_id))

    def update_pattern_topic(self, user_id: str, pattern_id: int, topic: str) -> None:
        """Student correction of a classification."""
        self._exec("UPDATE question_patterns SET topic=? WHERE id=? AND user_id=?", (topic, pattern_id, user_id))

    # -- long-term profile (E1) --------------------------------------------
    def get_profile(self, user_id: str) -> Optional[dict]:
        rows = self._all("SELECT data FROM profiles WHERE user_id=?", (user_id,))
        return json.loads(rows[0]["data"]) if rows else None

    def save_profile(self, user_id: str, data: dict) -> None:
        self.ensure_user(user_id)

        self._exec(
            "INSERT INTO profiles(user_id, data) VALUES (?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=CURRENT_TIMESTAMP",
            (user_id, json.dumps(data)),
        )

    def delete_learning_history(self, user_id: str) -> None:
        """Wipes the learner's profile and history (needs explicit confirmation in the UI)."""
        for table in ("profiles", "quiz_results", "quizzes", "study_plans"):
            self._exec(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        self._exec("UPDATE topics SET completed=0 WHERE user_id=?", (user_id,))
