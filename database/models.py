"""SQLite schema. Every learner-owned table carries user_id (isolation, C5)."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE(user_id, name)
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    path TEXT,
    kind TEXT NOT NULL DEFAULT 'material',   -- material | past_paper
    n_chunks INTEGER DEFAULT 0,
    flagged_chunks INTEGER DEFAULT 0,
    processed INTEGER DEFAULT 0,             -- past papers: questions extracted?
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    position INTEGER DEFAULT 0,              -- order of appearance = prerequisite order
    completed INTEGER DEFAULT 0,
    UNIQUE(user_id, subject_id, name)
);
CREATE TABLE IF NOT EXISTS quizzes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    topic TEXT,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS quiz_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    quiz_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    topic TEXT NOT NULL,
    question TEXT,
    correct_answer TEXT,
    user_answer TEXT,
    score REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS study_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS question_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    document_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    year INTEGER,
    marks REAL,
    qtype TEXT,
    topic TEXT
);
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""
