"""
PromptGuard database layer.

Stores AI evaluation results in a local SQLite database.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


# Database file will live in the project root.
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "promptguard.db"


def get_connection():
    """
    Create and return a SQLite database connection.
    """

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def init_database():
    """
    Create the evaluations table if it does not already exist.
    """

    connection = get_connection()

    try:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                prompt TEXT NOT NULL,

                response TEXT NOT NULL,

                safety_score INTEGER NOT NULL,

                accuracy_score INTEGER NOT NULL,

                relevance_score INTEGER NOT NULL,

                helpfulness_score INTEGER NOT NULL,

                tone_clarity_score INTEGER NOT NULL,

                confidence_score INTEGER NOT NULL,

                overall_score INTEGER NOT NULL,

                status TEXT NOT NULL,

                key_issues TEXT NOT NULL,

                recommendations TEXT NOT NULL,

                created_at TEXT NOT NULL
            )
            """
        )

        connection.commit()

    finally:

        connection.close()


def save_evaluation(
    prompt: str,
    response: str,
    evaluation: dict
) -> int:
    """
    Save an evaluation and return its database ID.
    """

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            INSERT INTO evaluations (
                prompt,
                response,
                safety_score,
                accuracy_score,
                relevance_score,
                helpfulness_score,
                tone_clarity_score,
                confidence_score,
                overall_score,
                status,
                key_issues,
                recommendations,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prompt,
                response,

                evaluation["safety"]["score"],

                evaluation["accuracy"]["score"],

                evaluation["relevance"]["score"],

                evaluation["helpfulness"]["score"],

                evaluation["tone_clarity"]["score"],

                evaluation["confidence"]["score"],

                evaluation["overall"],

                evaluation["status"],

                json.dumps(
                    evaluation["key_issues"],
                    ensure_ascii=False
                ),

                json.dumps(
                    evaluation["recommendations"],
                    ensure_ascii=False
                ),

                datetime.utcnow().isoformat()
            )
        )

        connection.commit()

        return cursor.lastrowid

    finally:

        connection.close()


def get_evaluation(evaluation_id: int):
    """
    Retrieve one evaluation by ID.
    """

    connection = get_connection()

    try:

        row = connection.execute(
            """
            SELECT *
            FROM evaluations
            WHERE id = ?
            """,
            (evaluation_id,)
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    finally:

        connection.close()


def get_recent_evaluations(limit: int = 20):
    """
    Retrieve the most recent evaluations.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT
                id,
                overall_score,
                status,
                created_at
            FROM evaluations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:

        connection.close()


def delete_evaluation(evaluation_id: int) -> bool:
    """
    Delete an evaluation by ID.

    Returns True if a record was deleted.
    """

    connection = get_connection()

    try:

        cursor = connection.execute(
            """
            DELETE FROM evaluations
            WHERE id = ?
            """,
            (evaluation_id,)
        )

        connection.commit()

        return cursor.rowcount > 0

    finally:

        connection.close()