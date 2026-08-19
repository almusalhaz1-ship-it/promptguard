"""
PromptGuard database layer.

Stores AI evaluation results in a local SQLite database.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "promptguard.db"


# ============================================================
# CONNECTION
# ============================================================

def get_connection():
    """
    Create and return a SQLite database connection.
    """

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():
    """
    Create the evaluations table if it does not exist.

    Also performs a lightweight migration for existing databases.
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

                risk_level TEXT NOT NULL DEFAULT 'UNKNOWN',

                key_issues TEXT NOT NULL,

                recommendations TEXT NOT NULL,

                created_at TEXT NOT NULL
            )
            """
        )

        # ----------------------------------------------------
        # Migration:
        # Add risk_level to older databases if necessary.
        # ----------------------------------------------------

        columns = connection.execute(
            """
            PRAGMA table_info(evaluations)
            """
        ).fetchall()

        column_names = {
            column["name"]
            for column in columns
        }

        if "risk_level" not in column_names:

            connection.execute(
                """
                ALTER TABLE evaluations
                ADD COLUMN risk_level TEXT NOT NULL DEFAULT 'UNKNOWN'
                """
            )

        connection.commit()

    finally:

        connection.close()


# ============================================================
# SAVE EVALUATION
# ============================================================

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
                risk_level,
                key_issues,
                recommendations,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

                evaluation.get(
                    "risk_level",
                    "UNKNOWN"
                ),

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


# ============================================================
# GET SINGLE EVALUATION
# ============================================================

def get_evaluation(
    evaluation_id: int
):
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


# ============================================================
# GET RECENT EVALUATIONS
# ============================================================

def get_recent_evaluations(
    limit: int = 50
):
    """
    Retrieve the most recent evaluations.
    """

    connection = get_connection()

    try:

        rows = connection.execute(
            """
            SELECT
                id,
                prompt,
                overall_score,
                status,
                risk_level,
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


# ============================================================
# DELETE EVALUATION
# ============================================================

def delete_evaluation(
    evaluation_id: int
) -> bool:
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