import database


def sample_evaluation():
    return {
        "safety": {
            "score": 5,
            "reason": "Safe response."
        },
        "accuracy": {
            "score": 4,
            "reason": "Mostly accurate."
        },
        "relevance": {
            "score": 5,
            "reason": "Directly relevant."
        },
        "helpfulness": {
            "score": 4,
            "reason": "Helpful response."
        },
        "tone_clarity": {
            "score": 5,
            "reason": "Clear and professional."
        },
        "confidence": {
            "score": 5,
            "reason": "High confidence."
        },
        "overall": 88,
        "status": "Excellent",
        "key_issues": [
            "Minor limitation."
        ],
        "recommendations": [
            "Provide slightly more detail."
        ]
    }


def test_database_initialization(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        db_path
    )

    database.init_database()

    connection = database.get_connection()

    try:
        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'evaluations'
            """
        ).fetchone()

        assert row is not None

    finally:
        connection.close()


def test_save_and_get_evaluation(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        db_path
    )

    database.init_database()

    evaluation = sample_evaluation()

    evaluation_id = database.save_evaluation(
        "Test prompt",
        "Test response",
        evaluation
    )

    assert evaluation_id == 1

    result = database.get_evaluation(
        evaluation_id
    )

    assert result is not None
    assert result["prompt"] == "Test prompt"
    assert result["response"] == "Test response"
    assert result["safety_score"] == 5
    assert result["accuracy_score"] == 4
    assert result["overall_score"] == 88
    assert result["status"] == "Excellent"


def test_recent_evaluations(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        db_path
    )

    database.init_database()

    evaluation = sample_evaluation()

    database.save_evaluation(
        "Prompt 1",
        "Response 1",
        evaluation
    )

    database.save_evaluation(
        "Prompt 2",
        "Response 2",
        evaluation
    )

    results = database.get_recent_evaluations(
        limit=20
    )

    assert len(results) == 2
    assert results[0]["id"] == 2
    assert results[1]["id"] == 1


def test_delete_evaluation(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        db_path
    )

    database.init_database()

    evaluation = sample_evaluation()

    evaluation_id = database.save_evaluation(
        "Delete me",
        "This will be deleted",
        evaluation
    )

    deleted = database.delete_evaluation(
        evaluation_id
    )

    assert deleted is True

    result = database.get_evaluation(
        evaluation_id
    )

    assert result is None


def test_delete_nonexistent_evaluation(
    tmp_path,
    monkeypatch
):
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(
        database,
        "DATABASE_PATH",
        db_path
    )

    database.init_database()

    deleted = database.delete_evaluation(
        999
    )

    assert deleted is False