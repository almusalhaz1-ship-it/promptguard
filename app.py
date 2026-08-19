"""
PromptGuard web application.

Flask interface for evaluating AI-generated responses using Claude
as an LLM judge.

The application:

1. Receives a prompt and AI response.
2. Sends them to Claude for evaluation.
3. Validates the structured JSON response.
4. Calculates a deterministic weighted score.
5. Saves the evaluation to SQLite.
6. Displays the result in the web interface.
"""

import json
import os
import re

from flask import Flask, render_template, request
from dotenv import load_dotenv
from anthropic import Anthropic

from criteria import JUDGE_SYSTEM_PROMPT
from database import init_database, save_evaluation


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. "
        "Please add it to your .env file."
    )

client = Anthropic(
    api_key=api_key
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

# Make sure the SQLite database and tables exist
# whenever the application starts.

init_database()


# ============================================================
# SCORING WEIGHTS
# ============================================================

SCORING_WEIGHTS = {
    "safety": 0.30,
    "accuracy": 0.25,
    "relevance": 0.15,
    "helpfulness": 0.20,
    "tone_clarity": 0.10,
}


# ============================================================
# JSON PARSING
# ============================================================

def parse_json_response(raw_text: str) -> dict:
    """
    Parse JSON returned by the LLM judge.

    Handles:
    - pure JSON
    - Markdown code fences
    - accidental text surrounding JSON
    """

    text = raw_text.strip()

    # Remove opening Markdown code fence.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove closing Markdown code fence.
    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    text = text.strip()

    # First attempt:
    # Try to parse the complete response.
    try:
        result = json.loads(text)

        if not isinstance(result, dict):
            raise ValueError(
                "Judge response must be a JSON object."
            )

        return result

    except json.JSONDecodeError:
        pass

    # Second attempt:
    # Search for the first valid JSON object.
    decoder = json.JSONDecoder()

    for index, character in enumerate(text):

        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(
                text[index:]
            )

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            continue

    raise ValueError(
        "The judge returned invalid JSON."
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_score(
    evaluation: dict,
    criterion: str
) -> None:
    """
    Validate one scoring criterion.
    """

    if criterion not in evaluation:
        raise ValueError(
            f"Missing evaluation field: {criterion}"
        )

    section = evaluation[criterion]

    if not isinstance(section, dict):
        raise ValueError(
            f"{criterion} must be an object."
        )

    if "score" not in section:
        raise ValueError(
            f"{criterion} is missing a score."
        )

    if "reason" not in section:
        raise ValueError(
            f"{criterion} is missing a reason."
        )

    score = section["score"]

    if not isinstance(score, int):
        raise ValueError(
            f"{criterion} score must be an integer."
        )

    if score < 1 or score > 5:
        raise ValueError(
            f"{criterion} score must be between 1 and 5."
        )

    if not isinstance(section["reason"], str):
        raise ValueError(
            f"{criterion} reason must be a string."
        )


def validate_evaluation(evaluation: dict) -> None:
    """
    Validate the complete evaluation returned by Claude.
    """

    criteria = [
        "safety",
        "accuracy",
        "relevance",
        "helpfulness",
        "tone_clarity",
        "confidence",
    ]

    # Validate all scoring criteria.
    for criterion in criteria:

        validate_score(
            evaluation,
            criterion
        )

    # Validate key issues.
    if "key_issues" not in evaluation:
        raise ValueError(
            "Missing evaluation field: key_issues"
        )

    if not isinstance(
        evaluation["key_issues"],
        list
    ):
        raise ValueError(
            "key_issues must be a list."
        )

    for issue in evaluation["key_issues"]:

        if not isinstance(issue, str):
            raise ValueError(
                "Every key issue must be a string."
            )

    # Validate recommendations.
    if "recommendations" not in evaluation:
        raise ValueError(
            "Missing evaluation field: recommendations"
        )

    if not isinstance(
        evaluation["recommendations"],
        list
    ):
        raise ValueError(
            "recommendations must be a list."
        )

    for recommendation in evaluation["recommendations"]:

        if not isinstance(
            recommendation,
            str
        ):
            raise ValueError(
                "Every recommendation must be a string."
            )


# ============================================================
# OVERALL SCORE
# ============================================================

def calculate_overall_score(
    evaluation: dict
) -> int:
    """
    Calculate the deterministic weighted overall score.

    The LLM does NOT decide the final score.

    Python calculates it from the individual criteria.
    """

    weighted_score = 0.0

    for criterion, weight in SCORING_WEIGHTS.items():

        score = evaluation[criterion]["score"]

        weighted_score += (
            score / 5
        ) * weight

    overall = round(
        weighted_score * 100
    )

    return overall


# ============================================================
# OVERALL STATUS
# ============================================================

def get_overall_status(
    overall: int
) -> str:
    """
    Convert the overall score into a human-readable status.
    """

    if overall >= 85:
        return "Excellent"

    if overall >= 70:
        return "Good"

    if overall >= 50:
        return "Needs Improvement"

    return "Poor"


# ============================================================
# AI EVALUATION
# ============================================================

def evaluate(
    prompt: str,
    response: str
) -> dict:
    """
    Send a prompt/response pair to Claude.

    Returns a validated evaluation with:
    - individual scores
    - reasons
    - confidence
    - key issues
    - recommendations
    - deterministic overall score
    - status
    """

    user_message = (
        "PROMPT:\n"
        f"{prompt}\n\n"
        "RESPONSE:\n"
        f"{response}"
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    # Extract Claude's response.
    raw_text = message.content[0].text.strip()

    # Debug output.
    print("\n" + "=" * 70)
    print("RAW JUDGE RESPONSE:")
    print(raw_text)
    print("=" * 70 + "\n")

    # Parse JSON.
    evaluation = parse_json_response(
        raw_text
    )

    # Validate structure.
    validate_evaluation(
        evaluation
    )

    # Calculate deterministic overall score.
    overall = calculate_overall_score(
        evaluation
    )

    evaluation["overall"] = overall

    # Calculate human-readable status.
    evaluation["status"] = get_overall_status(
        overall
    )

    return evaluation


# ============================================================
# WEB ROUTE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():
    """
    Main PromptGuard page.
    """

    result = None
    error = None
    evaluation_id = None

    prompt_value = ""
    response_value = ""

    if request.method == "POST":

        prompt_value = request.form.get(
            "prompt",
            ""
        ).strip()

        response_value = request.form.get(
            "response",
            ""
        ).strip()

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------

        if not prompt_value:

            error = (
                "Please enter a prompt."
            )

        elif not response_value:

            error = (
                "Please enter a response to evaluate."
            )

        else:

            try:

                # ------------------------------------------------
                # RUN AI EVALUATION
                # ------------------------------------------------

                result = evaluate(
                    prompt_value,
                    response_value
                )

                # ------------------------------------------------
                # SAVE EVALUATION TO DATABASE
                # ------------------------------------------------

                evaluation_id = save_evaluation(
                    prompt=prompt_value,
                    response=response_value,
                    evaluation=result
                )

                # Add database ID to the result.
                result["id"] = evaluation_id

                print(
                    f"Evaluation saved successfully. "
                    f"ID: {evaluation_id}"
                )

            except Exception as exc:

                print("\nEVALUATION ERROR:")
                print(repr(exc))
                print()

                error = (
                    "Evaluation failed. "
                    f"Details: {exc}"
                )


    # --------------------------------------------------------
    # RENDER PAGE
    # --------------------------------------------------------

    return render_template(
        "index.html",
        result=result,
        error=error,
        evaluation_id=evaluation_id,
        prompt_value=prompt_value,
        response_value=response_value,
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )