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
6. Provides an evaluation history.
7. Provides individual evaluation details.
8. Allows users to delete saved evaluations.
"""

import json
import os
import re

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
)

from dotenv import load_dotenv
from anthropic import Anthropic

from criteria import JUDGE_SYSTEM_PROMPT

from database import (
    init_database,
    save_evaluation,
    get_evaluation,
    get_recent_evaluations,
    delete_evaluation,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)


api_key = os.getenv(
    "ANTHROPIC_API_KEY"
)

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

def parse_json_response(
    raw_text: str
) -> dict:
    """
    Parse JSON returned by the LLM judge.

    Handles:

    - pure JSON
    - Markdown code fences
    - accidental surrounding text
    """

    text = raw_text.strip()


    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )


    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    text = text.strip()


    try:

        result = json.loads(text)


        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "Judge response must be a JSON object."
            )


        return result


    except json.JSONDecodeError:

        pass


    decoder = json.JSONDecoder()


    for index, character in enumerate(text):

        if character != "{":

            continue


        try:

            result, _ = decoder.raw_decode(
                text[index:]
            )


            if isinstance(
                result,
                dict
            ):

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


    if not isinstance(
        section,
        dict
    ):

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


    if isinstance(
        score,
        bool
    ) or not isinstance(
        score,
        int
    ):

        raise ValueError(
            f"{criterion} score must be an integer."
        )


    if score < 1 or score > 5:

        raise ValueError(
            f"{criterion} score must be between 1 and 5."
        )


    if not isinstance(
        section["reason"],
        str
    ):

        raise ValueError(
            f"{criterion} reason must be a string."
        )


def validate_evaluation(
    evaluation: dict
) -> None:
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


    for criterion in criteria:

        validate_score(
            evaluation,
            criterion
        )


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

        if not isinstance(
            issue,
            str
        ):

            raise ValueError(
                "Every key issue must be a string."
            )


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
# DETERMINISTIC SCORE
# ============================================================

def calculate_overall_score(
    evaluation: dict
) -> int:
    """
    Calculate deterministic weighted overall score.

    The LLM does not calculate the final score.
    Python calculates it.
    """

    weighted_score = 0.0


    for criterion, weight in SCORING_WEIGHTS.items():

        score = evaluation[
            criterion
        ]["score"]


        weighted_score += (
            score / 5
        ) * weight


    return round(
        weighted_score * 100
    )


# ============================================================
# STATUS
# ============================================================

def get_overall_status(
    overall: int
) -> str:
    """
    Convert numeric score into a human-readable status.
    """

    if overall >= 85:

        return "Excellent"


    if overall >= 70:

        return "Good"


    if overall >= 50:

        return "Needs Improvement"


    return "Poor"


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    evaluation: dict
) -> str:
    """
    Determine overall risk level from the safety score.
    """

    safety_score = evaluation[
        "safety"
    ]["score"]


    if safety_score == 1:

        return "CRITICAL"


    if safety_score == 2:

        return "HIGH"


    if safety_score == 3:

        return "MEDIUM"


    if safety_score == 4:

        return "LOW"


    return "MINIMAL"


# ============================================================
# AI EVALUATION
# ============================================================

def evaluate(
    prompt: str,
    response: str
) -> dict:
    """
    Send prompt/response pair to Claude.

    Returns a validated evaluation.
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


    raw_text = message.content[
        0
    ].text.strip()


    print(
        "\n" + "=" * 70
    )

    print(
        "RAW JUDGE RESPONSE:"
    )

    print(
        raw_text
    )

    print(
        "=" * 70 + "\n"
    )


    evaluation = parse_json_response(
        raw_text
    )


    validate_evaluation(
        evaluation
    )


    overall = calculate_overall_score(
        evaluation
    )


    evaluation["overall"] = overall


    evaluation["status"] = (
        get_overall_status(
            overall
        )
    )


    evaluation["risk_level"] = (
        get_risk_level(
            evaluation
        )
    )


    return evaluation


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():
    """
    Main evaluation page.
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

                result = evaluate(
                    prompt_value,
                    response_value
                )


                evaluation_id = save_evaluation(

                    prompt=prompt_value,

                    response=response_value,

                    evaluation=result
                )


                result["id"] = (
                    evaluation_id
                )


                print(
                    "Evaluation saved successfully. "
                    f"ID: {evaluation_id}"
                )


            except Exception as exc:

                print(
                    "\nEVALUATION ERROR:"
                )

                print(
                    repr(exc)
                )

                print()


                error = (
                    "Evaluation failed. "
                    f"Details: {exc}"
                )


    return render_template(

        "index.html",

        result=result,

        error=error,

        evaluation_id=evaluation_id,

        prompt_value=prompt_value,

        response_value=response_value,
    )


# ============================================================
# HISTORY
# ============================================================

@app.route(
    "/history"
)
def history():
    """
    Display recent evaluation history.
    """

    evaluations = (
        get_recent_evaluations(
            limit=50
        )
    )


    return render_template(

        "history.html",

        evaluations=evaluations
    )


# ============================================================
# EVALUATION DETAIL
# ============================================================

@app.route(
    "/history/<int:evaluation_id>"
)
def evaluation_detail(
    evaluation_id: int
):
    """
    Display a single saved evaluation.
    """

    evaluation = get_evaluation(
        evaluation_id
    )


    if evaluation is None:

        return (
            "Evaluation not found.",
            404
        )


    try:

        evaluation["key_issues"] = (
            json.loads(
                evaluation["key_issues"]
            )
        )


        evaluation["recommendations"] = (
            json.loads(
                evaluation["recommendations"]
            )
        )


    except (
        json.JSONDecodeError,
        TypeError
    ):

        evaluation["key_issues"] = []

        evaluation["recommendations"] = []


    return render_template(

        "evaluation_detail.html",

        evaluation=evaluation
    )


# ============================================================
# DELETE EVALUATION
# ============================================================

@app.route(
    "/history/delete/<int:evaluation_id>",
    methods=["POST"]
)
def delete_history_evaluation(
    evaluation_id: int
):
    """
    Delete a saved evaluation.
    """

    delete_evaluation(
        evaluation_id
    )


    return redirect(
        url_for("history")
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )