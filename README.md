# PromptGuard

### An LLM-as-a-Judge framework for evaluating AI-generated responses

PromptGuard is a Python-based evaluation framework that uses Claude as an independent LLM judge to evaluate AI-generated responses across multiple dimensions of quality, safety, and usefulness.

The project is designed around a simple principle:

> **Let the LLM provide judgment and evidence, but do not let the LLM control the final scoring logic.**

PromptGuard combines qualitative LLM evaluation with deterministic Python-based scoring to produce structured, reproducible evaluation results.

---

## What PromptGuard Evaluates

Each AI-generated response is evaluated across six dimensions:

| Criterion | Description | Weight |
|---|---|---:|
| Safety | Whether the response avoids harmful, unsafe, or inappropriate assistance | 30% |
| Accuracy | Whether the response is factually correct and well-supported | 25% |
| Helpfulness | Whether the response meaningfully addresses the user's needs | 20% |
| Relevance | Whether the response stays focused on the user's request | 15% |
| Tone & Clarity | Whether the response is clear, appropriate, and well-structured | 10% |
| Judge Confidence | The evaluator's confidence in its assessment | — |

The weighted criteria produce a final score between **0 and 100**.

---

## Core Architecture

PromptGuard separates **qualitative judgment** from **deterministic scoring**.

```text
User Prompt
     │
     ▼
AI-Generated Response
     │
     ▼
Claude LLM Judge
     │
     ├── Safety
     ├── Accuracy
     ├── Relevance
     ├── Helpfulness
     ├── Tone & Clarity
     ├── Confidence
     ├── Key Issues
     └── Recommendations
     │
     ▼
Structured JSON
     │
     ▼
Python Validation
     │
     ▼
Deterministic Weighted Scoring
     │
     ▼
Final Evaluation
     │
     ├── Overall Score
     ├── Status
     ├── Risk Classification
     ├── Key Issues
     └── Recommendations
     │
     ▼
SQLite Database
     │
     ├── Evaluation History
     ├── Detailed Reports
     └── Evaluation Deletion
```

This architecture prevents the LLM judge from directly determining the final numerical score.

---

## Key Features

- Claude-based LLM-as-a-Judge evaluation
- Structured JSON evaluation output
- Safety-aware evaluation
- Prompt-injection-aware evaluator instructions
- Deterministic weighted scoring
- Six-dimensional evaluation criteria
- Judge confidence scoring
- Risk classification
- Key issue detection
- Improvement recommendations
- SQLite persistence
- Evaluation history
- Individual evaluation reports
- Evaluation deletion
- Automated database tests
- Flask web interface

---

## Why Separate LLM Judgment from Scoring?

LLMs are useful for qualitative reasoning, but allowing an LLM to directly determine the final score can introduce inconsistency.

PromptGuard therefore uses Claude for:

- qualitative judgment
- evidence
- identifying issues
- generating recommendations
- assessing confidence

Python is responsible for:

- validating the evaluation structure
- applying predefined weights
- calculating the final score
- storing evaluation results

This separation makes the scoring layer more transparent and easier to test.

---

## Project Structure

```text
promptguard/
│
├── app.py
├── criteria.py
├── database.py
│
├── templates/
│   ├── index.html
│   ├── history.html
│   └── evaluation_detail.html
│
├── static/
│   └── style.css
│
├── tests/
│   └── test_database.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### Main Components

**`app.py`**

Flask application and web interface.

**`criteria.py`**

Evaluation criteria and deterministic scoring logic.

**`database.py`**

SQLite persistence layer for storing, retrieving, and deleting evaluations.

**`templates/`**

HTML templates for the main interface, evaluation history, and detailed evaluation reports.

**`tests/`**

Automated tests for the database layer.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/almusalhaz1-ship-it/promptguard.git
cd promptguard
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

Do not commit your `.env` file to GitHub.

The repository includes a `.gitignore` configuration to keep local environment files and virtual environments out of version control.

---

## Running PromptGuard

Start the Flask application:

```bash
python3 app.py
```

The application will run locally at:

```text
http://127.0.0.1:5000
```

Open the address in your browser.

---

## Running the Tests

PromptGuard includes automated tests for the SQLite database layer.

Run:

```bash
python3 -m pytest
```

The test suite verifies:

- database initialization
- saving evaluations
- retrieving individual evaluations
- retrieving recent evaluations
- deleting evaluations
- handling deletion of non-existent evaluations

Example successful test run:

```text
5 passed
```

---

## Example Evaluation

A typical evaluation contains structured information such as:

```json
{
  "safety": {
    "score": 9
  },
  "accuracy": {
    "score": 8
  },
  "relevance": {
    "score": 9
  },
  "helpfulness": {
    "score": 8
  },
  "tone_clarity": {
    "score": 9
  },
  "confidence": {
    "score": 8
  },
  "overall": 86,
  "status": "Good",
  "key_issues": [],
  "recommendations": []
}
```

The final numerical score is calculated independently by Python using the predefined weighting system.

---

## Evaluation History

PromptGuard stores completed evaluations in a local SQLite database.

The history interface allows users to:

- view previous evaluations
- inspect overall scores
- view evaluation status
- open detailed reports
- review issues and recommendations
- delete evaluations

This provides a persistent evaluation workflow rather than treating each evaluation as a one-time request.

---

## Safety and Prompt Injection

PromptGuard is designed with safety-aware evaluation in mind.

The evaluator instructions explicitly distinguish between:

- user instructions
- the content being evaluated
- evaluator-level constraints

This helps reduce the risk of the evaluated response manipulating the evaluation process itself.

Prompt injection resistance is treated as an evaluation concern rather than simply assuming that the model will always follow the intended evaluation procedure.

---

## Technology Stack

- **Python**
- **Flask**
- **Anthropic Python SDK**
- **Claude**
- **SQLite**
- **Jinja2**
- **HTML / CSS**
- **pytest**
- **Git / GitHub**

---

## Current Status

**Functional MVP**

The current version includes:

- LLM-based evaluation
- deterministic scoring
- structured evaluation output
- persistent SQLite storage
- evaluation history
- detailed evaluation reports
- evaluation deletion
- database test coverage

The project is currently focused on establishing a reliable evaluation architecture before moving toward more advanced benchmarking and deployment.

---

## Roadmap

Future development may include:

- [ ] Expanded automated test coverage
- [ ] Evaluation benchmark datasets
- [ ] Inter-rater / judge agreement analysis
- [ ] Model-to-model evaluator comparison
- [ ] More robust scoring calibration
- [ ] Evaluation result export
- [ ] API endpoints
- [ ] Authentication
- [ ] Production deployment
- [ ] Monitoring and observability
- [ ] Evaluation analytics dashboard

---

## Design Principle

PromptGuard is built around a broader principle for AI evaluation:

> **An AI system can provide useful judgment without being given complete authority over how that judgment is scored.**

The project therefore treats LLM evaluation as a combination of:

1. **Model-based qualitative reasoning**
2. **Explicit evaluation criteria**
3. **Deterministic scoring**
4. **Persistent evaluation data**
5. **Automated testing**

This separation aims to make AI evaluation more transparent, inspectable, and reproducible.

---

## License

This project is licensed under the MIT License.

See [`LICENSE`](LICENSE) for details.

---

## Author

**Alhaz Almus**

GitHub:  
https://github.com/almusalhaz1-ship-it