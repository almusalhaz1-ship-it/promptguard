# PromptGuard

### An LLM-as-a-Judge framework for evaluating AI-generated responses

PromptGuard is a Python-based evaluation system that uses Claude as an independent LLM judge to evaluate AI-generated responses across multiple dimensions of quality, safety, and usefulness.

The system evaluates responses across:

- Safety
- Accuracy
- Relevance
- Helpfulness
- Tone & Clarity
- Judge Confidence

PromptGuard separates **LLM-based qualitative evaluation** from **deterministic scoring**. Claude provides structured judgments and evidence, while Python validates the output and calculates the final weighted score.

---

## Core Idea

> Let the LLM provide judgment and evidence, but do not let the LLM control the final scoring logic.

The final score is calculated by Python using predefined weights:

| Criterion | Weight |
|---|---:|
| Safety | 30% |
| Accuracy | 25% |
| Helpfulness | 20% |
| Relevance | 15% |
| Tone & Clarity | 10% |

The result is converted to a 0–100 score.

---

## Key Features

- Claude-based LLM-as-a-Judge evaluation
- Structured JSON evaluation
- Safety-aware judging
- Prompt-injection-aware evaluator instructions
- Deterministic weighted scoring
- Evaluation confidence score
- Risk classification
- Key issue detection
- Improvement recommendations
- SQLite persistence
- Evaluation history
- Individual evaluation reports
- Evaluation deletion

---

## Tech Stack

- Python
- Flask
- Anthropic Python SDK
- Claude
- SQLite
- HTML
- CSS
- Jinja2
- Git / GitHub

---

## Project Status

**Current status: Functional MVP**

The core evaluation pipeline, persistent storage, evaluation history, detailed reports, and deletion functionality are currently implemented.

Further development will focus on evaluation reliability, benchmarking, testing, and production deployment.