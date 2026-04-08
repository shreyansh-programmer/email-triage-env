---
title: Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - email-triage
  - rl-environment
pinned: true
license: bsd-3-clause
---

# 📧 Email Triage RL Environment — OpenEnv

> Train and evaluate AI agents on real-world inbox management: classifying priority, routing to departments, and composing responses under time pressure.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3-orange.svg)](LICENSE)

---

## 🌍 Why Email Triage?

**347 billion emails are sent daily worldwide.** Every knowledge worker spends significant time triaging their inbox — deciding what's urgent, who should handle it, and how to respond. This is a **universally relatable, high-impact** task that requires:

- **Multi-dimensional classification**: urgency × sender importance × topic × deadline
- **Department routing**: understanding organizational structure
- **Response selection**: escalate vs. acknowledge vs. delegate
- **Efficiency**: processing quickly without making mistakes

This environment lets AI agents practice these skills in a realistic, graded simulation.

---

## 🎯 Tasks

| Task | Difficulty | Emails | Max Steps | Objective |
|---|---|---|---|---|
| `priority_classification` | 🟢 Easy | 10 | 35 | Classify each email as HIGH / MEDIUM / LOW priority |
| `route_and_classify` | 🟡 Medium | 15 | 55 | Classify priority AND route to correct department |
| `full_triage` | 🔴 Hard | 20 | 80 | Classify + route + choose response type (with email threads) |

### Scoring

All scores are in the range `(0.0, 1.0)` with partial credit:

- **Priority**: exact match = 1.0, off-by-one = 0.5, wrong = 0.0
- **Routing**: exact match = 1.0, related department = 0.3, wrong = 0.0
- **Response**: exact match = 1.0, safe fallback = 0.2, wrong = 0.0
- **Efficiency bonus**: reward for using fewer steps

---

## 📐 Action Space

```python
class EmailTriageAction(Action):
    action_type: str       # "read" | "classify" | "route" | "respond" | "skip" | "finish"
    priority_label: str    # "HIGH" | "MEDIUM" | "LOW" (for classify)
    department: str        # "Engineering" | "Sales" | "Legal" | "HR" | "Support" | "Executive" (for route)
    response_type: str     # "acknowledge" | "escalate" | "delegate" | "decline" | "info_request" (for respond)
    response_text: str     # Optional free-text (for respond)
```

## 👁️ Observation Space

```python
class EmailTriageObservation(Observation):
    current_email_id: str       # ID of current email
    email_from: str             # Sender name, title, email
    email_subject: str          # Subject line
    email_body: str             # Full email body
    email_timestamp: str        # When received
    email_thread_id: str        # Thread ID (for related emails)
    total_emails: int           # Total in inbox
    emails_processed: int       # Already handled
    emails_remaining: int       # Left to process
    steps_used: int             # Steps consumed
    max_steps: int              # Step budget
    last_action_result: str     # Feedback from last action
    last_action_error: str      # Error if any
    task_name: str              # Current task
    task_description: str       # Task instructions
    available_actions: list     # Valid actions for this task
```

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.10
- Docker (for containerized deployment)
- `pip install openenv-core`

### Local Development

```bash
# Clone and navigate
cd email_triage_env

# Install dependencies
pip install -e .
# Or using uv:
uv sync

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Inference

```bash
# Set environment variables
export HF_TOKEN="your-hf-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run baseline agent
python inference.py
```

### Docker

```bash
# Build
cd email_triage_env
docker build -f server/Dockerfile -t email_triage_env:latest .

# Run
docker run -p 8000:8000 email_triage_env:latest
```

### Deploy to HF Spaces

```bash
cd email_triage_env
openenv push --repo-id your-username/email-triage-env
```

---

## 🏗️ Project Structure

```
hello/
├── inference.py                      # Baseline inference script
├── README.md                         # This file
└── email_triage_env/
    ├── __init__.py                   # Package exports
    ├── models.py                     # Typed Action & Observation models
    ├── client.py                     # EnvClient implementation
    ├── email_data.py                 # Deterministic email corpus generator
    ├── evaluation.py                 # Grading & reward functions
    ├── openenv.yaml                  # OpenEnv manifest
    ├── pyproject.toml                # Dependencies
    └── server/
        ├── __init__.py
        ├── app.py                    # FastAPI application
        ├── email_triage_env_environment.py  # Core environment logic
        ├── Dockerfile                # Container build
        └── requirements.txt
```

---

## 📊 Reward Design

### Per-Step Rewards
| Action | Reward |
|---|---|
| Read email | +0.05 |
| Correct priority classification | +0.30 |
| Off-by-one priority | +0.15 |
| Correct department routing | +0.25 |
| Related department | +0.075 |
| Correct response type | +0.20 |
| Classify without reading | -0.30 |
| Skip HIGH priority email | -0.40 |
| Invalid action | -0.05 |
| Duplicate classification | -0.10 |

### Final Score Weights
| Task | Priority | Routing | Response | Efficiency |
|---|---|---|---|---|
| Easy | 85% | — | — | 15% |
| Medium | 40% | 40% | — | 20% |
| Hard | 30% | 25% | 30% | 15% |

---

## 📈 Baseline Scores

| Task | Baseline Score | Steps Used |
|---|---|---|
| `priority_classification` | ~0.55 | ~25 |
| `route_and_classify` | ~0.45 | ~50 |
| `full_triage` | ~0.35 | ~75 |

*Scores measured with Qwen2.5-72B-Instruct. Better prompting and reasoning can improve these significantly.*

---

## ✅ Validation

```bash
# Run OpenEnv validation
cd email_triage_env
openenv validate

# Run pre-submission checks
./validate-submission.sh https://your-space.hf.space
```

---

## 📋 Environment Details

- **Runtime**: < 20 minutes for all 3 tasks
- **Memory**: < 100MB for environment server
- **Compatibility**: 2 vCPU, 8GB RAM sufficient
- **Dependencies**: Pure Python, no heavy ML libraries on server side
- **Deterministic**: Same seed produces identical email corpus

---

## 📜 License

BSD 3-Clause License
