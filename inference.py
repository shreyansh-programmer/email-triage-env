"""
Inference Script for Email Triage Environment
==============================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local Docker image for the environment.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from email_triage_env import EmailTriageAction, EmailTriageEnv

# ─── Configuration ──────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") # Set this in your environment or a .env file
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("EMAIL_TRIAGE_BENCHMARK", "email_triage_env")

TASKS = ["priority_classification", "route_and_classify", "full_triage"]
MAX_STEPS_MAP = {
    "priority_classification": 35,
    "route_and_classify": 55,
    "full_triage": 80,
}

TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.15


# ─── Logging helpers ────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── System prompts ────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "priority_classification": textwrap.dedent("""
        You are an AI email triage assistant. Your job is to read emails and classify
        their priority as HIGH, MEDIUM, or LOW.

        On each turn you will see the current email. You must respond with a JSON object
        specifying your action. Available actions:
        - {"action_type": "read"} — read the next email
        - {"action_type": "classify", "priority_label": "HIGH|MEDIUM|LOW"} — classify the current email
        - {"action_type": "skip"} — skip the current email
        - {"action_type": "finish"} — end the episode

        Workflow: read → classify → read → classify → ... → finish

        Priority guidelines:
        - HIGH: Urgent, time-sensitive, from VIPs, production issues, compliance deadlines
        - MEDIUM: Important but not time-critical, code reviews, pipeline updates, training
        - LOW: FYI, social events, newsletters, style guide updates

        Respond ONLY with valid JSON. No extra text.
    """).strip(),

    "route_and_classify": textwrap.dedent("""
        You are an AI email triage assistant. Your job is to read emails, classify priority
        (HIGH/MEDIUM/LOW), and route to the correct department.

        Available actions (respond with JSON):
        - {"action_type": "read"} — read the next email
        - {"action_type": "classify", "priority_label": "HIGH|MEDIUM|LOW"} — classify priority
        - {"action_type": "route", "department": "Engineering|Sales|Legal|HR|Support|Executive"} — route to department
        - {"action_type": "skip"} — skip current email
        - {"action_type": "finish"} — end the episode

        Workflow: read → classify → route → read → classify → route → ... → finish

        Departments:
        - Engineering: code, bugs, deployments, infrastructure, security
        - Sales: deals, pipeline, pricing, prospects
        - Legal: contracts, compliance, audits, regulations
        - HR: hiring, benefits, performance reviews, workplace issues
        - Support: customer issues, tickets, feedback
        - Executive: board meetings, strategy, CEO/CFO requests

        Respond ONLY with valid JSON.
    """).strip(),

    "full_triage": textwrap.dedent("""
        You are an AI email triage assistant performing full inbox triage. For each email:
        1) Classify priority (HIGH/MEDIUM/LOW)
        2) Route to department (Engineering/Sales/Legal/HR/Support/Executive)
        3) Choose the appropriate response type

        Available actions (respond with JSON):
        - {"action_type": "read"} — read the next email
        - {"action_type": "classify", "priority_label": "HIGH|MEDIUM|LOW"}
        - {"action_type": "route", "department": "Engineering|Sales|Legal|HR|Support|Executive"}
        - {"action_type": "respond", "response_type": "acknowledge|escalate|delegate|decline|info_request"}
        - {"action_type": "skip"} — skip current email
        - {"action_type": "finish"} — end the episode

        Workflow: read → classify → route → respond → read → ... → finish

        Response type guidelines:
        - escalate: Critical issues, production outages, VIP urgent requests
        - delegate: Tasks that should go to another team/person
        - acknowledge: Standard confirmation or FYI items
        - info_request: When you need more information
        - decline: Spam, irrelevant, or duplicate requests

        Respond ONLY with valid JSON.
    """).strip(),
}


# ─── LLM interaction ───────────────────────────────────────────

def build_user_prompt(obs_dict: Dict, step: int, history: List[str]) -> str:
    """Build the user prompt from the current observation."""
    parts = [f"Step {step} of {obs_dict.get('max_steps', '?')}"]
    parts.append(f"Emails processed: {obs_dict.get('emails_processed', 0)}/{obs_dict.get('total_emails', 0)}")
    parts.append(f"Last result: {obs_dict.get('last_action_result', 'N/A')}")

    if obs_dict.get("last_action_error"):
        parts.append(f"⚠ Error: {obs_dict['last_action_error']}")

    if obs_dict.get("email_subject"):
        parts.append(f"\n--- Current Email ---")
        parts.append(f"From: {obs_dict.get('email_from', 'Unknown')}")
        parts.append(f"Subject: {obs_dict.get('email_subject', '')}")
        parts.append(f"Time: {obs_dict.get('email_timestamp', '')}")
        if obs_dict.get("email_thread_id"):
            parts.append(f"Thread: {obs_dict['email_thread_id']}")
        parts.append(f"\n{obs_dict.get('email_body', '')}")
        parts.append(f"---")

    if history:
        parts.append(f"\nRecent actions: {'; '.join(history[-5:])}")

    parts.append("\nYour next action (JSON only):")
    return "\n".join(parts)


def get_model_action(
    client: OpenAI,
    task_name: str,
    obs_dict: Dict,
    step: int,
    history: List[str],
) -> Dict:
    """Query the LLM for the next action."""
    user_prompt = build_user_prompt(obs_dict, step, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_name]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            # Extract content between code blocks
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines).strip()

        # Try to parse JSON
        action = json.loads(text)
        return action

    except json.JSONDecodeError:
        # Fallback: try to extract action type from text
        text_lower = text.lower() if text else ""
        if "read" in text_lower:
            return {"action_type": "read"}
        elif "finish" in text_lower:
            return {"action_type": "finish"}
        elif "skip" in text_lower:
            return {"action_type": "skip"}
        else:
            return {"action_type": "read"}

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "read"}


def obs_to_dict(obs) -> Dict:
    """Convert observation object to dict for prompt building."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    elif hasattr(obs, "__dict__"):
        return {k: v for k, v in obs.__dict__.items() if not k.startswith("_")}
    elif isinstance(obs, dict):
        return obs
    return {}


# ─── Main loop ──────────────────────────────────────────────────

async def run_task(client_openai: OpenAI, env: EmailTriageEnv, task_name: str) -> float:
    """Run a single task and return the final score."""
    max_steps = MAX_STEPS_MAP.get(task_name, 50)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation
        obs_dict = obs_to_dict(obs)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            # Get action from LLM
            action_dict = get_model_action(client_openai, task_name, obs_dict, step, history)
            action_type = action_dict.get("action_type", "read")

            # Build action
            action = EmailTriageAction(
                action_type=action_type,
                priority_label=action_dict.get("priority_label"),
                department=action_dict.get("department"),
                response_type=action_dict.get("response_type"),
                response_text=action_dict.get("response_text"),
            )

            # Step
            result = await env.step(action)
            obs = result.observation
            obs_dict = obs_to_dict(obs)

            reward = result.reward or 0.0
            done = result.done
            error = obs_dict.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            # Format action string for logging
            action_str = action_type
            if action_dict.get("priority_label"):
                action_str += f"({action_dict['priority_label']})"
            if action_dict.get("department"):
                action_str += f"({action_dict['department']})"
            if action_dict.get("response_type"):
                action_str += f"({action_dict['response_type']})"

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} → reward={reward:.2f}")

            if done:
                # The final reward on done is the graded score
                score = max(0.0, min(1.0, reward))
                break

        if not result.done:
            # If we exhausted steps without finishing, use the accumulated score
            score = max(0.0, min(1.0, sum(rewards) / max(1, len(rewards))))

        score = max(0.01, min(0.99, score))  # clamp to valid range
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        score = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    """Run all tasks sequentially."""
    if not API_KEY:
        print("[ERROR] HF_TOKEN environment variable is not set. Cannot proceed.", flush=True)
        # Still emit [START]/[END] for each task so the validator gets valid output
        for task_name in TASKS:
            log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.01, rewards=[])
        return

    client_openai = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = {}

    for task_name in TASKS:
        env = None
        try:
            if IMAGE_NAME:
                env = await EmailTriageEnv.from_docker_image(IMAGE_NAME)
            else:
                # Default: connect to running server
                base_url = os.getenv("EMAIL_TRIAGE_BASE_URL", "http://localhost:8000")
                env = EmailTriageEnv(base_url=base_url)

            score = await run_task(client_openai, env, task_name)
            all_scores[task_name] = score
        except Exception as e:
            print(f"[ERROR] Task {task_name} failed with: {e}", flush=True)
            # Emit valid [START]/[END] logs so the validator doesn't break
            if task_name not in all_scores:
                log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
                log_end(success=False, steps=0, score=0.01, rewards=[])
            all_scores[task_name] = 0.01
        finally:
            if env is not None:
                try:
                    await env.close()
                except Exception as e:
                    print(f"[DEBUG] env.close() error: {e}", flush=True)

    print(f"\n[SUMMARY] Scores: {json.dumps(all_scores, indent=2)}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}", flush=True)
        import sys
        sys.exit(1)
