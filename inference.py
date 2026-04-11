"""
Inference Script for FounderForge CEO Simulator
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script emits exactly three line types to stdout:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

# Add the local directory to sys.path so we can import founderforge_env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from founderforge_env.founderforge_env.server.environment import FounderForgeEnvironment
from founderforge_env.founderforge_env.models import FounderForgeAction
from founderforge_env.founderforge_env.evaluation import GRADERS

# ── Mandatory Environment Variables ──────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

BENCHMARK = "founderforge_env"
TASKS = ["bootstrap_survival", "growth_stage", "unicorn_ipo"]

SYSTEM_PROMPT = textwrap.dedent("""
You are the CEO of a startup in FounderForge, a business simulation.
Your goal is to maximize user growth and reach funding milestones without going bankrupt.

RULES:
- Each turn = 1 month. Every month burns cash (base $10k + salaries + marketing).
- Engineers cost $12k/mo and boost product quality (+0.5 each).
- Sales reps cost $8k/mo and improve direct-sales channels.
- You MUST call exactly ONE of the provided tools each month.
- Read the [MARKET UPDATE] carefully — it may demand you pivot strategy or adjust hiring.

STRATEGY OPTIONS:
- 'product_led': Default. Standard operations.
- 'sales_led': Marketing ROI boosted 1.5x.
- 'survival_mode': Halves burn rate but loses 5% users/month.

FUNDING ROUNDS (require user thresholds):
- Pre-Seed: 500 users → $250k
- Seed: 5,000 users → $1M
- Series A: 50,000 users → $5M
- Series B: 200,000 users → $20M
- IPO: 1,000,000 users → $100M

Think step-by-step. Always choose the most strategically sound action.
""").strip()


# ── Logging Helpers (exact format required by OpenEnv validator) ─────────

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


# ── Agent Logic ──────────────────────────────────────────────────────────

def build_user_prompt(obs, step: int) -> str:
    """Build the structured situation report the LLM sees each month."""
    return textwrap.dedent(f"""
Month {step} Status Report:
  Cash: ${obs.cash:,.0f}
  Users: {obs.users:,.0f}
  Product Quality: {obs.product_quality:.1f}
  Team: Engineers={obs.team.get('engineers', 0)}, Sales={obs.team.get('sales', 0)}
  Current Round: {obs.current_round}
  Strategy: {obs.strategy}

{obs.last_action_result}

Previous Tool Result: {obs.tool_result or 'N/A'}

Select your next action using one of the available tools.
    """).strip()


def get_action_via_tools(client: OpenAI, obs, step: int, history: List):
    """Call the LLM with tool definitions and return the parsed action."""
    user_prompt = build_user_prompt(obs, step)

    openai_tools = [{"type": "function", "function": t} for t in obs.tools_list]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ] + history + [
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=openai_tools,
            tool_choice="required",
            temperature=0.3,
            max_tokens=256,
        )
        msg = completion.choices[0].message

        if msg.tool_calls:
            tc = msg.tool_calls[0]
            action = FounderForgeAction(
                action_type="ToolCallAction",
                tool_name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            return action, msg, None
        else:
            return FounderForgeAction(action_type="skip"), msg, None

    except Exception as exc:
        return FounderForgeAction(action_type="skip"), None, str(exc)


# ── Task Runner ──────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str, env: FounderForgeEnvironment) -> None:
    """Run a single task end-to-end and emit structured logs."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task_name=task_name)
    history: List = []
    rewards: List[float] = []
    step = 0
    score = 0.0
    success = False

    try:
        while step < 50:  # safety ceiling
            if obs.done:
                break

            step += 1
            action, msg, error_msg = get_action_via_tools(client, obs, step, history)

            # Format action string for log
            if action.tool_name:
                action_str = f"{action.tool_name}({json.dumps(action.arguments)})"
            else:
                action_str = action.action_type

            obs = env.step(action)
            reward = obs.reward or 0.0
            rewards.append(reward)

            log_step(step=step, action=action_str, reward=reward, done=obs.done, error=error_msg)

            # Maintain conversation history for tool-calling flow
            if msg and getattr(msg, "tool_calls", None):
                history.append(msg)
                history.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_calls[0].id,
                    "name": action.tool_name,
                    "content": str(obs.tool_result or obs.last_action_result),
                })

            if obs.done:
                # Use the programmatic grader for final score
                grader = GRADERS.get(task_name)
                if grader:
                    final_obs_dict = {
                        "users": obs.users,
                        "cash": obs.cash,
                        "current_round": obs.current_round,
                        "team": obs.team,
                        "product_quality": obs.product_quality,
                    }
                    score = grader(final_obs_dict)
                else:
                    score = max(0.01, min(0.99, reward))
                success = score >= 0.20
                break

    except Exception as exc:
        # Ensure [END] is always emitted even on crash
        pass

    finally:
        log_end(success=success, steps=step, score=score, rewards=rewards)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = FounderForgeEnvironment()

    try:
        for task in TASKS:
            run_task(client, task, env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
