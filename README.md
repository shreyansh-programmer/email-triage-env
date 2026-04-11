# FounderForge CEO Simulator

Welcome to **FounderForge**, a dynamic Tool-Calling Startup Simulator compliant with the OpenEnv Hackathon limits.

## Overview
FounderForge challenges the user (or LLM Agent) to manage a startup's finances and product strategy. You must properly allocate initial capital, strategically hire engineers and sales representatives, market the product effectively, and consistently reach funding milestones. 

## Features
- **3 Dynamic Tasks**: `bootstrap_survival`, `growth_stage`, `unicorn_ipo`.
- **Advanced Tool-Calling Architecture**: The environment natively exposes `hire_personnel`, `launch_marketing_campaign`, and `attempt_fundraise` in a structured JSON schema, matching high-tier validations like `calendar_env`.
- **Validation Ready**: Contains fully automated CI/CD and pre-submission checklist verification tools.

To run:
```bash
./validate-submission.sh
```
