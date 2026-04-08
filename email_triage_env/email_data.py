"""
Email Corpus Generator for the Email Triage Environment.

Generates realistic email datasets with ground-truth labels for:
- Priority classification (HIGH / MEDIUM / LOW)
- Department routing (Engineering, Sales, Legal, HR, Support, Executive)
- Response type (acknowledge, escalate, delegate, decline, info_request)

All data is deterministic given a seed, ensuring reproducible evaluations.
"""

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Email:
    """A single email with ground-truth labels."""

    id: str
    sender_name: str
    sender_title: str
    sender_email: str
    subject: str
    body: str
    timestamp: str
    thread_id: Optional[str] = None

    # Ground truth labels
    gt_priority: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    gt_department: str = "Support"
    gt_response_type: str = "acknowledge"

    # Difficulty signals
    has_urgency_keywords: bool = False
    has_vip_sender: bool = False
    has_deadline: bool = False
    is_thread: bool = False


# ─── Sender pools ───────────────────────────────────────────────

VIP_SENDERS = [
    ("Sarah Chen", "CEO", "sarah.chen@company.com"),
    ("Michael Torres", "VP Engineering", "m.torres@company.com"),
    ("Priya Sharma", "CFO", "p.sharma@company.com"),
    ("David Kim", "CTO", "d.kim@company.com"),
    ("Lisa Wang", "VP Sales", "l.wang@company.com"),
]

REGULAR_SENDERS = [
    ("James Wilson", "Senior Engineer", "j.wilson@company.com"),
    ("Emily Brown", "Product Manager", "e.brown@company.com"),
    ("Alex Johnson", "Sales Rep", "a.johnson@company.com"),
    ("Maria Garcia", "HR Coordinator", "m.garcia@company.com"),
    ("Robert Lee", "Support Lead", "r.lee@company.com"),
    ("Jennifer Davis", "Marketing Analyst", "j.davis@company.com"),
    ("Chris Martinez", "Legal Counsel", "c.martinez@company.com"),
    ("Amanda Foster", "DevOps Engineer", "a.foster@company.com"),
    ("Tom Nakamura", "Account Executive", "t.nakamura@company.com"),
    ("Rachel Green", "UX Designer", "r.green@company.com"),
]

EXTERNAL_SENDERS = [
    ("John Smith", "Customer", "john.smith@acmecorp.com"),
    ("Karen White", "Vendor Partner", "k.white@supplierx.com"),
    ("Paul Reeves", "Auditor", "p.reeves@auditing.com"),
    ("Nina Patel", "Applicant", "nina.patel@gmail.com"),
    ("Derek Chang", "Regulatory Advisor", "d.chang@regboard.gov"),
]

# ─── Email templates by (priority, department) ──────────────────

EMAIL_TEMPLATES: Dict[str, List[Dict]] = {
    # ── HIGH priority ──
    "HIGH_Engineering": [
        {
            "subject": "URGENT: Production database failing – immediate action required",
            "body": (
                "Team,\n\nOur production database cluster is showing critical failures. "
                "Error rates spiked to 15% in the last 30 minutes and customer-facing APIs "
                "are returning 500 errors. The on-call team has been paged but we need "
                "senior engineering support immediately.\n\nPlease join the incident bridge "
                "call ASAP. Rollback plan needs to be reviewed.\n\nThis is P0."
            ),
            "response_type": "escalate",
        },
        {
            "subject": "CRITICAL: Security vulnerability discovered in auth service",
            "body": (
                "We've identified a critical SQL injection vulnerability in the authentication "
                "service. An attacker could potentially bypass login and access user data. "
                "We need to patch this within the next 2 hours before our next deployment window. "
                "All hands on deck.\n\nCVE details attached. Notify the security team immediately."
            ),
            "response_type": "escalate",
        },
    ],
    "HIGH_Sales": [
        {
            "subject": "Enterprise deal at risk – client threatening to walk",
            "body": (
                "Hi,\n\nOur largest enterprise prospect (AcmeCorp, $2.4M ARR deal) just called. "
                "They're threatening to go with our competitor unless we can match their pricing "
                "by end of day Friday. The VP of Procurement wants a call with our CEO today.\n\n"
                "This is our highest-value pipeline deal this quarter. Need executive decision ASAP."
            ),
            "response_type": "escalate",
        },
    ],
    "HIGH_Legal": [
        {
            "subject": "URGENT: Regulatory compliance deadline tomorrow",
            "body": (
                "Reminding everyone that our SOC 2 audit documentation is due tomorrow at 5 PM EST. "
                "We're still missing 3 critical control narratives from Engineering and HR. "
                "Non-compliance could result in loss of certification and major client contracts.\n\n"
                "Legal needs completed documents by noon tomorrow at the latest."
            ),
            "response_type": "escalate",
        },
    ],
    "HIGH_HR": [
        {
            "subject": "Employee harassment complaint – immediate investigation needed",
            "body": (
                "An employee has filed a formal harassment complaint against their direct manager. "
                "Per our policy, we need to initiate an investigation within 24 hours. "
                "The complainant is requesting immediate transfer to a different team.\n\n"
                "HR leadership needs to meet today to assign an investigator and ensure "
                "the complainant's safety. This is confidential and time-sensitive."
            ),
            "response_type": "escalate",
        },
    ],
    "HIGH_Support": [
        {
            "subject": "Major client outage – all services down for 45 minutes",
            "body": (
                "Our top-tier enterprise client (BankCo) is reporting a complete service outage. "
                "All their users are locked out. Their SLA guarantees 99.99% uptime and we're "
                "already in breach. The client's CTO is on the phone demanding a root cause "
                "analysis within the hour.\n\nEscalate to engineering and get status update immediately."
            ),
            "response_type": "escalate",
        },
    ],
    "HIGH_Executive": [
        {
            "subject": "Board meeting prep – materials needed by tonight",
            "body": (
                "The quarterly board meeting is tomorrow morning at 9 AM. We're still missing "
                "the financial forecast slides and the product roadmap update. Board members have "
                "been asking for the pre-read materials since yesterday.\n\n"
                "CEO needs these finalized and sent out by 8 PM tonight. No exceptions."
            ),
            "response_type": "acknowledge",
        },
    ],

    # ── MEDIUM priority ──
    "MEDIUM_Engineering": [
        {
            "subject": "Code review needed: Payment service refactor PR #4521",
            "body": (
                "Hi team,\n\nI've submitted PR #4521 for the payment service refactor. "
                "It addresses the technical debt items we discussed in last sprint's retro. "
                "Changes include migrating from synchronous to async processing and adding "
                "retry logic for failed transactions.\n\nCould someone review by end of week? "
                "Not blocking any release but it would be nice to get it merged before next sprint."
            ),
            "response_type": "acknowledge",
        },
        {
            "subject": "Staging environment performance degradation",
            "body": (
                "Noticed that our staging environment has been running 40% slower than usual "
                "this week. It's not affecting production but it's slowing down QA testing. "
                "Likely related to the new logging pipeline we deployed last Tuesday.\n\n"
                "Can someone from the platform team take a look when they have bandwidth?"
            ),
            "response_type": "delegate",
        },
    ],
    "MEDIUM_Sales": [
        {
            "subject": "Q3 pipeline review – need updated numbers by Thursday",
            "body": (
                "Team,\n\nLet's get the Q3 pipeline numbers updated before our Thursday "
                "review meeting. Please ensure all opportunities in stage 3+ have accurate "
                "close dates and deal values. Lisa will be presenting to the leadership team "
                "on Friday.\n\nDashboard link: [internal]\n\nThanks for keeping things current."
            ),
            "response_type": "acknowledge",
        },
    ],
    "MEDIUM_Legal": [
        {
            "subject": "Contract review: New vendor agreement for cloud services",
            "body": (
                "Please review the attached master service agreement from CloudVendor Inc. "
                "They've sent a revised version with updated data processing terms. Key changes "
                "are in Section 7 (data residency) and Section 12 (liability caps).\n\n"
                "Not urgent but we'd like to finalize by end of next week. Initial read suggests "
                "most terms are standard but the indemnification clause needs a closer look."
            ),
            "response_type": "acknowledge",
        },
    ],
    "MEDIUM_HR": [
        {
            "subject": "New benefits enrollment period starting next month",
            "body": (
                "Hi all,\n\nOpen enrollment for 2026 benefits starts May 1st and runs through "
                "May 15th. We're introducing two new plan options this year. HR will be hosting "
                "info sessions on April 25th and 28th.\n\nPlease share this with your teams and "
                "encourage everyone to review the benefits comparison guide on the intranet."
            ),
            "response_type": "delegate",
        },
        {
            "subject": "Performance review cycle: Manager training schedule",
            "body": (
                "The mid-year performance review cycle kicks off June 1st. All managers are "
                "required to complete the calibration training before submitting reviews. "
                "Sessions are available next week – please sign up on the HR portal.\n\n"
                "Reminder: self-assessments are due by May 28th."
            ),
            "response_type": "acknowledge",
        },
    ],
    "MEDIUM_Support": [
        {
            "subject": "Customer feedback trends: Increasing tickets about onboarding",
            "body": (
                "Hi,\n\nI've been analyzing this month's support tickets and there's a notable "
                "25% increase in onboarding-related issues. The most common complaints are about "
                "the initial setup wizard and API key generation flow.\n\n"
                "Suggest we flag this for the product team. Not an emergency but the trend is "
                "worth addressing before it impacts our NPS score."
            ),
            "response_type": "delegate",
        },
    ],
    "MEDIUM_Executive": [
        {
            "subject": "Monthly all-hands meeting agenda review",
            "body": (
                "Hi,\n\nPlease review the draft agenda for next Friday's all-hands. I've included "
                "the product launch announcement and the new hire introductions. Let me know if "
                "you'd like to add any items by Wednesday.\n\nAlso, CEO wants 5 minutes for a "
                "special recognition segment. Need to confirm the award recipients."
            ),
            "response_type": "acknowledge",
        },
    ],

    # ── LOW priority ──
    "LOW_Engineering": [
        {
            "subject": "FYI: Updated coding style guide published",
            "body": (
                "Hey everyone,\n\nThe updated coding style guide is now live on Confluence. "
                "Main changes: we're adopting Black formatter for Python and Prettier for JS. "
                "No immediate action needed – just be aware for new PRs going forward.\n\n"
                "The linting CI checks will be enforced starting next month."
            ),
            "response_type": "acknowledge",
        },
        {
            "subject": "Tech talk next Wednesday: Intro to WebAssembly",
            "body": (
                "We're hosting a tech talk next Wednesday at 3 PM in the main conference room. "
                "Rachel from the frontend team will be presenting on WebAssembly and its potential "
                "applications in our product.\n\nPizza will be provided. RSVP on the calendar invite."
            ),
            "response_type": "acknowledge",
        },
    ],
    "LOW_Sales": [
        {
            "subject": "Updated sales collateral available on shared drive",
            "body": (
                "Marketing has updated the product one-pagers and case studies for Q2. "
                "The new materials reflect the latest feature releases and pricing changes. "
                "Find them in the usual shared drive location.\n\n"
                "Let me know if you need any customized versions for specific prospects."
            ),
            "response_type": "acknowledge",
        },
    ],
    "LOW_Legal": [
        {
            "subject": "Annual compliance training reminder",
            "body": (
                "This is a friendly reminder to complete your annual compliance training "
                "modules. The deadline is end of month. Modules include anti-bribery, "
                "data privacy, and workplace safety.\n\nAccess them through the Learning "
                "Management System. Estimated time: 45 minutes total."
            ),
            "response_type": "acknowledge",
        },
    ],
    "LOW_HR": [
        {
            "subject": "Office social: Team building event next Friday",
            "body": (
                "We're organizing a team building event next Friday afternoon. Activities include "
                "an escape room challenge and a cooking class. Please sign up by Tuesday so we can "
                "finalize the headcount.\n\nSpouses and partners are welcome. Details on the intranet."
            ),
            "response_type": "acknowledge",
        },
    ],
    "LOW_Support": [
        {
            "subject": "Knowledge base article update suggestion",
            "body": (
                "While helping a customer today, I noticed our KB article on API rate limiting "
                "is outdated. It still references the old 1000 req/min limit when we've since "
                "updated to 5000 req/min for pro tier users.\n\n"
                "Low priority but would be nice to update when someone has time."
            ),
            "response_type": "acknowledge",
        },
    ],
    "LOW_Executive": [
        {
            "subject": "FYI: Industry newsletter – AI trends in enterprise SaaS",
            "body": (
                "Sharing this interesting newsletter from McKinsey about AI adoption trends in "
                "enterprise SaaS. Some relevant insights for our strategic planning. Key takeaway: "
                "80% of enterprise buyers are now evaluating AI-native solutions.\n\n"
                "Good background reading for the strategy offsite next month."
            ),
            "response_type": "acknowledge",
        },
    ],
}


def _deterministic_hash(seed: int, index: int) -> int:
    """Produce a deterministic hash from seed and index."""
    h = hashlib.md5(f"{seed}-{index}".encode()).hexdigest()
    return int(h, 16)


def generate_email_corpus(
    task_name: str,
    seed: int = 42,
) -> List[Email]:
    """Generate a deterministic email corpus appropriate for the given task.

    Args:
        task_name: One of 'priority_classification', 'route_and_classify', 'full_triage'
        seed: Random seed for reproducibility

    Returns:
        List of Email objects with ground-truth labels
    """
    rng = random.Random(seed)

    if task_name == "priority_classification":
        return _gen_priority_corpus(rng, count=10)
    elif task_name == "route_and_classify":
        return _gen_route_corpus(rng, count=15)
    elif task_name == "full_triage":
        return _gen_full_corpus(rng, count=20)
    else:
        return _gen_priority_corpus(rng, count=10)


def _pick_sender(rng: random.Random, priority: str):
    """Pick an appropriate sender based on priority."""
    if priority == "HIGH":
        # HIGH priority emails more likely from VIPs
        if rng.random() < 0.6:
            return rng.choice(VIP_SENDERS)
        else:
            return rng.choice(REGULAR_SENDERS)
    elif priority == "MEDIUM":
        if rng.random() < 0.2:
            return rng.choice(VIP_SENDERS)
        else:
            return rng.choice(REGULAR_SENDERS)
    else:
        # LOW: mix of all
        pool = REGULAR_SENDERS + EXTERNAL_SENDERS
        return rng.choice(pool)


def _gen_timestamps(rng: random.Random, count: int) -> List[str]:
    """Generate realistic timestamps spread over a workday."""
    hours = sorted(rng.sample(range(7, 19), min(count, 12)))
    while len(hours) < count:
        hours.append(rng.randint(7, 18))
    hours = sorted(hours[:count])
    return [
        f"2026-04-{rng.randint(1, 8):02d}T{h:02d}:{rng.randint(0, 59):02d}:00Z"
        for h in hours
    ]


def _gen_priority_corpus(rng: random.Random, count: int) -> List[Email]:
    """Generate corpus for easy task: priority classification only."""
    priorities = ["HIGH"] * 3 + ["MEDIUM"] * 4 + ["LOW"] * 3
    rng.shuffle(priorities)
    priorities = priorities[:count]

    departments = list(EMAIL_TEMPLATES.keys())
    timestamps = _gen_timestamps(rng, count)
    emails = []

    for i, priority in enumerate(priorities):
        # Pick a matching template
        matching_keys = [k for k in EMAIL_TEMPLATES if k.startswith(priority)]
        key = rng.choice(matching_keys)
        dept = key.split("_", 1)[1]
        template = rng.choice(EMAIL_TEMPLATES[key])

        sender = _pick_sender(rng, priority)
        email = Email(
            id=f"email_{i+1:03d}",
            sender_name=sender[0],
            sender_title=sender[1],
            sender_email=sender[2],
            subject=template["subject"],
            body=template["body"],
            timestamp=timestamps[i],
            gt_priority=priority,
            gt_department=dept,
            gt_response_type=template["response_type"],
            has_urgency_keywords=priority == "HIGH",
            has_vip_sender=sender in VIP_SENDERS,
            has_deadline="deadline" in template["body"].lower() or "by end of" in template["body"].lower(),
        )
        emails.append(email)

    return emails


def _gen_route_corpus(rng: random.Random, count: int) -> List[Email]:
    """Generate corpus for medium task: priority + routing."""
    priorities = ["HIGH"] * 4 + ["MEDIUM"] * 6 + ["LOW"] * 5
    rng.shuffle(priorities)
    priorities = priorities[:count]

    timestamps = _gen_timestamps(rng, count)
    emails = []

    for i, priority in enumerate(priorities):
        matching_keys = [k for k in EMAIL_TEMPLATES if k.startswith(priority)]
        key = rng.choice(matching_keys)
        dept = key.split("_", 1)[1]
        template = rng.choice(EMAIL_TEMPLATES[key])

        sender = _pick_sender(rng, priority)
        email = Email(
            id=f"email_{i+1:03d}",
            sender_name=sender[0],
            sender_title=sender[1],
            sender_email=sender[2],
            subject=template["subject"],
            body=template["body"],
            timestamp=timestamps[i],
            gt_priority=priority,
            gt_department=dept,
            gt_response_type=template["response_type"],
            has_urgency_keywords=priority == "HIGH",
            has_vip_sender=sender in VIP_SENDERS,
            has_deadline="deadline" in template["body"].lower() or "by end of" in template["body"].lower(),
        )
        emails.append(email)

    return emails


def _gen_full_corpus(rng: random.Random, count: int) -> List[Email]:
    """Generate corpus for hard task: priority + routing + responses, with threads."""
    priorities = ["HIGH"] * 6 + ["MEDIUM"] * 8 + ["LOW"] * 6
    rng.shuffle(priorities)
    priorities = priorities[:count]

    timestamps = _gen_timestamps(rng, count)
    emails = []

    # Create some threads (groups of related emails)
    thread_groups = {}
    for i in range(0, count, 4):
        tid = f"thread_{i//4 + 1}"
        thread_groups[i] = tid
        if i + 1 < count:
            thread_groups[i + 1] = tid

    for i, priority in enumerate(priorities):
        matching_keys = [k for k in EMAIL_TEMPLATES if k.startswith(priority)]
        key = rng.choice(matching_keys)
        dept = key.split("_", 1)[1]
        template = rng.choice(EMAIL_TEMPLATES[key])

        sender = _pick_sender(rng, priority)
        thread_id = thread_groups.get(i)

        email = Email(
            id=f"email_{i+1:03d}",
            sender_name=sender[0],
            sender_title=sender[1],
            sender_email=sender[2],
            subject=template["subject"],
            body=template["body"],
            timestamp=timestamps[i],
            thread_id=thread_id,
            gt_priority=priority,
            gt_department=dept,
            gt_response_type=template["response_type"],
            has_urgency_keywords=priority == "HIGH",
            has_vip_sender=sender in VIP_SENDERS,
            has_deadline="deadline" in template["body"].lower() or "by end of" in template["body"].lower(),
            is_thread=thread_id is not None,
        )
        emails.append(email)

    return emails
