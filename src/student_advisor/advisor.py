"""
advisor.py
The core AI layer.

GPT-4o receives a student's raw academic profile and returns:
  - risk_level       : Low | Medium | High
  - risk_summary     : plain-English explanation of why
  - interventions    : list of specific advisor actions
  - urgency          : Immediate | Within 1 week | Monitor

No rules. No ML model. GPT-4o reasons over the data directly.
"""

import json
from functools import lru_cache

import pandas as pd
from openai import OpenAI

from student_advisor.config import DATA_FILE, GPT_MODEL, OPENAI_API_KEY


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic advisor at a university.
You will receive a student's academic profile and assess their risk of
academic failure or dropout.

Return ONLY a valid JSON object — no markdown, no preamble — with this structure:
{
  "risk_level": "Low | Medium | High",
  "risk_summary": "2-3 sentence plain English summary of the student's situation",
  "key_concerns": ["list of specific concerns identified from the data"],
  "interventions": [
    {
      "action": "Short title of the intervention",
      "detail": "Specific, actionable description for the advisor",
      "priority": "High | Medium | Low"
    }
  ],
  "urgency": "Immediate | Within 1 week | Within 2 weeks | Monitor",
  "positive_signals": ["any positive academic signals worth noting"]
}

Be specific and evidence-based. Reference the actual numbers from the profile.
If the student is doing well, say so clearly. Do not over-flag students."""


# ── Helpers ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)


def load_students() -> pd.DataFrame:
    """Load the student dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_FILE}. "
            "Run: python -m student_advisor.data_generator"
        )
    return pd.read_csv(DATA_FILE)


def get_student(student_id: str) -> pd.Series:
    """Return a single student row by ID."""
    df = load_students()
    row = df[df["student_id"] == student_id]
    if row.empty:
        raise ValueError(f"Student '{student_id}' not found.")
    return row.iloc[0]


def _build_prompt(student: pd.Series) -> str:
    """Format a student record as a natural language profile for GPT-4o."""
    fin = "Yes — financial hold on account" if student["financial_hold"] else "No"

    return f"""Student Academic Profile
========================
Name            : {student['name']}
Student ID      : {student['student_id']}
Department      : {student['department']}
Credits (term)  : {student['credits_this_term']}

Academic Indicators
-------------------
GPA                    : {student['gpa']} / 4.0
Attendance             : {student['attendance_pct']}%
Assignment Completion  : {student['assignments_completed']}%
LMS Logins / Week      : {student['lms_logins_per_week']}
Financial Hold         : {fin}

Please assess this student's academic risk and recommend advisor actions."""


# ── Main advisor function ──────────────────────────────────────────────────────

def advise(student_id: str) -> dict:
    """
    Ask GPT-4o to assess a student's risk and recommend interventions.

    Parameters
    ----------
    student_id : str

    Returns
    -------
    dict with keys: student_id, name, department, risk_level,
                    risk_summary, key_concerns, interventions,
                    urgency, positive_signals
    """
    student = get_student(student_id)
    prompt  = _build_prompt(student)

    response = _client().chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    # Attach student metadata
    result["student_id"]  = student_id
    result["name"]        = student["name"]
    result["department"]  = student["department"]
    result["tokens_used"] = response.usage.total_tokens

    return result
