"""
config.py — central settings for the Student AI Advisor.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]
DATA_FILE  = ROOT_DIR / "data" / "students.csv"
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL      = "gpt-4o"

# ── Dataset ────────────────────────────────────────────────────────────────────
NUM_STUDENTS = 500
RANDOM_SEED  = 42

DEPARTMENTS = [
    "Computer Science",
    "Business Administration",
    "Nursing",
    "Engineering",
    "Psychology",
    "Education",
    "Social Work",
    "Arts & Humanities",
]
