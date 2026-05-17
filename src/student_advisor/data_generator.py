"""
data_generator.py
Generates synthetic student records for the AI advisor demo.
All data is fictional.

Run:
    python -m student_advisor.data_generator
"""

import numpy as np
import pandas as pd

from student_advisor.config import DATA_FILE, DEPARTMENTS, NUM_STUDENTS, RANDOM_SEED


def generate(n: int = NUM_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate n synthetic student records.

    Fields
    ------
    student_id              Unique identifier (STU00001 …)
    name                    Fictional first + last name
    department              Academic department
    gpa                     0.0 – 4.0
    attendance_pct          0 – 100 %
    assignments_completed   0 – 100 %
    lms_logins_per_week     Average weekly LMS logins
    financial_hold          Boolean flag
    credits_this_term       3 – 18
    """
    rng = np.random.default_rng(seed)

    ids   = [f"STU{str(i).zfill(5)}" for i in range(1, n + 1)]
    names = [f"TestFirstName{i} TestLastName{i}" for i in range(1, n + 1)]

    gpa         = np.clip(rng.normal(2.8, 0.7, n), 0.0, 4.0).round(2)
    attendance  = np.clip(rng.normal(78, 15, n), 0, 100).round(1)
    assignments = np.clip(rng.normal(75, 18, n), 0, 100).round(1)
    lms_logins  = np.clip(rng.poisson(5, n), 0, 30).astype(float)
    fin_hold    = rng.choice([False, True], n, p=[0.85, 0.15])
    credits     = rng.integers(3, 19, n).astype(float)
    departments = rng.choice(DEPARTMENTS, n)

    df = pd.DataFrame({
        "student_id":            ids,
        "name":                  names,
        "department":            departments,
        "gpa":                   gpa,
        "attendance_pct":        attendance,
        "assignments_completed": assignments,
        "lms_logins_per_week":   lms_logins,
        "financial_hold":        fin_hold,
        "credits_this_term":     credits,
    })

    return df


def save(df: pd.DataFrame) -> None:
    df.to_csv(DATA_FILE, index=False)
    print(f"[data] Saved {len(df)} student records -> {DATA_FILE}")


if __name__ == "__main__":
    df = generate()
    save(df)
    print(df.head())
