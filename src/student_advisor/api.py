"""
api.py — FastAPI for the Student AI Advisor.

Endpoints
---------
GET /health
GET /students
GET /students/{student_id}/advise

Run:
    uvicorn student_advisor.api:app --reload
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from student_advisor.advisor import advise, load_students
from student_advisor.config import GPT_MODEL

app = FastAPI(
    title="Student AI Advisor",
    version="1.0.0",
    description=(
        "GPT-4o powered academic advisor. "
        "Analyses student data and returns risk assessment + interventions. "
        "Synthetic data — portfolio demo only."
    ),
)


# ── Response models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str

class Intervention(BaseModel):
    action: str
    detail: str
    priority: str

class AdviceResponse(BaseModel):
    student_id: str
    name: str
    department: str
    risk_level: str
    risk_summary: str
    key_concerns: list[str]
    interventions: list[Intervention]
    urgency: str
    positive_signals: list[str]
    tokens_used: Optional[int] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Check the API is running."""
    return HealthResponse(status="healthy", model=GPT_MODEL)


@app.get("/students", tags=["Data"])
def list_students(limit: int = Query(default=20, le=200)):
    """Return a list of student records."""
    try:
        df = load_students()
        records = (
            df[["student_id", "name", "department", "gpa", "attendance_pct"]]
            .head(limit)
            .to_dict(orient="records")
        )
        return {"total": len(df), "returned": len(records), "students": records}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/students/{student_id}/advise", response_model=AdviceResponse, tags=["AI Advisor"])
def advise_student(student_id: str):
    """
    Ask GPT-4o to analyse a student's academic profile and return
    a risk assessment with specific advisor interventions.
    """
    try:
        result = advise(student_id)
        return AdviceResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-4o error: {str(e)}")
