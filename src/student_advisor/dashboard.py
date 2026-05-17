"""
dashboard.py — Streamlit dashboard for the Student AI Advisor.

Run:
    streamlit run src/student_advisor/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from student_advisor.advisor import advise, load_students
from student_advisor.config import OPENAI_API_KEY, GPT_MODEL

# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student AI Advisor",
    page_icon="🎓",
    layout="wide",
)

RISK_COLOUR = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c",
}

URGENCY_ICON = {
    "Immediate":      "🔴",
    "Within 1 week":  "🟡",
    "Within 2 weeks": "🟡",
    "Monitor":        "🟢",
}

PRIORITY_ICON = {
    "High":   "🔴",
    "Medium": "🟡",
    "Low":    "🟢",
}


# ── Data ───────────────────────────────────────────────────────────────────────

@st.cache_data
def get_students() -> pd.DataFrame:
    try:
        return load_students()
    except FileNotFoundError:
        return pd.DataFrame()


df = get_students()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎓 Student AI Advisor")
    st.caption(f"Powered by {GPT_MODEL}")
    st.divider()

    if OPENAI_API_KEY:
        st.success("OpenAI connected")
    else:
        st.error("OPENAI_API_KEY missing\nAdd it to your .env file")

    if not df.empty:
        dept_options = ["All"] + sorted(df["department"].unique().tolist())
        dept_filter  = st.selectbox("Filter by department", dept_options)
    else:
        dept_filter = "All"

    st.divider()
    st.caption("Synthetic data · Portfolio demo")


# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🎓 Student AI Advisor")
st.caption("GPT-4o analyses each student's academic profile and recommends advisor actions.")
st.divider()


# ── No data guard ──────────────────────────────────────────────────────────────

if df.empty:
    st.warning("No student data found. Run `python -m student_advisor.data_generator` first.")
    st.stop()


# ── Apply filter ───────────────────────────────────────────────────────────────

filtered_df = df if dept_filter == "All" else df[df["department"] == dept_filter]


# ── KPI row ────────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Students",  f"{len(filtered_df):,}")
c2.metric("Avg GPA",         f"{filtered_df['gpa'].mean():.2f}")
c3.metric("Avg Attendance",  f"{filtered_df['attendance_pct'].mean():.1f}%")
c4.metric("Financial Holds", f"{filtered_df['financial_hold'].sum():,}")

st.divider()


# ── Overview charts ────────────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("GPA Distribution")
    fig = px.histogram(
        filtered_df, x="gpa", nbins=30,
        color_discrete_sequence=["#4A90D9"],
        labels={"gpa": "GPA"},
    )
    fig.update_layout(height=260, margin=dict(t=0, b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Attendance Distribution")
    fig2 = px.histogram(
        filtered_df, x="attendance_pct", nbins=30,
        color_discrete_sequence=["#7B68EE"],
        labels={"attendance_pct": "Attendance %"},
    )
    fig2.update_layout(height=260, margin=dict(t=0, b=0), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()


# ── Student table ──────────────────────────────────────────────────────────────

st.subheader("📋 Student List")

display = filtered_df[[
    "student_id", "name", "department",
    "gpa", "attendance_pct", "assignments_completed",
    "lms_logins_per_week", "financial_hold",
]].rename(columns={
    "student_id":            "ID",
    "name":                  "Name",
    "department":            "Department",
    "gpa":                   "GPA",
    "attendance_pct":        "Attendance %",
    "assignments_completed": "Assignments %",
    "lms_logins_per_week":   "LMS Logins/wk",
    "financial_hold":        "Fin. Hold",
})

st.dataframe(display, use_container_width=True, height=260)

st.divider()


# ── AI Advisor panel ───────────────────────────────────────────────────────────

st.subheader("🤖 AI Advisor")
st.caption("Select a student and GPT-4o will analyse their profile and recommend actions.")

selected_id = st.selectbox(
    "Select student",
    filtered_df["student_id"].tolist(),
    format_func=lambda sid: (
        f"{sid} — "
        + filtered_df.loc[filtered_df['student_id'] == sid, 'name'].values[0]
        + f" ({filtered_df.loc[filtered_df['student_id'] == sid, 'department'].values[0]})"
    ),
)

# Show student raw data
if selected_id:
    student_row = filtered_df[filtered_df["student_id"] == selected_id].iloc[0]

    with st.expander("📊 Student profile data", expanded=False):
        profile_cols = st.columns(4)
        profile_cols[0].metric("GPA",          f"{student_row['gpa']:.2f}")
        profile_cols[1].metric("Attendance",   f"{student_row['attendance_pct']:.1f}%")
        profile_cols[2].metric("Assignments",  f"{student_row['assignments_completed']:.1f}%")
        profile_cols[3].metric("LMS Logins/wk", f"{student_row['lms_logins_per_week']:.0f}")

    analyse_btn = st.button("Ask GPT-4o to advise", type="primary", use_container_width=False)

    if analyse_btn:
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not set. Add it to your .env file.")
        else:
            with st.spinner(f"GPT-4o analysing {student_row['name']}..."):
                try:
                    result = advise(selected_id)

                    # ── Risk badge ─────────────────────────────────────────────
                    risk  = result["risk_level"]
                    color = RISK_COLOUR.get(risk, "#888")
                    urgency = result.get("urgency", "")
                    u_icon  = URGENCY_ICON.get(urgency, "⚪")

                    col_risk, col_urgency = st.columns([1, 2])
                    col_risk.markdown(
                        f"<div style='background:{color};padding:16px;border-radius:10px;"
                        f"text-align:center;color:white;font-size:22px;font-weight:600'>"
                        f"{risk} Risk</div>",
                        unsafe_allow_html=True,
                    )
                    col_urgency.markdown(
                        f"<div style='padding:16px;border-radius:10px;"
                        f"background:var(--secondary-background-color);font-size:16px'>"
                        f"<b>Urgency:</b> {u_icon} {urgency}</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("")

                    # ── GPT summary ────────────────────────────────────────────
                    st.info(result["risk_summary"])

                    col_l, col_r = st.columns(2)

                    with col_l:
                        # Key concerns
                        if result.get("key_concerns"):
                            st.markdown("**⚠️ Key concerns**")
                            for concern in result["key_concerns"]:
                                st.markdown(f"- {concern}")

                        # Positive signals
                        if result.get("positive_signals"):
                            st.markdown("**✅ Positive signals**")
                            for signal in result["positive_signals"]:
                                st.markdown(f"- {signal}")

                    with col_r:
                        # Interventions
                        st.markdown("**📋 Recommended interventions**")
                        for iv in result.get("interventions", []):
                            icon = PRIORITY_ICON.get(iv.get("priority", ""), "⚪")
                            with st.expander(
                                f"{icon} {iv['action']} · {iv['priority']} priority"
                            ):
                                st.write(iv["detail"])

                    # Footer
                    st.caption(
                        f"Model: {GPT_MODEL} · "
                        f"Tokens used: {result.get('tokens_used', 'N/A')}"
                    )

                except EnvironmentError as e:
                    st.error(str(e))
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
