"""Tests for data generation."""

import pandas as pd
import numpy as np
import pytest

from student_advisor.data_generator import generate


class TestDataGenerator:

    def test_correct_row_count(self):
        df = generate(n=100)
        assert len(df) == 100

    def test_required_columns(self):
        df = generate(n=50)
        required = [
            "student_id", "name", "department", "gpa",
            "attendance_pct", "assignments_completed",
            "lms_logins_per_week", "financial_hold", "credits_this_term",
        ]
        for col in required:
            assert col in df.columns, f"Missing: {col}"

    def test_gpa_bounds(self):
        df = generate(n=200)
        assert df["gpa"].between(0.0, 4.0).all()

    def test_attendance_bounds(self):
        df = generate(n=200)
        assert df["attendance_pct"].between(0, 100).all()

    def test_assignments_bounds(self):
        df = generate(n=200)
        assert df["assignments_completed"].between(0, 100).all()

    def test_student_ids_unique(self):
        df = generate(n=100)
        assert df["student_id"].nunique() == 100

    def test_reproducible(self):
        df1 = generate(n=50, seed=1)
        df2 = generate(n=50, seed=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_financial_hold_is_boolean(self):
        df = generate(n=100)
        assert df["financial_hold"].dtype == bool

    def test_roughly_15_pct_financial_hold(self):
        df = generate(n=1000, seed=0)
        rate = df["financial_hold"].mean()
        assert 0.05 < rate < 0.30, f"Financial hold rate unexpected: {rate:.2f}"
