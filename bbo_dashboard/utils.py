"""
utils.py
========
All the "data" logic for the dashboard lives here, separate from app.py.

Why split it up like this?
  - app.py stays short and focuses on *what to show* (layout, charts, text).
  - utils.py focuses on *getting the data ready* (reading Excel, cleaning,
    reshaping).
This is a common pattern once a Streamlit app grows past a single page —
it keeps things easy to read and easy to debug.

Every function that touches the Excel file is wrapped with @st.cache_data.
That decorator tells Streamlit: "run this once, remember the result, and
re-use it on every rerun unless the underlying file changes." Streamlit
reruns your whole script top-to-bottom every time a user clicks something,
so caching the slow part (reading Excel) keeps the app snappy.
"""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "weekly_optimisation_summary.xlsx"

WEEK_COLS = [f"Week {i}" for i in range(1, 14)]  # "Week 1" ... "Week 13"


# ---------------------------------------------------------------------------
# Loading the two sheets
# ---------------------------------------------------------------------------

@st.cache_data
def load_weeksummary() -> pd.DataFrame:
    """
    Load the 'weeksummary' sheet: one row per (Week, Function) submission.

    The real header is on row 2 of the sheet (row 1 is a merged title), so
    we skip the first row when reading.
    """
    df = pd.read_excel(DATA_PATH, sheet_name="weeksummary", skiprows=1)
    df["Week"] = df["Week"].astype(int)
    df["Function"] = df["Function"].astype(int)
    return df


@st.cache_data
def load_results() -> pd.DataFrame:
    """
    Load the 'result' sheet: one row per Function, with its actual output
    for every week plus initial/best/leaderboard summary columns.
    """
    df = pd.read_excel(DATA_PATH, sheet_name="result", skiprows=1)
    df["Function"] = df["Function"].astype(int)
    return df


@st.cache_data
def load_results_long() -> pd.DataFrame:
    """
    Reshape the 'result' sheet from "wide" (one column per week) to "long"
    (one row per Function+Week). Long format is what plotting libraries
    like Plotly want for line charts with a legend per function.

    Week 1 is 'n/a' for every function (week 1 was the initial sampling
    phase, before any optimisation submission), so those rows are dropped.
    """
    wide = load_results()
    long_df = wide.melt(
        id_vars=["Function", "Function Description", "Archetype", "Dimension",
                  "Initial Sample", "Final Sample", "Initial Best", "BBO Best",
                  "Leaderboard"],
        value_vars=WEEK_COLS,
        var_name="Week",
        value_name="Actual Output",
    )
    # "Week 7" -> 7 (integer), and drop the 'n/a' Week-1 placeholder rows
    long_df["Week"] = long_df["Week"].str.replace("Week ", "", regex=False).astype(int)
    long_df = long_df[pd.to_numeric(long_df["Actual Output"], errors="coerce").notna()]
    long_df["Actual Output"] = long_df["Actual Output"].astype(float)
    return long_df.sort_values(["Function", "Week"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Small derived helpers
# ---------------------------------------------------------------------------

def function_options(results: pd.DataFrame) -> dict:
    """Build a {label: function_number} dict for a selectbox, e.g.
    {'Function 1 — Radiation field': 1, ...}"""
    return {
        f"Function {row['Function']} — {row['Function Description']}": row["Function"]
        for _, row in results.iterrows()
    }


def normalise_progress(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max scale each function's weekly output to a 0-1 range.

    The 8 functions live on wildly different scales (one hovers near 1e-16,
    another climbs into the thousands), so a raw overlay chart is useless.
    Scaling each function's own trajectory to [0, 1] lets you compare
    *shapes* of progress (fast improvers vs. slow, steady vs. jumpy) on one
    chart, regardless of the underlying units.
    """
    out = long_df.copy()
    out["Normalised Output"] = out.groupby("Function")["Actual Output"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.5
    )
    return out


GENERATOR_FAMILIES = [
    # (test function, family label) — first match wins, so order matters
    (lambda s: "bake-off" in s, "Model Bake-off"),
    (lambda s: "trust-region" in s, "Trust Region + EI"),
    (lambda s: "loocv" in s and "best model" in s, "Model Selection (LOOCV)"),
    (lambda s: "local surrogate" in s or "local bootstrap-gp" in s, "Local Surrogate"),
    (lambda s: "mc-dropout" in s or "dropout nn" in s or _word(s, "mlp"), "NN (MC-Dropout / MLP)"),
    (lambda s: _word(s, "gp") and _word(s, "rf"), "GP + RF Hybrid"),
    (lambda s: _word(s, "gp") and _word(s, "svr"), "GP + SVR Blend"),
    (lambda s: _word(s, "gp") and _word(s, "svm"), "GP + SVM"),
    (lambda s: _word(s, "gp") and "polynomial" in s, "GP + Polynomial"),
    (lambda s: _word(s, "gp") or "gaussian" in s, "Gaussian Process"),
    (lambda s: "extratrees" in s, "ExtraTrees"),
    (lambda s: _word(s, "rf") or "random forest" in s, "Random Forest"),
    (lambda s: "tree ensemble" in s or "bootstrap tree" in s, "Tree Ensemble"),
    (lambda s: _word(s, "svr"), "SVR"),
]


def _word(s: str, w: str) -> bool:
    """True if `w` appears in `s` as a whole word (avoids 'rf' matching
    inside 'overfitting', 'gp' matching inside random substrings, etc.)."""
    return re.search(rf"\b{re.escape(w)}\b", s) is not None


def classify_generator(text: str) -> str:
    """
    Map a free-text 'Candidate Generator / Surrogate Model' note (e.g.
    'GP (Matern ARD) - UCB') to a short model-family label (e.g.
    'Gaussian Process') so we can chart how often each family was used.
    """
    if not isinstance(text, str) or not text.strip():
        return "Not recorded"
    s = text.lower()
    for test, label in GENERATOR_FAMILIES:
        if test(s):
            return label
    return "Other"


@st.cache_data
def weeksummary_with_family(weeksummary: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Model Family' column derived from the generator free-text."""
    df = weeksummary.copy()
    df["Model Family"] = df["Candidate Generator / Surrogate Model"].apply(classify_generator)
    return df
