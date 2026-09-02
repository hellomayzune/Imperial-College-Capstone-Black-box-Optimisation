"""
app.py — Black-Box Optimisation Dashboard
==========================================

HOW STREAMLIT WORKS (read this if you're new to it)
-----------------------------------------------------
Streamlit turns a plain Python script into a web app. There's no HTML or
JavaScript to write:
  - Every `st.something(...)` call draws one piece of UI, top to bottom,
    in the order you call it.
  - The ENTIRE script re-runs from top to bottom every time the user
    interacts with a widget (clicks a button, moves a slider, picks a
    dropdown option). That's why we cache slow steps (see utils.py) and
    why the code below reads like "describe the page", not "handle events".
  - Run it from a terminal with:  streamlit run app.py
    Streamlit opens a browser tab and gives you a local URL.

This file only handles LAYOUT. All data loading/reshaping lives in
utils.py — open that file too if you want to see how the Excel data is
turned into tables the charts can use.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

# ---------------------------------------------------------------------------
# Page setup — must be the first Streamlit call in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Finding a Mountain Peak in Zero Visibility",
    page_icon="🎯",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data once (cached — see utils.py). Every page below just reuses these.
# ---------------------------------------------------------------------------
weeksummary = utils.load_weeksummary()
weeksummary = utils.weeksummary_with_family(weeksummary)
results = utils.load_results()
results_long = utils.load_results_long()
func_options = utils.function_options(results)

N_FUNCTIONS = results["Function"].nunique()
N_WEEKS = weeksummary["Week"].nunique()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🎯 May Zune | Imperial-College-Capstone-Black-box-Optimisation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "📈 Function Progress",
        "🔬 Compare Functions",
        "🏆 Leaderboard",
        "🧪 Candidate Generators",
        "📋 Submission Log",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Data: `weekly_optimisation_summary.xlsx`\n\n"
    f"{N_FUNCTIONS} functions × up to {N_WEEKS} weeks."
)

# ===========================================================================
# PAGE 1 — Overview
# ===========================================================================
if page == "🏠 Overview":
    st.title("Weekly Black-Box Optimisation — Overview")
    st.write(
        "Across 13 weeks, candidate query points were submitted for **8 black-box functions** " 
        "based on **surrogate** model predictions. Each week, one point was evaluated per function, " 
        "and the corresponding true function outputs were recorded."
    )

    # --- headline metrics -------------------------------------------------
    total_submissions = weeksummary["Submitted Query (x)"].notna().sum()
    improved = weeksummary["Actual Output Above Initial Best"].eq("YES").sum()
    best_rank_row = results.loc[results["Leaderboard"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Functions", N_FUNCTIONS)
    c2.metric("Weeks tracked", N_WEEKS)
    c3.metric("Submissions logged", int(total_submissions))
    c4.metric(
        "Best leaderboard position",
        f"Function {int(best_rank_row.Function)}",
        help=best_rank_row["Function Description"],
    )

    st.markdown("---")

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Initial best vs. final BBO best, by function")
        st.caption(
            "Shows the distance each function progressed from its initial best value "
            "to the highest value discovered during the challenge. Note that bar heights "
            "cannot be compared across functions due to scaling differences—refer to the caption below. "
        )
        cmp_df = results[["Function", "Function Description", "Initial Best", "BBO Best"]].copy()
        cmp_df["Function Label"] = "F" + cmp_df["Function"].astype(str)
        fig = go.Figure()
        fig.add_bar(name="Initial Best", x=cmp_df["Function Label"], y=cmp_df["Initial Best"])
        fig.add_bar(name="BBO Best", x=cmp_df["Function Label"], y=cmp_df["BBO Best"])
        fig.update_layout(barmode="group", yaxis_title="Output value", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "⚠️ Because each function operates on a different output scale (ranging from around "
            "1e-16 to thousands), use this chart only to verify "
            "the *direction* of movement for individual functions. To compare "
            "functions directly, see the **Compare Functions** section."
        )

    with right:
        st.subheader("Submissions that outperformed the initial best value")
        counts = weeksummary["Actual Output Above Initial Best"].value_counts()
        fig2 = px.pie(
            names=counts.index, values=counts.values,
            color=counts.index,
            color_discrete_map={"YES": "#2ca02c", "NO": "#d62728"},
            hole=0.5,
        )
        fig2.update_layout(height=420, showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All functions — summary table")
    st.dataframe(
        results[[
            "Function", "Function Description", "Archetype", "Dimension",
            "Initial Sample", "Final Sample", "Initial Best", "BBO Best", "Leaderboard",
        ]].set_index("Function"),
        use_container_width=True,
    )

# ===========================================================================
# PAGE 2 — Function Progress
# ===========================================================================
elif page == "📈 Function Progress":
    st.title("Function Progress")

    label = st.selectbox("Choose a function", list(func_options.keys()))
    fn = func_options[label]
    row = results[results["Function"] == fn].iloc[0]
    fn_progress = results_long[results_long["Function"] == fn].sort_values("Week")
    fn_log = weeksummary[weeksummary["Function"] == fn].sort_values("Week")

    st.markdown(f"**Archetype:** {row.Archetype}  |  **Dimension:** {row.Dimension}D  "
                f"|  **Initial sample size:** {row['Initial Sample']}  "
                f"|  **Final sample size:** {row['Final Sample']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Initial best", f"{row['Initial Best']:.4g}")
    c2.metric("BBO best (final)", f"{row['BBO Best']:.4g}",
              delta=f"{row['BBO Best'] - row['Initial Best']:.4g}")
    c3.metric("Leaderboard position", int(row["Leaderboard"]))

    st.subheader("Actual output by week")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fn_progress["Week"], y=fn_progress["Actual Output"],
        mode="lines+markers", name="Actual output",
    ))
    fig.add_hline(
        y=row["Initial Best"], line_dash="dash", line_color="gray",
        annotation_text="Initial best", annotation_position="bottom right",
    )
    fig.update_layout(xaxis_title="Week", yaxis_title="Actual output", height=420,
                       xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekly submission log for this function")
    st.caption(
        "The candidate query point, model prediction, and actual result "
        "logged for each week, plus notes from the source notebooks."
    )
    show_cols = [
        "Week", "Submitted Query (x)", "Model's Predicted Value / Acquisition Score",
        "Returned Actual Output", "Actual Output Above Initial Best",
        "Candidate Generator / Surrogate Model", "Notes",
    ]
    st.dataframe(fn_log[show_cols].set_index("Week"), use_container_width=True)

# ===========================================================================
# PAGE 3 — Compare Functions
# ===========================================================================
elif page == "🔬 Compare Functions":
    st.title("Compare Functions")
    st.write(
        "Shows weekly progress overlaid across all eight functions. Since their output scales "
        "differ significantly, choose **Normalised (0–1)** to "
        "fairly compare their **performance trajectories** over time."
    )

    chosen_labels = st.multiselect(
        "Functions to show", list(func_options.keys()),
        default=list(func_options.keys()),
    )
    chosen = [func_options[l] for l in chosen_labels]
    scale_mode = st.radio("Scale", ["Normalised (0–1)", "Raw values"], horizontal=True)

    plot_df = results_long[results_long["Function"].isin(chosen)].copy()
    plot_df["Function Label"] = "F" + plot_df["Function"].astype(str)

    if scale_mode == "Normalised (0–1)":
        plot_df = utils.normalise_progress(plot_df)
        y_col, y_title = "Normalised Output", "Normalised output (0 = worst week, 1 = best week)"
    else:
        y_col, y_title = "Actual Output", "Actual output (raw scale)"

    if plot_df.empty:
        st.info("Pick at least one function above.")
    else:
        fig = px.line(
            plot_df, x="Week", y=y_col, color="Function Label",
            markers=True, hover_data=["Function Description"],
        )
        fig.update_layout(xaxis_title="Week", yaxis_title=y_title, height=520,
                           xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# PAGE 4 — Leaderboard
# ===========================================================================
elif page == "🏆 Leaderboard":
    st.title("Leaderboard")
    st.caption(
        "Displays the leaderboard rank for each function as recorded in the source spreadsheet. "
        "Please review your challenge's rules to verify whether a higher or lower score"
        "determines first place—this chart simply orders values in ascending order. "
    )

    lb = results[["Function", "Function Description", "Leaderboard"]].sort_values("Leaderboard")
    lb["Function Label"] = "F" + lb["Function"].astype(str) + " — " + lb["Function Description"]

    fig = px.bar(
        lb.sort_values("Leaderboard", ascending=False),
        x="Leaderboard", y="Function Label", orientation="h",
        color="Leaderboard", color_continuous_scale="Blues_r",
    )
    fig.update_layout(height=450, yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(lb[["Function", "Function Description", "Leaderboard"]].set_index("Function"),
                 use_container_width=True)

# ===========================================================================
# PAGE 5 — Candidate Generators
# ===========================================================================
elif page == "🧪 Candidate Generators":
    st.title("Candidate Generators / Surrogate Models")
    st.write(
        "Groups each week's candidate generator note (e.g., GP (Matern ARD) - UCB) "
        "into broader model families. This lets you track which modeling approaches were "
        "used most frequently, either overall or for a specific function. "
    )

    fn_filter = st.multiselect(
        "Filter by function (optional)", list(func_options.keys()),
    )
    df = weeksummary.copy()
    if fn_filter:
        chosen = [func_options[l] for l in fn_filter]
        df = df[df["Function"].isin(chosen)]

    fam_counts = df["Model Family"].value_counts().reset_index()
    fam_counts.columns = ["Model Family", "Count"]

    c1, c2 = st.columns([1, 1.3])
    with c1:
        fig = px.bar(fam_counts.sort_values("Count"), x="Count", y="Model Family", orientation="h")
        fig.update_layout(height=440, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Family usage by week")
        heat = (
            df.dropna(subset=["Candidate Generator / Surrogate Model"])
              .groupby(["Week", "Model Family"]).size().reset_index(name="Count")
        )
        fig2 = px.density_heatmap(
            heat, x="Week", y="Model Family", z="Count",
            color_continuous_scale="Blues", nbinsx=13,
        )
        fig2.update_layout(height=440, xaxis=dict(dtick=1))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("See the raw generator notes behind each family"):
        st.dataframe(
            df[["Week", "Function", "Candidate Generator / Surrogate Model", "Model Family"]]
            .sort_values(["Week", "Function"]).reset_index(drop=True),
            use_container_width=True,
        )

# ===========================================================================
# PAGE 6 — Submission Log (full detail, filterable)
# ===========================================================================
elif page == "📋 Submission Log":
    st.title("Full Submission Log")
    st.write("Every logged (Week, Function) submission. Filter and search below.")

    c1, c2, c3 = st.columns(3)
    with c1:
        week_range = st.slider("Week range", 1, N_WEEKS, (1, N_WEEKS))
    with c2:
        fn_filter = st.multiselect("Function(s)", list(func_options.keys()))
    with c3:
        family_filter = st.multiselect(
            "Model family", sorted(weeksummary["Model Family"].unique())
        )
    search = st.text_input("Search notes (optional)", "")

    df = weeksummary[
        weeksummary["Week"].between(week_range[0], week_range[1])
    ]
    if fn_filter:
        chosen = [func_options[l] for l in fn_filter]
        df = df[df["Function"].isin(chosen)]
    if family_filter:
        df = df[df["Model Family"].isin(family_filter)]
    if search:
        df = df[df["Notes"].fillna("").str.contains(search, case=False)]

    st.caption(f"{len(df)} of {len(weeksummary)} rows shown")
    st.dataframe(
        df.drop(columns=["Function Description"]).sort_values(["Week", "Function"]),
        use_container_width=True, height=520,
    )

    st.download_button(
        "⬇️ Download filtered rows as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="bbo_submission_log_filtered.csv",
        mime="text/csv",
    )
