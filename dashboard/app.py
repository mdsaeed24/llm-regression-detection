"""Streamlit dashboard over the SQLite run history.

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from regress.config import load_settings  # noqa: E402
from regress.dataset import load_dataset  # noqa: E402
from regress.eval import compare_runs  # noqa: E402
from regress.eval.results import RunRecord  # noqa: E402
from regress.models import CATEGORIES  # noqa: E402
from regress.storage import RunStore  # noqa: E402

# Validated categorical slots (light-mode steps), assigned in fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
METRICS = {
    "Accuracy": ("accuracy", ".1%"),
    "Pass rate": ("pass_rate", ".1%"),
    "Avg relevance (1-5)": ("avg_relevance", ".2f"),
    "Cost per run (USD)": ("cost_usd", "$.4f"),
    "Latency p95 (ms)": ("latency_p95_ms", ",.0f"),
}

st.set_page_config(page_title="Prompt regression dashboard", layout="wide")
if os.environ.get("REGRESS_DEMO"):  # set by streamlit_app.py (public Streamlit Cloud demo)
    st.info(
        "**Public demo:** a read-only snapshot of real evaluation runs of this prompt-regression system. "
        "No evaluation is run and no LLM is called on this page. "
        "[Code, tests and write-up on GitHub](https://github.com/mdsaeed24/llm-regression-detection)"
    )
settings = load_settings(ROOT / "config.yaml")
store = RunStore(ROOT / settings.storage.db_path)
records = store.list_runs(500)
if not records:
    st.info("No runs yet. Create one with `python -m regress baseline --prompt prompts/v1.yaml`.")
    st.stop()


def to_frame(rs: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in rs:
        a = r.aggregates
        rows.append(
            {
                "run_id": r.run_id,
                "created_at": r.created_at,
                "kind": r.kind,
                "prompt": r.prompt_version,
                "dataset": f"v{r.dataset_version}",
                "accuracy": a.accuracy,
                "pass_rate": a.pass_rate,
                "avg_relevance": a.avg_relevance,
                "cost_usd": a.cost_usd + a.judge_cost_usd,
                "latency_p95_ms": a.latency_p95_ms,
                "status": r.status or ("baseline" if r.kind == "baseline" else "-"),
                **{f"acc_{c}": a.per_category_accuracy.get(c) for c in CATEGORIES},
            }
        )
    frame = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    # Runs are events, not a time series: plot them in order, evenly spaced, so
    # three runs in one afternoon do not get squeezed against a week-long gap.
    frame["run"] = [f"{p} · {t:%m-%d %H:%M}" for p, t in zip(frame["prompt"], frame["created_at"])]
    return frame


df_all = to_frame(records)

st.title("Prompt regression dashboard")
st.caption(f"{len(df_all)} runs in `{settings.storage.db_path}` · accuracy = category correct; "
           f"pass = correct category and relevance ≥ {settings.evaluation.relevance_pass_min:g}")  # fmt: skip

# Filters in one row above the charts.
f1, f2 = st.columns(2)
datasets = sorted(df_all["dataset"].unique())
chosen_datasets = f1.multiselect("Dataset version", datasets, default=[datasets[-1]])
prompts = sorted(df_all["prompt"].unique())
chosen_prompts = f2.multiselect("Prompt version", prompts, default=prompts)
df = df_all[df_all["dataset"].isin(chosen_datasets) & df_all["prompt"].isin(chosen_prompts)]
if df.empty:
    st.warning("No runs match the filters.")
    st.stop()

latest = df.iloc[-1]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest accuracy", f"{latest.accuracy:.1%}")
m2.metric("Latest pass rate", f"{latest.pass_rate:.1%}")
m3.metric("Latest relevance", "n/a" if pd.isna(latest.avg_relevance) else f"{latest.avg_relevance:.2f}")
m4.metric("Latest status", f"{latest.status} · prompt {latest.prompt}")

# --- Trend per metric (one series -> no legend; the title names it) ----------
st.subheader("Trend")
metric_label = st.selectbox("Metric", list(METRICS))
column, fmt = METRICS[metric_label]
RUN_AXIS = alt.X("run:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30))
base = alt.Chart(df).encode(
    x=RUN_AXIS,
    y=alt.Y(
        f"{column}:Q", title=metric_label, scale=alt.Scale(zero=False),
        axis=alt.Axis(format=fmt, tickCount=5),
    ),
    tooltip=[
        "run_id",
        "prompt",
        "kind",
        "status",
        alt.Tooltip(f"{column}:Q", title=metric_label, format=fmt),
    ],
)
trend = base.mark_line(strokeWidth=2, color=SLOTS[0]) + base.mark_point(
    size=70, filled=True, color=SLOTS[0], stroke="white", strokeWidth=2
)
if column == "accuracy":
    floor = alt.Chart(pd.DataFrame({"floor": [settings.drift.accuracy_floor]}))
    trend += floor.mark_rule(color="#c3c2b7", strokeWidth=1).encode(y="floor:Q")
    trend += floor.mark_text(align="left", dx=4, dy=-6, color="#898781").encode(
        y="floor:Q", x=alt.value(0), text=alt.value(f"drift floor {settings.drift.accuracy_floor:.0%}")
    )
st.altair_chart(trend, width="stretch")

# --- Per-category accuracy (4 series, fixed slot order, legend) --------------
st.subheader("Accuracy by category")
per_cat = df.melt(
    id_vars=["created_at", "run", "run_id", "prompt"],
    value_vars=[f"acc_{c}" for c in CATEGORIES],
    var_name="category",
    value_name="category_accuracy",
).dropna()
per_cat["category"] = per_cat["category"].str.removeprefix("acc_")
cat_chart = (
    alt.Chart(per_cat)
    .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50, filled=True))
    .encode(
        x=RUN_AXIS,
        y=alt.Y("category_accuracy:Q", title="Accuracy", axis=alt.Axis(format=".0%")),
        color=alt.Color(
            "category:N",
            scale=alt.Scale(domain=list(CATEGORIES), range=SLOTS),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["run_id", "prompt", "category", alt.Tooltip("category_accuracy:Q", format=".1%")],
    )
)
st.altair_chart(cat_chart, width="stretch")
# Aqua and yellow sit below 3:1 contrast on the light surface, so the values are
# also available as a table (the accessibility relief for low-contrast series).
with st.expander("Show as table"):
    table = df[["run_id", "prompt", *[f"acc_{c}" for c in CATEGORIES]]].rename(
        columns={f"acc_{c}": c for c in CATEGORIES}
    )
    st.dataframe(
        table.sort_values("run_id", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in CATEGORIES},
    )

# --- History table ------------------------------------------------------------
st.subheader("Run history")
st.dataframe(
    df.sort_values("created_at", ascending=False)[
        ["run_id", "kind", "prompt", "dataset", "accuracy", "pass_rate", "avg_relevance",
         "cost_usd", "latency_p95_ms", "status"]
    ],  # fmt: skip
    hide_index=True,
    width="stretch",
    column_config={
        "accuracy": st.column_config.NumberColumn(format="percent"),
        "pass_rate": st.column_config.NumberColumn(format="percent"),
        "avg_relevance": st.column_config.NumberColumn(format="%.2f"),
        "cost_usd": st.column_config.NumberColumn(format="$%.4f"),
        "latency_p95_ms": st.column_config.NumberColumn(format="%.0f ms"),
    },
)

# --- Run vs run diff ----------------------------------------------------------
st.subheader("Compare any two runs")
run_ids = list(df_all.sort_values("created_at", ascending=False)["run_id"])
baselines = [r.run_id for r in records if r.kind == "baseline"]
c1, c2 = st.columns(2)
run_a = c1.selectbox("Before (A)", run_ids, index=run_ids.index(baselines[0]) if baselines else 0)
run_b = c2.selectbox("After (B)", run_ids, index=0)

if run_a == run_b:
    st.info("Pick two different runs.")
else:
    comparison = compare_runs(store.load(run_a), store.load(run_b))
    dataset_path = ROOT / "data/golden.json"
    emails = {c.id: c.input_email for c in load_dataset(dataset_path).cases} if dataset_path.exists() else {}

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Accuracy", f"{comparison.accuracy_after:.1%}", f"{comparison.accuracy_delta_pp:+.1f} pp")
    s2.metric("Pass rate", f"{comparison.pass_rate_after:.1%}", f"{comparison.pass_rate_delta_pp:+.1f} pp")
    if comparison.avg_relevance_after is not None and comparison.avg_relevance_delta is not None:
        s3.metric("Avg relevance", f"{comparison.avg_relevance_after:.2f}", f"{comparison.avg_relevance_delta:+.2f}")
    s4.metric("Regressions / improvements", f"{len(comparison.regressions)} / {len(comparison.improvements)}")
    if comparison.dataset_mismatch:
        st.warning(f"Different dataset versions; compared the {comparison.compared_cases} shared cases.")

    st.dataframe(
        pd.DataFrame(
            [
                {"category": c, "cases": d.cases, "A": d.before, "B": d.after, "change (pp)": d.delta_pp}
                for c, d in comparison.per_category.items()
            ]
        ),
        hide_index=True,
        column_config={
            "A": st.column_config.NumberColumn(format="percent"),
            "B": st.column_config.NumberColumn(format="percent"),
            "change (pp)": st.column_config.NumberColumn(format="%+.1f"),
        },
    )

    def changes_frame(changes):  # type: ignore[no-untyped-def]
        return pd.DataFrame(
            [
                {
                    "case": ch.case_id,
                    "expected": ch.expected_category,
                    "difficulty": ch.expected_difficulty,
                    "email": emails.get(ch.case_id, ""),
                    "A output": f"{ch.before.predicted_category}: {ch.before.summary or ch.before.error}",
                    "B output": f"{ch.after.predicted_category}: {ch.after.summary or ch.after.error}",
                    "why": ch.reason,
                }
                for ch in changes
            ]
        )

    st.markdown(f"**Regressions (pass → fail): {len(comparison.regressions)}**")
    if comparison.regressions:
        st.dataframe(changes_frame(comparison.regressions), hide_index=True, width="stretch")
    st.markdown(f"**Improvements (fail → pass): {len(comparison.improvements)}**")
    if comparison.improvements:
        st.dataframe(changes_frame(comparison.improvements), hide_index=True, width="stretch")
