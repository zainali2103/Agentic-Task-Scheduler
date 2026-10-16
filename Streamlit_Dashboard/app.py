"""
NeuroPlanner Dashboard
Autonomous Task Planning with LLM-Powered Dynamic Scheduling
Zain Ali - MSc Data Science, Arden University (STU227908)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import json
import os
import copy

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroPlanner Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent with report figures)
# ──────────────────────────────────────────────────────────────────────────────
COLOURS = {
    "fifo":     "#D85A30",
    "pq":       "#BA7517",
    "edd":      "#2E75B6",
    "dqn":      "#1D9E75",
    "oracle":   "#8B5CF6",
    "bg_card":  "#F8F9FA",
    "text":     "#1A1A2E",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@st.cache_data
def load_baselines():
    return pd.read_csv(os.path.join(DATA_DIR, "baseline_results.csv"))


@st.cache_data
def load_dqn(seed=""):
    suffix = f"_seed{seed}" if seed else ""
    return pd.read_csv(os.path.join(DATA_DIR, f"dqn_test_results{suffix}.csv"))


@st.cache_data
def load_training():
    return pd.read_csv(os.path.join(DATA_DIR, "training_results.csv"))


@st.cache_data
def load_test_scenarios():
    with open(os.path.join(DATA_DIR, "test_scenarios.json")) as f:
        return json.load(f)


@st.cache_data
def load_all_seeds():
    """Load all 5 seed results and compute summary statistics."""
    seeds = {"42": load_dqn(), "7": load_dqn("7"), "17": load_dqn("17"),
             "31": load_dqn("31"), "53": load_dqn("53")}
    seed_tcrs = {s: df["tcr"].mean() for s, df in seeds.items()}
    vals = list(seed_tcrs.values())
    mean_tcr = np.mean(vals)
    std_tcr = np.std(vals, ddof=1)
    ci_lo = mean_tcr - 2.776 * std_tcr / np.sqrt(5)  # t(4,0.025)
    ci_hi = mean_tcr + 2.776 * std_tcr / np.sqrt(5)
    return seeds, seed_tcrs, mean_tcr, std_tcr, ci_lo, ci_hi


# ──────────────────────────────────────────────────────────────────────────────
# Helper: metric card
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, delta_color="normal"):
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("NeuroPlanner")
st.sidebar.caption("MSc Dissertation Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Training", "Compare", "Live Demo"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Zain Ali** (STU227908)")
st.sidebar.markdown("Arden University, 2026")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Gantt chart builder (dark theme, clear labels)
# ──────────────────────────────────────────────────────────────────────────────
def _build_gantt(gantt_data):
    """Build a dark-themed Gantt chart with deadline markers and hover info."""
    fig = go.Figure()
    seen_on_time = False
    seen_late = False
    for row in gantt_data:
        is_on_time = row["Status"] == "On time"
        color = "#2ECC71" if is_on_time else "#E74C3C"
        label = "On time" if is_on_time else "Late / Missed"
        if is_on_time and not seen_on_time:
            show = True; seen_on_time = True
        elif not is_on_time and not seen_late:
            show = True; seen_late = True
        else:
            show = False

        duration = row["End"] - row["Start"]
        fig.add_trace(go.Bar(
            x=[duration], y=[row["Task"]], base=[row["Start"]],
            orientation="h",
            marker_color=color, marker_line=dict(color="white", width=0.5),
            name=label, showlegend=show,
            hovertemplate=(
                f"<b>{row['Task']}</b><br>"
                f"Start: {row['Start']} min<br>"
                f"End: {row['End']} min<br>"
                f"Duration: {duration} min<br>"
                f"Deadline: {row['Deadline']} min<br>"
                f"Status: {row['Status']}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[row["Deadline"]], y=[row["Task"]],
            mode="markers",
            marker=dict(symbol="diamond", size=12, color="#F1C40F",
                        line=dict(color="white", width=1)),
            showlegend=False,
            hovertemplate=f"Deadline: {row['Deadline']} min<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="DQN Schedule (Gantt Chart)", font=dict(color="white", size=16)),
        xaxis_title="Time (minutes from 8:00 AM)",
        height=max(300, len(gantt_data) * 45 + 120),
        plot_bgcolor="#1E1E2E",
        paper_bgcolor="#1E1E2E",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#3A3A5C", zerolinecolor="#3A3A5C", color="white"),
        yaxis=dict(gridcolor="#3A3A5C", color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=40, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("NeuroPlanner: Overview")
    st.markdown(
        "An agentic AI system that combines LLM-based natural language parsing "
        "(Mistral 7B) with Deep Reinforcement Learning (DQN) for autonomous "
        "task scheduling in personal productivity environments."
    )

    # Key results row
    bl = load_baselines()
    dqn42 = load_dqn()
    _, seed_tcrs, mean_tcr, std_tcr, ci_lo, ci_hi = load_all_seeds()

    fifo_tcr = bl[bl.scheduler == "FIFO"]["tcr"].mean() * 100
    pq_tcr = bl[bl.scheduler == "Priority Queue"]["tcr"].mean() * 100
    edd_tcr = bl[bl.scheduler == "EDD"]["tcr"].mean() * 100
    dqn_tcr = dqn42["tcr"].mean() * 100

    st.markdown("### Key Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("FIFO", f"{fifo_tcr:.2f}%")
    with c2:
        metric_card("Priority Queue", f"{pq_tcr:.2f}%")
    with c3:
        metric_card("EDD", f"{edd_tcr:.2f}%")
    with c4:
        metric_card("DQN (seed 42)", f"{dqn_tcr:.2f}%",
                     delta=f"+{dqn_tcr - fifo_tcr:.2f} pp vs FIFO")
    with c5:
        metric_card("DQN (5-seed mean)", f"{mean_tcr * 100:.2f}%",
                     delta=f"SD {std_tcr * 100:.2f}%")

    st.markdown("---")

    # Oracle ceiling
    st.markdown("### Oracle TCR Ceiling")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "The oracle scheduler uses perfect foresight to compute the maximum "
            "achievable Task Completion Rate for each scenario. On the 200 test "
            "scenarios the oracle ceiling is **48.40%**, confirming that the "
            "original 75% target was mathematically impossible under the stress-test "
            "dataset design (80.5% infeasible scenarios)."
        )
        st.markdown(f"- DQN (seed 42) achieves **{dqn_tcr / 48.40 * 100:.1f}%** of the oracle ceiling")
        st.markdown(f"- DQN (5-seed mean) achieves **{mean_tcr * 100 / 48.40 * 100:.1f}%** of the oracle ceiling")
    with col_b:
        fig_oracle = go.Figure()
        labels = ["FIFO", "PQ", "EDD", "DQN", "Oracle"]
        values = [fifo_tcr, pq_tcr, edd_tcr, dqn_tcr, 48.40]
        colors = [COLOURS["fifo"], COLOURS["pq"], COLOURS["edd"],
                  COLOURS["dqn"], COLOURS["oracle"]]
        fig_oracle.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:.2f}%" for v in values],
            textposition="outside",
            textfont=dict(color="black", size=13),
        ))
        fig_oracle.update_layout(
            title="TCR: All Schedulers vs Oracle Ceiling",
            yaxis_title="TCR (%)", yaxis_range=[0, 60],
            height=350, margin=dict(t=40, b=20),
            plot_bgcolor="white",
        )
        st.plotly_chart(fig_oracle, use_container_width=True)

    st.markdown("---")

    # Quick info expanders
    with st.expander("System Architecture"):
        st.code("""
User Input (natural language)
        |
        v
NLP Pipeline (Mistral 7B via Ollama)
        |  extracts: name, deadline, duration, priority, dependencies
        v
Task Queue (structured JSON)
        |
        v
DQN Scheduling Agent
        |  observes: 140-dim state vector (20 tasks x 7 features)
        |  actions:  select next task to execute (discrete 0-19)
        |  reward:   +10 on-time, -15 missed, -2 reorder, +5 dep. resolved
        v
Optimised Schedule
        """, language=None)

    with st.expander("Dataset Info"):
        ds_cols = st.columns(4)
        with ds_cols[0]:
            metric_card("Total Scenarios", "1,000")
        with ds_cols[1]:
            metric_card("Train / Test", "800 / 200")
        with ds_cols[2]:
            metric_card("Infeasibility Rate", "80.5%")
        with ds_cols[3]:
            metric_card("Workday", "600 min (10 hrs)")
        st.info(
            "The dataset is a deliberately challenging stress-test benchmark "
            "designed to force prioritisation trade-offs, not a realistic "
            "representation of typical personal productivity workloads."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2: TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Training":
    st.title("DQN Training Progress")
    st.markdown("Training over 5,000 episodes with seed 42. Double DQN with Huber loss.")

    tr = load_training()

    # Smoothing slider
    window = st.slider("Smoothing window", 10, 500, 100, step=10)

    def smooth(vals, w):
        return pd.Series(vals).rolling(w, min_periods=1).mean().values

    # TCR chart
    fig_tcr = go.Figure()
    fig_tcr.add_trace(go.Scatter(
        x=tr["episode"], y=tr["tcr"] * 100,
        mode="lines", line=dict(color=COLOURS["dqn"], width=0.5),
        opacity=0.15, name="Raw", showlegend=True,
    ))
    fig_tcr.add_trace(go.Scatter(
        x=tr["episode"], y=smooth(tr["tcr"] * 100, window),
        mode="lines", line=dict(color=COLOURS["dqn"], width=2.5),
        name=f"Smoothed (w={window})",
    ))
    fig_tcr.add_hline(y=38.49, line_dash="dash", line_color=COLOURS["fifo"],
                      annotation_text="FIFO: 38.49%")
    fig_tcr.add_hline(y=37.79, line_dash="dash", line_color=COLOURS["pq"],
                      annotation_text="PQ: 37.79%")
    fig_tcr.add_hline(y=38.58, line_dash="dot", line_color=COLOURS["edd"],
                      annotation_text="EDD: 38.58%")
    fig_tcr.update_layout(
        title="Task Completion Rate over Training",
        xaxis_title="Episode", yaxis_title="TCR (%)",
        height=400, plot_bgcolor="white",
        legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
    )
    st.plotly_chart(fig_tcr, use_container_width=True)

    # Reward chart
    fig_rew = go.Figure()
    fig_rew.add_trace(go.Scatter(
        x=tr["episode"], y=tr["reward"],
        mode="lines", line=dict(color="#534AB7", width=0.5),
        opacity=0.15, name="Raw",
    ))
    fig_rew.add_trace(go.Scatter(
        x=tr["episode"], y=smooth(tr["reward"], window),
        mode="lines", line=dict(color="#534AB7", width=2.5),
        name=f"Smoothed (w={window})",
    ))
    fig_rew.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rew.update_layout(
        title="Episode Reward over Training",
        xaxis_title="Episode", yaxis_title="Total Reward",
        height=350, plot_bgcolor="white",
    )
    st.plotly_chart(fig_rew, use_container_width=True)

    # Epsilon and Loss side by side
    col_e, col_l = st.columns(2)
    with col_e:
        fig_eps = go.Figure()
        fig_eps.add_trace(go.Scatter(
            x=tr["episode"], y=tr["epsilon"],
            mode="lines", line=dict(color=COLOURS["pq"], width=2),
            fill="tozeroy", fillcolor="rgba(186,117,23,0.15)",
        ))
        fig_eps.update_layout(
            title="Epsilon Decay",
            xaxis_title="Episode", yaxis_title="Epsilon",
            height=300, plot_bgcolor="white",
        )
        st.plotly_chart(fig_eps, use_container_width=True)

    with col_l:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            x=tr["episode"], y=tr["loss"],
            mode="lines", line=dict(color=COLOURS["edd"], width=0.5),
            opacity=0.2, name="Raw",
        ))
        fig_loss.add_trace(go.Scatter(
            x=tr["episode"], y=smooth(tr["loss"], window),
            mode="lines", line=dict(color=COLOURS["edd"], width=2.5),
            name="Smoothed",
        ))
        fig_loss.update_layout(
            title="Training Loss (Huber)",
            xaxis_title="Episode", yaxis_title="Loss",
            height=300, plot_bgcolor="white",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Hyperparameters table
    st.markdown("### Hyperparameters")
    hp = pd.DataFrame({
        "Parameter": ["Training episodes", "Replay buffer", "Batch size",
                       "Learning rate", "Epsilon decay", "Target net update",
                       "Discount factor", "Algorithm"],
        "Value": ["5,000", "10,000 transitions", "64", "0.001 (Adam)",
                  "1.0 to 0.05 over 50,000 steps", "Every 100 episodes",
                  "0.95", "Double DQN + Huber loss"],
    })
    st.table(hp)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3: COMPARE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Compare":
    st.title("Scheduler Comparison")
    st.markdown(
        "Comparing four schedulers across 200 test scenarios: FIFO, "
        "Priority Queue (0.7 deadline / 0.3 priority weighting), "
        "Earliest Due Date (EDD), and DQN (seed 42). "
        "Statistical tests computed live using SciPy."
    )

    bl = load_baselines()
    dqn42 = load_dqn()
    seeds_data, seed_tcrs, mean_tcr, std_tcr, ci_lo, ci_hi = load_all_seeds()

    fifo = bl[bl.scheduler == "FIFO"].sort_values("scenario_id").reset_index(drop=True)
    pq = bl[bl.scheduler == "Priority Queue"].sort_values("scenario_id").reset_index(drop=True)
    edd = bl[bl.scheduler == "EDD"].sort_values("scenario_id").reset_index(drop=True)
    dqn = dqn42.sort_values("scenario_id").reset_index(drop=True)

    # ── Overall summary ───────────────────────────────────────────────────────
    st.markdown("### Overall Results")

    summary = pd.DataFrame({
        "Scheduler": ["FIFO", "Priority Queue", "EDD", "DQN (seed 42)", "DQN (5-seed mean)"],
        "TCR (%)": [
            f"{fifo.tcr.mean()*100:.2f}",
            f"{pq.tcr.mean()*100:.2f}",
            f"{edd.tcr.mean()*100:.2f}",
            f"{dqn.tcr.mean()*100:.2f}",
            f"{mean_tcr*100:.2f} (SD {std_tcr*100:.2f})",
        ],
        "Mean DD (min)": [
            f"{fifo.dd.mean():.2f}",
            f"{pq.dd.mean():.2f}",
            f"{edd.dd.mean():.2f}",
            f"{dqn.dd.mean():.2f}",
            "-",
        ],
        "Mean SRT (ms)": [
            "0.19",
            "0.31",
            "10.42",
            f"{dqn.mean_inference_ms.mean():.2f}" if "mean_inference_ms" in dqn.columns else "0.24",
            "-",
        ],
    })
    st.table(summary)

    # ── TCR bar chart ─────────────────────────────────────────────────────────
    st.markdown("### TCR Comparison")
    fig_bar = go.Figure()
    schedulers = ["FIFO", "Priority Queue", "EDD", "DQN"]
    tcr_vals = [fifo.tcr.mean()*100, pq.tcr.mean()*100,
                edd.tcr.mean()*100, dqn.tcr.mean()*100]
    bar_colors = [COLOURS["fifo"], COLOURS["pq"], COLOURS["edd"], COLOURS["dqn"]]

    fig_bar.add_trace(go.Bar(
        x=schedulers, y=tcr_vals,
        marker_color=bar_colors,
        text=[f"{v:.2f}%" for v in tcr_vals],
        textposition="outside",
        textfont=dict(color="black", size=13),
    ))
    fig_bar.update_layout(
        yaxis_title="TCR (%)", yaxis_range=[0, 55],
        height=400, plot_bgcolor="white",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Deadline trade-off analysis ─────────────────────────────────────────
    st.markdown("### Deadline Adherence Trade-off")
    st.markdown(
        "DQN prioritises task completion rate over strict deadline adherence. "
        "The table below shows the trade-off: baselines with simpler strategies "
        "achieve higher on-time percentages but lower overall TCR."
    )

    def compute_deadline_stats(df, label):
        if "on_time" in df.columns:
            total_completed = df["completed"].sum()
            total_on_time = df["on_time"].sum()
            total_tasks = df["completed"].sum() + df["missed"].sum()
            on_time_pct = total_on_time / total_tasks * 100 if total_tasks > 0 else 0
            late_count = total_tasks - total_on_time
        else:
            on_time_pct = df.tcr.mean() * 100
            late_count = "-"
        return {"Scheduler": label, "On-time (%)": f"{on_time_pct:.1f}%",
                "Late Tasks (total)": f"{late_count}"}

    deadline_data = [
        compute_deadline_stats(fifo, "FIFO"),
        compute_deadline_stats(pq, "Priority Queue"),
        compute_deadline_stats(edd, "EDD"),
        compute_deadline_stats(dqn, "DQN (seed 42)"),
    ]
    st.table(pd.DataFrame(deadline_data))

    # ── By complexity ─────────────────────────────────────────────────────────
    st.markdown("### TCR by Complexity Level")
    complexity_data = []
    for comp in ["low", "medium", "high"]:
        complexity_data.append({
            "Complexity": comp.title(),
            "FIFO": f"{fifo[fifo.complexity==comp].tcr.mean()*100:.1f}%",
            "Priority Queue": f"{pq[pq.complexity==comp].tcr.mean()*100:.1f}%",
            "EDD": f"{edd[edd.complexity==comp].tcr.mean()*100:.1f}%",
            "DQN": f"{dqn[dqn.complexity==comp].tcr.mean()*100:.1f}%",
        })
    st.table(pd.DataFrame(complexity_data))

    # Grouped bar chart by complexity
    fig_comp = go.Figure()
    for sched, df_s, col in [("FIFO", fifo, COLOURS["fifo"]),
                              ("PQ", pq, COLOURS["pq"]),
                              ("EDD", edd, COLOURS["edd"]),
                              ("DQN", dqn, COLOURS["dqn"])]:
        vals = [df_s[df_s.complexity == c].tcr.mean() * 100 for c in ["low", "medium", "high"]]
        fig_comp.add_trace(go.Bar(
            name=sched, x=["Low", "Medium", "High"], y=vals,
            marker_color=col, text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(color="black", size=12),
        ))
    fig_comp.update_layout(
        barmode="group", yaxis_title="TCR (%)", yaxis_range=[0, 75],
        height=400, plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── Statistical tests ─────────────────────────────────────────────────────
    st.markdown("### Statistical Tests (Per-Scenario, seed 42)")
    st.markdown("Paired comparisons across 200 test scenarios.")

    def run_test(a, b, label):
        """Run paired t-test and Mann-Whitney U, return dict."""
        diff = a - b
        _, p_shapiro = stats.shapiro(diff)
        t_stat, p_ttest = stats.ttest_rel(a, b)
        u_stat, p_mann = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = diff.mean() / diff.std() if diff.std() > 0 else 0
        ci_lo = diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))
        ci_hi = diff.mean() + 1.96 * diff.std() / np.sqrt(len(diff))
        return {
            "Comparison": label,
            "Mean Diff (pp)": f"{diff.mean()*100:+.2f}",
            "95% CI": f"[{ci_lo*100:.2f}, {ci_hi*100:.2f}]",
            "Cohen's d": f"{d:.3f}",
            "t-test p": f"{p_ttest:.4f}",
            "Mann-Whitney p": f"{p_mann:.4f}",
            "Shapiro p (normality)": f"{p_shapiro:.4f}",
        }

    test_results = [
        run_test(dqn.tcr.values, fifo.tcr.values, "DQN vs FIFO"),
        run_test(dqn.tcr.values, pq.tcr.values, "DQN vs PQ"),
        run_test(dqn.tcr.values, edd.tcr.values, "DQN vs EDD"),
    ]
    st.table(pd.DataFrame(test_results))

    # High-complexity focused
    st.markdown("### High-Complexity Scenarios")
    dqn_hi = dqn[dqn.complexity == "high"].tcr.values
    fifo_hi = fifo[fifo.complexity == "high"].tcr.values
    pq_hi = pq[pq.complexity == "high"].tcr.values
    edd_hi = edd[edd.complexity == "high"].tcr.values

    hi_results = [
        run_test(dqn_hi, fifo_hi, "DQN vs FIFO (high)"),
        run_test(dqn_hi, pq_hi, "DQN vs PQ (high)"),
        run_test(dqn_hi, edd_hi, "DQN vs EDD (high)"),
    ]
    st.table(pd.DataFrame(hi_results))

    # ── Multi-seed summary ────────────────────────────────────────────────────
    st.markdown("### Multi-Seed Robustness (5 seeds)")
    st.markdown(
        "To assess training stability, the DQN was trained independently "
        "with 5 random seeds. Results below show per-seed TCR on the same "
        "200 test scenarios."
    )

    seed_df = pd.DataFrame({
        "Seed": list(seed_tcrs.keys()),
        "TCR (%)": [f"{v*100:.2f}" for v in seed_tcrs.values()],
    })
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        st.table(seed_df)
        st.markdown(f"**Mean:** {mean_tcr*100:.2f}%")
        st.markdown(f"**SD:** {std_tcr*100:.2f}%")
        st.markdown(f"**95% CI:** [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]")
        st.markdown(f"4 out of 5 seeds outperform all three baselines.")

    with col_s2:
        fig_seeds = go.Figure()
        fig_seeds.add_trace(go.Bar(
            x=[f"Seed {s}" for s in seed_tcrs.keys()],
            y=[v * 100 for v in seed_tcrs.values()],
            marker_color=COLOURS["dqn"],
            text=[f"{v*100:.2f}%" for v in seed_tcrs.values()],
            textposition="outside",
            textfont=dict(color="black", size=13),
        ))
        fig_seeds.add_hline(y=mean_tcr * 100, line_dash="solid",
                            line_color="black", line_width=1.5,
                            annotation_text=f"Mean: {mean_tcr*100:.2f}%")
        fig_seeds.add_hline(y=fifo.tcr.mean() * 100, line_dash="dash",
                            line_color=COLOURS["fifo"],
                            annotation_text="FIFO: 38.49%")
        fig_seeds.update_layout(
            yaxis_title="TCR (%)", yaxis_range=[0, 55],
            height=350, plot_bgcolor="white",
        )
        st.plotly_chart(fig_seeds, use_container_width=True)

    # Cross-seed significance
    st.markdown("**Cross-seed significance** (5-seed means vs baselines):")
    baseline_means = {
        "FIFO": fifo.tcr.mean(),
        "PQ": pq.tcr.mean(),
        "EDD": edd.tcr.mean(),
    }
    seed_vals = np.array(list(seed_tcrs.values()))
    cross_tests = []
    for bname, bval in baseline_means.items():
        t_s, p_v = stats.ttest_1samp(seed_vals, bval)
        cross_tests.append({
            "Comparison": f"DQN seeds vs {bname}",
            "t-statistic": f"{t_s:.3f}",
            "p-value": f"{p_v:.4f}",
            "Significant (p<0.05)": "Yes" if p_v < 0.05 else "No (borderline)" if p_v < 0.10 else "No",
        })
    st.table(pd.DataFrame(cross_tests))


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4: LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Demo":
    st.title("Live Demo: DQN Scheduler")
    st.markdown(
        "Type tasks in natural language. The system will parse them using "
        "Mistral 7B (via Ollama) and schedule them using the trained DQN agent."
    )

    # ── Load DQN model ────────────────────────────────────────────────────────
    import torch
    import torch.nn as nn

    WORKDAY_MINUTES = 600
    MAX_TASKS = 20
    N_FEATURES = 7
    STATE_SIZE = 140
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class DQNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_SIZE, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, MAX_TASKS),
            )
        def forward(self, x):
            return self.net(x)

    @st.cache_resource
    def load_model():
        net = DQNetwork().to(DEVICE)
        model_path = os.path.join(DATA_DIR, "dqn_model.pth")
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        net.load_state_dict(checkpoint["online_net"])
        net.eval()
        return net

    net = load_model()

    # ── NLP parsing via Ollama ────────────────────────────────────────────────
    def parse_tasks_with_llm(user_input):
        """Parse natural language tasks using Mistral 7B via Ollama."""
        try:
            import requests
            prompt = f"""You are a task parser. Extract tasks from the user input below.
For each task, return a JSON array of objects with these fields:
- "name": short task name
- "deadline_min": deadline in minutes from start of workday (0=8:00AM, 600=6:00PM). Use 600 if no deadline given.
- "duration_min": estimated duration in minutes. Estimate if not given.
- "priority": 1 (lowest) to 5 (highest). Estimate from urgency words.
- "dependencies": list of task names this depends on (empty list if none)
- "vague": true if the task description is ambiguous

Return ONLY valid JSON array, no other text.

User input: {user_input}"""

            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=120,
            )
            if resp.status_code == 200:
                text = resp.json().get("response", "")
                # Try to extract JSON from response
                import re
                match = re.search(r'\[.*\]', text, re.DOTALL)
                if match:
                    tasks = json.loads(match.group())
                    # Add IDs
                    for i, t in enumerate(tasks):
                        t["id"] = f"T{i+1}"
                        t.setdefault("dependencies", [])
                        t.setdefault("vague", False)
                        t.setdefault("deadline_min", 600)
                        t.setdefault("duration_min", 30)
                        t.setdefault("priority", 3)
                    return tasks, None
            return None, "Ollama returned an error"
        except requests.exceptions.ConnectionError:
            return None, "Ollama is not running. Start it with: ollama serve"
        except Exception as e:
            return None, str(e)

    # ── DQN scheduling functions ──────────────────────────────────────────────
    def encode_task(task, current_time, completed_ids):
        time_left = max(task["deadline_min"] - current_time, 0)
        deadline_urgency = float(np.clip(1.0 - (time_left / WORKDAY_MINUTES), 0.0, 1.0))
        duration_norm = float(np.clip(task["duration_min"] / 240.0, 0.0, 1.0))
        priority_norm = (task["priority"] - 1) / 4.0
        unmet = [d for d in task.get("dependencies", []) if d not in completed_ids]
        has_dep = 1.0 if unmet else 0.0
        is_vague = 1.0 if task.get("vague", False) else 0.0
        time_pressure = float(np.clip(task["duration_min"] / max(time_left, 1), 0.0, 1.0))
        return np.array([deadline_urgency, duration_norm, priority_norm,
                         has_dep, is_vague, time_pressure, 1.0], dtype=np.float32)

    def encode_state(queue, current_time, completed_ids):
        state = np.zeros((MAX_TASKS, N_FEATURES), dtype=np.float32)
        for i, task in enumerate(queue[:MAX_TASKS]):
            state[i] = encode_task(task, current_time, completed_ids)
        return state.flatten()

    def greedy_action(state, queue):
        mask = np.zeros(MAX_TASKS, dtype=bool)
        mask[:len(queue)] = True
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q = net(s).squeeze(0)
            m = torch.BoolTensor(mask).to(DEVICE)
            q[~m] = float("-inf")
            return int(q.argmax().item())

    def run_dqn_schedule(tasks):
        """Run the DQN scheduler on a list of parsed tasks."""
        queue = copy.deepcopy(tasks)
        current_time = 0
        completed = []
        missed = []
        schedule_log = []
        steps = 0

        while queue and current_time < WORKDAY_MINUTES and steps < 200:
            steps += 1
            completed_ids = {t["id"] for t in completed}
            state = encode_state(queue, current_time, completed_ids)
            action = greedy_action(state, queue)
            action = min(action, len(queue) - 1)
            task = queue.pop(action)

            # Dependency check
            if task.get("dependencies"):
                unmet = [d for d in task["dependencies"] if d not in completed_ids]
                if unmet:
                    queue.append(task)
                    current_time += 10
                    continue

            finish = current_time + task["duration_min"]
            if finish > WORKDAY_MINUTES:
                missed.append(task)
                schedule_log.append({
                    "Task": task["name"], "Start": current_time,
                    "End": finish, "Status": "Missed (no time)",
                    "Deadline": task["deadline_min"],
                })
                continue

            on_time = finish <= task["deadline_min"]
            current_time = finish
            completed.append({**task, "finish_time": finish, "on_time": on_time})
            schedule_log.append({
                "Task": task["name"],
                "Start": finish - task["duration_min"],
                "End": finish,
                "Status": "On time" if on_time else "Late",
                "Deadline": task["deadline_min"],
            })

        for t in queue:
            missed.append(t)
            schedule_log.append({
                "Task": t["name"], "Start": "-", "End": "-",
                "Status": "Missed (not reached)", "Deadline": t["deadline_min"],
            })

        return completed, missed, schedule_log

    # ── UI ────────────────────────────────────────────────────────────────────
    st.markdown("---")

    # Option to use sample scenario or free text
    demo_mode = st.radio(
        "Input mode",
        ["Type your own tasks", "Use a test scenario"],
        horizontal=True,
    )

    if demo_mode == "Type your own tasks":
        user_input = st.text_area(
            "Enter your tasks (natural language)",
            placeholder="Example: Finish the report by 2pm (2 hours), "
                        "reply to emails (30 mins, low priority), "
                        "prepare slides for 4pm meeting (1.5 hours, high priority)",
            height=120,
        )
        if st.button("Parse and Schedule", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some tasks first.")
            else:
                with st.spinner("Parsing tasks with Mistral 7B..."):
                    tasks, error = parse_tasks_with_llm(user_input)

                if error:
                    st.error(f"NLP parsing failed: {error}")
                    st.info(
                        "The Live Demo requires Ollama running locally with the "
                        "Mistral model. Run: `ollama serve` then `ollama pull mistral`"
                    )
                else:
                    st.success(f"Parsed {len(tasks)} tasks")
                    st.markdown("**Parsed tasks:**")
                    st.json(tasks)

                    with st.spinner("Running DQN scheduler..."):
                        completed, missed, log = run_dqn_schedule(tasks)

                    total = len(completed) + len(missed)
                    on_time = [t for t in completed if t.get("on_time")]
                    tcr = len(on_time) / total * 100 if total > 0 else 0

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        metric_card("Tasks Completed", f"{len(completed)}/{total}")
                    with c2:
                        metric_card("On Time", f"{len(on_time)}/{total}")
                    with c3:
                        metric_card("TCR", f"{tcr:.1f}%")

                    st.markdown("**Schedule:**")
                    st.table(pd.DataFrame(log))

                    # Gantt chart
                    gantt_data = [r for r in log if r["Start"] != "-"]
                    if gantt_data:
                        fig_gantt = _build_gantt(gantt_data)
                        st.plotly_chart(fig_gantt, use_container_width=True)

    else:
        # Use a random test scenario
        test_scenarios = load_test_scenarios()
        scenario_ids = [s["scenario_id"] for s in test_scenarios]
        selected = st.selectbox("Pick a test scenario", scenario_ids)

        if st.button("Run DQN on this scenario", type="primary"):
            scenario = next(s for s in test_scenarios if s["scenario_id"] == selected)
            st.markdown(f"**Scenario:** {selected} | "
                        f"**Complexity:** {scenario['complexity']} | "
                        f"**Tasks:** {scenario['n_tasks']} | "
                        f"**Disruptions:** {scenario['n_disruptions']}")

            with st.spinner("Running DQN scheduler..."):
                completed, missed, log = run_dqn_schedule(scenario["tasks"])

            total = len(completed) + len(missed)
            on_time = [t for t in completed if t.get("on_time")]
            tcr = len(on_time) / total * 100 if total > 0 else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Tasks Completed", f"{len(completed)}/{total}")
            with c2:
                metric_card("On Time", f"{len(on_time)}/{total}")
            with c3:
                metric_card("TCR", f"{tcr:.1f}%")

            st.markdown("**Schedule:**")
            st.table(pd.DataFrame(log))

            # Gantt chart for test scenario
            gantt_data = [r for r in log if r["Start"] != "-"]
            if gantt_data:
                fig_gantt = _build_gantt(gantt_data)
                st.plotly_chart(fig_gantt, use_container_width=True)

    st.markdown("---")
    st.markdown("#### How to read the results")
    st.markdown(
        "- **Green bars** = tasks completed **on time** (before their deadline)\n"
        "- **Red bars** = tasks completed **late** or missed entirely\n"
        "- **Yellow diamonds** = the task's **deadline** position\n"
        "- **TCR** = Task Completion Rate: percentage of tasks finished on time\n"
        "- **Hover** over any bar or diamond for full timing details"
    )
    st.markdown("---")
    st.caption(
        "Note: The 'Type your own tasks' mode requires Ollama running locally "
        "with the Mistral 7B model. The 'Use a test scenario' mode works "
        "without Ollama as it uses pre-generated structured data."
    )
