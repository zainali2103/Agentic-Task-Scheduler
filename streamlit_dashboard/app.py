"""
NeuroPlanner Dashboard
Streamlit-based interactive dashboard for the autonomous task planning system.
Loads real data, runs real model inference, and connects to Mistral 7B via Ollama.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import copy
import random
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import requests
import time
import os

# ── Page config ──
st.set_page_config(
    page_title="NeuroPlanner Dashboard",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ──
WORKDAY_MINUTES = 600
MAX_TASKS = 20
N_FEATURES = 7
STATE_SIZE = MAX_TASKS * N_FEATURES
R_ON_TIME = 10.0
R_MISSED = -15.0
R_REORDER = -2.0
R_DEP_RESOLVED = 5.0
R_VAGUE = 3.0

# ── Data paths ──
DATA_DIR = "data"
TRAIN_RESULTS_PATH = os.path.join(DATA_DIR, "training_results.csv")
DQN_RESULTS_PATH = os.path.join(DATA_DIR, "dqn_test_results.csv")
BASELINE_RESULTS_PATH = os.path.join(DATA_DIR, "baseline_results.csv")
MODEL_PATH = os.path.join(DATA_DIR, "dqn_model.pth")
SCENARIOS_PATH = os.path.join(DATA_DIR, "scenarios.json")
TEST_SCENARIOS_PATH = os.path.join(DATA_DIR, "test_scenarios.json")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL AND ENVIRONMENT CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class DQNetwork(nn.Module):
    """Deep Q-Network with 3 hidden layers. Input: 140-dim state, Output: 20 Q-values."""
    def __init__(self, state_size=STATE_SIZE, action_size=MAX_TASKS,
                 h1=256, h2=256, h3=128):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, action_size),
        )

    def forward(self, x):
        return self.net(x)


def encode_task(task, current_time, completed_ids):
    """Encode a single task into a 7-dimensional feature vector."""
    time_left = max(task['deadline_min'] - current_time, 0)
    deadline_urgency = 1.0 - (time_left / WORKDAY_MINUTES)
    deadline_urgency = float(np.clip(deadline_urgency, 0.0, 1.0))
    duration_norm = float(np.clip(task['duration_min'] / 240.0, 0.0, 1.0))
    priority_norm = (task['priority'] - 1) / 4.0
    unmet_deps = [d for d in task.get('dependencies', []) if d not in completed_ids]
    has_dependency = 1.0 if unmet_deps else 0.0
    is_vague = 1.0 if task.get('vague', False) else 0.0
    time_pressure = float(np.clip(task['duration_min'] / max(time_left, 1), 0.0, 1.0))
    is_available = 1.0
    return np.array([deadline_urgency, duration_norm, priority_norm,
                     has_dependency, is_vague, time_pressure, is_available],
                    dtype=np.float32)


def encode_state(task_queue, current_time, completed_ids):
    """Encode the full task queue into a flat 140-dim state vector."""
    state = np.zeros((MAX_TASKS, N_FEATURES), dtype=np.float32)
    sorted_queue = sorted(task_queue, key=lambda t: t['deadline_min'])
    for i, task in enumerate(sorted_queue[:MAX_TASKS]):
        state[i] = encode_task(task, current_time, completed_ids)
    return state.flatten()


def compute_reward(task, finish_time, prev_order, completed_ids):
    """Compute reward for completing a task."""
    reward = 0.0
    info = {}
    if finish_time <= task['deadline_min']:
        if task.get('vague', False):
            reward += R_VAGUE
            info['on_time'] = True
        else:
            reward += R_ON_TIME
            info['on_time'] = True
    else:
        reward += R_MISSED
        info['on_time'] = False
    for dep_id in task.get('dependencies', []):
        if dep_id in completed_ids:
            reward += R_DEP_RESOLVED
    if prev_order and len(prev_order) > 0 and task['id'] != prev_order[0]:
        reward += R_REORDER
    info['reward'] = reward
    return reward, info


def run_dqn_scheduling(model, task_queue):
    """Run the trained DQN model on a task queue and return scheduled results."""
    queue = copy.deepcopy(task_queue)
    current_time = 0
    completed = []
    missed = []
    completed_ids = set()
    prev_order = [t['id'] for t in queue]
    steps = 0

    while queue and current_time < WORKDAY_MINUTES and steps < 200:
        steps += 1
        state = encode_state(queue, current_time, completed_ids)

        # Get action from model
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_t).squeeze(0)

            # Mask invalid actions
            mask = torch.zeros(MAX_TASKS, dtype=torch.bool)
            for i in range(min(len(queue), MAX_TASKS)):
                mask[i] = True
            q_values[~mask] = float('-inf')

            action = int(q_values.argmax().item())

        action = min(action, len(queue) - 1)
        task = queue.pop(action)

        # Dependency check
        if task.get('dependencies'):
            unmet = [d for d in task['dependencies'] if d not in completed_ids]
            if unmet:
                queue.append(task)
                current_time += 10
                continue

        finish_time = current_time + task['duration_min']
        if finish_time > WORKDAY_MINUTES:
            missed.append(task)
            continue

        current_time = finish_time
        on_time = finish_time <= task['deadline_min']
        completed.append({
            **task,
            'start_min': finish_time - task['duration_min'],
            'finish_time': finish_time,
            'on_time': on_time
        })
        completed_ids.add(task['id'])
        prev_order = [t['id'] for t in queue]

    # Remaining tasks are missed
    for t in queue:
        missed.append(t)

    return completed, missed


def min_to_time(m):
    """Convert minutes from workday start to HH:MM format."""
    h = m // 60 + 8
    mins = m % 60
    return f"{h:02d}:{mins:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_training_results():
    return pd.read_csv(TRAIN_RESULTS_PATH)

@st.cache_data
def load_dqn_results():
    return pd.read_csv(DQN_RESULTS_PATH)

@st.cache_data
def load_baseline_results():
    return pd.read_csv(BASELINE_RESULTS_PATH)

@st.cache_data
def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_test_scenarios():
    with open(TEST_SCENARIOS_PATH) as f:
        return json.load(f)

@st.cache_resource
def load_model():
    model = DQNetwork()
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['online_net'])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# NLP PARSING (REAL MISTRAL 7B VIA OLLAMA)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a task extraction assistant. Extract tasks from natural language input.
Return ONLY a valid JSON array with no preamble or explanation.

Each task object must have these fields:
- "id": sequential task ID (T001, T002, etc.)
- "name": short task name (max 8 words)
- "deadline": ISO 8601 timestamp (working day is 2025-01-06, 08:00 to 18:00)
- "deadline_min": minutes from 08:00 (0 to 600)
- "duration_min": estimated duration in minutes
- "priority": 1 (low) to 5 (critical)
- "dependencies": list of task IDs that must complete first
- "status": always "pending"
- "vague": true if deadline or duration was guessed

Priority mapping:
- "urgent", "critical", "ASAP" = 5
- "important" = 4
- no signal = 3
- "when you have time", "low priority" = 2
- "if possible", "optional" = 1

If no deadline is mentioned, default to deadline_min: 540 (17:00).
If no duration is mentioned, default to 60 minutes.
"""

FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Submit the budget report by 2pm"},
    {"role": "assistant", "content": '[{"id":"T001","name":"Submit budget report","deadline":"2025-01-06T14:00:00","deadline_min":360,"duration_min":60,"priority":3,"dependencies":[],"status":"pending","vague":false}]'},
    {"role": "user", "content": "Urgently fix the login bug before noon, then write the incident report"},
    {"role": "assistant", "content": '[{"id":"T001","name":"Fix login bug","deadline":"2025-01-06T12:00:00","deadline_min":240,"duration_min":60,"priority":5,"dependencies":[],"status":"pending","vague":false},{"id":"T002","name":"Write incident report","deadline":"2025-01-06T17:00:00","deadline_min":540,"duration_min":60,"priority":4,"dependencies":["T001"],"status":"pending","vague":true}]'},
]


def parse_with_ollama(user_input):
    """Send task description to Mistral 7B via Ollama and parse the response."""
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({"role": "user", "content": user_input})

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "mistral", "messages": messages, "stream": False},
            timeout=60
        )

        if response.status_code != 200:
            return None, f"Ollama returned status {response.status_code}"

        result = response.json()
        content = result.get("message", {}).get("content", "")

        # Clean and parse JSON
        import re
        content = content.strip()
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)

        # Find JSON array
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            tasks = json.loads(match.group())
            # Validate and clamp fields
            for task in tasks:
                task['priority'] = max(1, min(5, task.get('priority', 3)))
                task['duration_min'] = max(5, min(600, task.get('duration_min', 60)))
                task['deadline_min'] = max(0, min(600, task.get('deadline_min', 540)))
                task.setdefault('dependencies', [])
                task.setdefault('status', 'pending')
                task.setdefault('vague', False)
            return tasks, None
        else:
            return None, "Could not find JSON array in model response"

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."
    except Exception as e:
        return None, str(e)


def check_ollama_status():
    """Check if Ollama is running and Mistral model is available."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            has_mistral = any('mistral' in m.lower() for m in models)
            return True, has_mistral, models
        return False, False, []
    except:
        return False, False, []


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("NeuroPlanner")
st.sidebar.caption("Autonomous Task Planning System")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Training", "Compare", "Live Demo"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**MSc Data Science**")
st.sidebar.markdown("Arden University Berlin")
st.sidebar.markdown("Zain Ali (STU227908)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    st.title("NeuroPlanner: Autonomous Task Planner")
    st.markdown("LLM-Powered Dynamic Scheduling with Deep Reinforcement Learning")

    # Load real data
    try:
        dqn_results = load_dqn_results()
        baseline_results = load_baseline_results()
        training_results = load_training_results()

        fifo = baseline_results[baseline_results['scheduler'] == 'FIFO']
        pq = baseline_results[baseline_results['scheduler'] == 'Priority Queue']

        dqn_tcr = dqn_results['tcr'].mean() * 100
        fifo_tcr = fifo['tcr'].mean() * 100
        pq_tcr = pq['tcr'].mean() * 100

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("DQN Test TCR", f"{dqn_tcr:.2f}%", f"+{dqn_tcr - fifo_tcr:.2f}pp vs FIFO")
        col2.metric("FIFO Baseline", f"{fifo_tcr:.2f}%")
        col3.metric("Training Episodes", f"{len(training_results):,}")
        col4.metric("Dataset Scenarios", "1,000", "800 train, 200 test")

        st.markdown("---")

        # Research question
        st.subheader("Research Question")
        st.info(
            "To what extent does LLM-based dynamic task re-prioritisation outperform "
            "static scheduling algorithms in multi-constraint personal productivity "
            "environments, as measured by task completion rate, deadline adherence, "
            "and schedule recovery time?"
        )

        # System architecture
        st.subheader("System Architecture")
        arch_cols = st.columns(5)
        arch_items = [
            ("User Input", "Natural language"),
            ("NLP Pipeline", "Mistral 7B via Ollama"),
            ("Task Queue", "Structured JSON"),
            ("DQN Agent", "140-dim state, 20 actions"),
            ("Schedule", "Optimised ordering"),
        ]
        for col, (name, sub) in zip(arch_cols, arch_items):
            col.markdown(f"**{name}**")
            col.caption(sub)

        st.markdown("---")

        # Key findings
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Key Findings")
            st.markdown(f"- DQN outperforms both baselines on 200 test scenarios ({dqn_tcr:.2f}% vs {fifo_tcr:.2f}% FIFO)")
            st.markdown(f"- Agent learns from ~12% TCR (episode 1) to ~45% TCR (episode 5,000)")
            st.markdown(f"- Strongest improvement on high-complexity scenarios: +11.78pp over FIFO")
            st.markdown(f"- NLP pipeline achieves 95% parse success rate on 100 test cases")

        with col_right:
            st.subheader("MDP Formulation")
            st.markdown("**State Space:** 140-dimensional vector (20 tasks x 7 features)")
            st.markdown("**Action Space:** Discrete {0...19}, select next task to execute")
            st.markdown("**Reward:** +10 on-time, -15 missed, -2 reorder, +5 dep. resolved")
            st.markdown("**Discount Factor:** 0.95")

    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Please ensure all data files are in the '{DATA_DIR}/' folder.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: TRAINING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Training":
    st.title("DQN Training Results")
    st.markdown("5,000 episodes on 800 training scenarios")

    try:
        tr = load_training_results()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Episodes", f"{len(tr):,}")
        col2.metric("Total Gradient Steps", f"{tr['loss'].notna().sum():,}")
        col3.metric("Final Epsilon", f"{tr['epsilon'].iloc[-1]:.3f}")
        col4.metric("Final 100-ep TCR", f"{tr['tcr'].tail(100).mean()*100:.1f}%")

        st.markdown("---")

        # Training curve
        st.subheader("Task Completion Rate Over Training")
        window = st.slider("Smoothing window (episodes)", 10, 200, 100, step=10)

        tr['tcr_smooth'] = tr['tcr'].rolling(window=window, min_periods=1).mean() * 100

        fig_tcr = go.Figure()
        fig_tcr.add_trace(go.Scatter(
            x=tr['episode'], y=tr['tcr'] * 100,
            mode='markers', marker=dict(size=1, color='rgba(29,158,117,0.15)'),
            name='Per-episode TCR'
        ))
        fig_tcr.add_trace(go.Scatter(
            x=tr['episode'], y=tr['tcr_smooth'],
            mode='lines', line=dict(color='#1D9E75', width=2),
            name=f'{window}-episode rolling mean'
        ))
        fig_tcr.add_hline(y=38.5, line_dash="dash", line_color="red",
                          annotation_text="Baseline (38.5%)")
        fig_tcr.update_layout(
            xaxis_title="Episode", yaxis_title="TCR (%)",
            height=450, template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_tcr, use_container_width=True)

        # Reward and loss
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Episode Reward")
            tr['reward_smooth'] = tr['reward'].rolling(window=window, min_periods=1).mean()
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=tr['episode'], y=tr['reward_smooth'],
                mode='lines', line=dict(color='#534AB7', width=2),
                name='Smoothed reward'
            ))
            fig_reward.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_reward.update_layout(
                xaxis_title="Episode", yaxis_title="Total Reward",
                height=350, template="plotly_white"
            )
            st.plotly_chart(fig_reward, use_container_width=True)

        with col_right:
            st.subheader("Training Loss (Huber)")
            loss_data = tr[tr['loss'] > 0].copy()
            loss_data['loss_smooth'] = loss_data['loss'].rolling(window=100, min_periods=1).mean()
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=loss_data['episode'], y=loss_data['loss_smooth'],
                mode='lines', line=dict(color='#e09a3a', width=2),
                name='Smoothed loss'
            ))
            fig_loss.update_layout(
                xaxis_title="Episode", yaxis_title="Huber Loss",
                height=350, template="plotly_white"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        # Epsilon decay
        st.subheader("Epsilon Decay")
        fig_eps = go.Figure()
        fig_eps.add_trace(go.Scatter(
            x=tr['episode'], y=tr['epsilon'],
            mode='lines', line=dict(color='#e05c4a', width=2)
        ))
        fig_eps.update_layout(
            xaxis_title="Episode", yaxis_title="Epsilon",
            height=300, template="plotly_white"
        )
        st.plotly_chart(fig_eps, use_container_width=True)

        # Network architecture
        st.subheader("Network Architecture")
        st.markdown("**Input (140)** > Dense (256, ReLU) > Dense (256, ReLU) > Dense (128, ReLU) > **Output (20)**")
        st.markdown("Total parameters: 137,364 | Optimizer: Adam (lr=0.001) | Loss: Huber (smooth L1)")

    except FileNotFoundError as e:
        st.error(f"Training results file not found: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: COMPARE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Compare":
    st.title("Scheduler Comparison")
    st.markdown("200 held-out test scenarios, same evaluation protocol for all schedulers")

    try:
        dqn = load_dqn_results()
        bl = load_baseline_results()
        fifo = bl[bl['scheduler'] == 'FIFO'].copy()
        pq = bl[bl['scheduler'] == 'Priority Queue'].copy()

        # Overall TCR
        dqn_tcr = dqn['tcr'].mean() * 100
        fifo_tcr = fifo['tcr'].mean() * 100
        pq_tcr = pq['tcr'].mean() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("DQN Agent", f"{dqn_tcr:.2f}%", f"+{dqn_tcr - fifo_tcr:.2f}pp vs FIFO")
        col2.metric("FIFO Baseline", f"{fifo_tcr:.2f}%")
        col3.metric("Priority Queue", f"{pq_tcr:.2f}%")

        st.markdown("---")

        # TCR by complexity
        st.subheader("TCR by Complexity Level")

        complexity_data = []
        for comp in ['low', 'medium', 'high']:
            dqn_c = dqn[dqn['complexity'] == comp]['tcr'].mean() * 100
            fifo_c = fifo[fifo['complexity'] == comp]['tcr'].mean() * 100
            pq_c = pq[pq['complexity'] == comp]['tcr'].mean() * 100
            complexity_data.append({'Complexity': comp.capitalize(), 'DQN': dqn_c, 'FIFO': fifo_c, 'Priority Queue': pq_c})

        comp_df = pd.DataFrame(complexity_data)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name='DQN Agent', x=comp_df['Complexity'],
                                  y=comp_df['DQN'], marker_color='#1D9E75'))
        fig_comp.add_trace(go.Bar(name='FIFO', x=comp_df['Complexity'],
                                  y=comp_df['FIFO'], marker_color='#e05c4a'))
        fig_comp.add_trace(go.Bar(name='Priority Queue', x=comp_df['Complexity'],
                                  y=comp_df['Priority Queue'], marker_color='#e09a3a'))
        fig_comp.update_layout(
            barmode='group', yaxis_title="TCR (%)",
            height=400, template="plotly_white"
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # Statistical significance
        st.subheader("Statistical Significance Testing")

        # FIFO vs PQ
        merged = fifo.merge(pq, on='scenario_id', suffixes=('_fifo', '_pq'))
        t_fifo_pq_tcr, p_fifo_pq_tcr = stats.ttest_rel(merged['tcr_fifo'], merged['tcr_pq'])
        t_fifo_pq_dd, p_fifo_pq_dd = stats.ttest_rel(merged['dd_fifo'], merged['dd_pq'])

        # DQN vs baselines (one-sample against baseline means)
        t_dqn_fifo, p_dqn_fifo = stats.ttest_1samp(dqn['tcr'], fifo['tcr'].mean())
        t_dqn_pq, p_dqn_pq = stats.ttest_1samp(dqn['tcr'], pq['tcr'].mean())

        sig_data = pd.DataFrame([
            {"Comparison": "FIFO vs PQ (TCR)", "Mean Diff": f"+{(fifo_tcr-pq_tcr):.2f} pp",
             "t-statistic": f"{t_fifo_pq_tcr:.3f}", "p-value": f"{p_fifo_pq_tcr:.4f}",
             "Significant": "No" if p_fifo_pq_tcr > 0.05 else "Yes"},
            {"Comparison": "FIFO vs PQ (DD)", "Mean Diff": f"{(fifo['dd'].mean()-pq['dd'].mean()):.2f} min",
             "t-statistic": f"{t_fifo_pq_dd:.3f}", "p-value": f"{p_fifo_pq_dd:.4f}",
             "Significant": "No" if p_fifo_pq_dd > 0.05 else "Yes"},
            {"Comparison": "DQN vs FIFO (TCR)", "Mean Diff": f"+{(dqn_tcr-fifo_tcr):.2f} pp",
             "t-statistic": f"{t_dqn_fifo:.3f}", "p-value": f"{p_dqn_fifo:.4f}",
             "Significant": "No" if p_dqn_fifo > 0.05 else "Yes"},
            {"Comparison": "DQN vs PQ (TCR)", "Mean Diff": f"+{(dqn_tcr-pq_tcr):.2f} pp",
             "t-statistic": f"{t_dqn_pq:.3f}", "p-value": f"{p_dqn_pq:.4f}",
             "Significant": "No" if p_dqn_pq > 0.05 else "Yes"},
        ])
        st.dataframe(sig_data, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Detailed results table
        st.subheader("Detailed Results")
        detail_data = pd.DataFrame([
            {"Metric": "TCR (overall)", "FIFO": f"{fifo_tcr:.2f}%", "Priority Queue": f"{pq_tcr:.2f}%", "DQN Agent": f"{dqn_tcr:.2f}%"},
            {"Metric": "TCR (low)", "FIFO": f"{fifo[fifo['complexity']=='low']['tcr'].mean()*100:.2f}%",
             "Priority Queue": f"{pq[pq['complexity']=='low']['tcr'].mean()*100:.2f}%",
             "DQN Agent": f"{dqn[dqn['complexity']=='low']['tcr'].mean()*100:.2f}%"},
            {"Metric": "TCR (medium)", "FIFO": f"{fifo[fifo['complexity']=='medium']['tcr'].mean()*100:.2f}%",
             "Priority Queue": f"{pq[pq['complexity']=='medium']['tcr'].mean()*100:.2f}%",
             "DQN Agent": f"{dqn[dqn['complexity']=='medium']['tcr'].mean()*100:.2f}%"},
            {"Metric": "TCR (high)", "FIFO": f"{fifo[fifo['complexity']=='high']['tcr'].mean()*100:.2f}%",
             "Priority Queue": f"{pq[pq['complexity']=='high']['tcr'].mean()*100:.2f}%",
             "DQN Agent": f"{dqn[dqn['complexity']=='high']['tcr'].mean()*100:.2f}%"},
            {"Metric": "Mean DD (min)", "FIFO": f"{fifo['dd'].mean():.2f}",
             "Priority Queue": f"{pq['dd'].mean():.2f}",
             "DQN Agent": f"{dqn['dd'].mean():.2f}"},
            {"Metric": "Total on-time", "FIFO": f"{fifo['on_time'].sum()}",
             "Priority Queue": f"{pq['on_time'].sum()}",
             "DQN Agent": f"{dqn['completed'].sum()}"},
        ])
        st.dataframe(detail_data, use_container_width=True, hide_index=True)

    except FileNotFoundError as e:
        st.error(f"Results file not found: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Live Demo":
    st.title("Live Scheduler Demo")
    st.markdown("Add tasks in natural language, then run the real DQN agent")

    # Initialize session state
    if 'demo_tasks' not in st.session_state:
        st.session_state.demo_tasks = [
            {'id': 'T001', 'name': 'Write quarterly report', 'deadline_min': 240,
             'duration_min': 120, 'priority': 4, 'dependencies': [], 'status': 'pending', 'vague': False},
            {'id': 'T002', 'name': 'Attend team meeting', 'deadline_min': 120,
             'duration_min': 60, 'priority': 3, 'dependencies': [], 'status': 'pending', 'vague': False},
            {'id': 'T003', 'name': 'Fix critical API bug', 'deadline_min': 180,
             'duration_min': 90, 'priority': 5, 'dependencies': [], 'status': 'pending', 'vague': False},
            {'id': 'T004', 'name': 'Review pull request', 'deadline_min': 480,
             'duration_min': 30, 'priority': 2, 'dependencies': [], 'status': 'pending', 'vague': False},
        ]
    if 'schedule_results' not in st.session_state:
        st.session_state.schedule_results = None

    col_input, col_queue = st.columns([1, 1])

    with col_input:
        st.subheader("NLP Task Parser")

        # Check Ollama status
        ollama_running, has_mistral, models = check_ollama_status()

        if ollama_running and has_mistral:
            st.success("Ollama is running with Mistral 7B")
            parser_mode = "ollama"
        else:
            if not ollama_running:
                st.warning("Ollama is not running. Using keyword-based parsing. Start Ollama for real NLP parsing.")
            else:
                st.warning(f"Mistral model not found. Available models: {models}. Run 'ollama pull mistral' to install.")
            parser_mode = "keyword"

        user_input = st.text_area(
            "Describe your task in natural language",
            placeholder='e.g. "Urgently fix the login bug before noon"',
            height=100
        )

        if st.button("Parse Task", type="primary", use_container_width=True):
            if user_input.strip():
                with st.spinner("Parsing with " + ("Mistral 7B" if parser_mode == "ollama" else "keyword parser") + "..."):
                    if parser_mode == "ollama":
                        tasks, error = parse_with_ollama(user_input)
                        if error:
                            st.error(f"Parsing error: {error}")
                        elif tasks:
                            for task in tasks:
                                task['id'] = f"T{len(st.session_state.demo_tasks)+1:03d}"
                                st.session_state.demo_tasks.append(task)
                            st.success(f"Parsed {len(tasks)} task(s) with Mistral 7B")
                            st.session_state.schedule_results = None
                    else:
                        # Keyword-based fallback
                        text = user_input.lower()
                        priority = 5 if ('urgent' in text or 'critical' in text) else \
                                   4 if 'important' in text else \
                                   2 if ('when you have time' in text or 'if possible' in text) else 3

                        deadline_min = 60 if '9am' in text else 120 if '10am' in text else \
                                       180 if '11am' in text else 240 if ('noon' in text or '12' in text) else \
                                       300 if '1pm' in text else 360 if '2pm' in text else \
                                       420 if '3pm' in text else 480 if '4pm' in text else \
                                       540 if ('5pm' in text or 'end of day' in text or 'eod' in text) else 480

                        duration_min = 120 if '2 hour' in text else 60 if 'hour' in text else \
                                       30 if ('30 min' in text or 'half' in text) else \
                                       15 if ('15 min' in text or 'quick' in text) else 60

                        new_task = {
                            'id': f"T{len(st.session_state.demo_tasks)+1:03d}",
                            'name': user_input[:45],
                            'deadline_min': deadline_min,
                            'duration_min': duration_min,
                            'priority': priority,
                            'dependencies': [],
                            'status': 'pending',
                            'vague': deadline_min == 480 and '4pm' not in text,
                        }
                        st.session_state.demo_tasks.append(new_task)
                        st.success("Task parsed (keyword mode)")
                        st.session_state.schedule_results = None

        if st.button("Clear All Tasks", use_container_width=True):
            st.session_state.demo_tasks = []
            st.session_state.schedule_results = None
            st.rerun()

    with col_queue:
        st.subheader(f"Task Queue ({len(st.session_state.demo_tasks)} tasks)")

        for i, task in enumerate(st.session_state.demo_tasks):
            priority_colors = {1: '🟦', 2: '🟦', 3: '🟨', 4: '🟧', 5: '🟥'}
            priority_labels = {1: 'Low', 2: 'Low-Med', 3: 'Medium', 4: 'High', 5: 'Critical'}

            col_task, col_remove = st.columns([5, 1])
            with col_task:
                st.markdown(
                    f"{priority_colors.get(task['priority'], '🟨')} **{task['name']}** "
                    f"| Due: {min_to_time(task['deadline_min'])} "
                    f"| {task['duration_min']}min "
                    f"| P{task['priority']} {priority_labels.get(task['priority'], '')}"
                    f"{'  (vague)' if task.get('vague') else ''}"
                )
            with col_remove:
                if st.button("X", key=f"remove_{i}"):
                    st.session_state.demo_tasks.pop(i)
                    st.session_state.schedule_results = None
                    st.rerun()

    st.markdown("---")

    # Run DQN button
    if st.button("Run DQN Scheduling Agent", type="primary", use_container_width=True):
        if len(st.session_state.demo_tasks) == 0:
            st.warning("Add at least one task to the queue first.")
        else:
            with st.spinner("Running DQN agent..."):
                try:
                    model = load_model()
                    completed, missed = run_dqn_scheduling(model, st.session_state.demo_tasks)
                    st.session_state.schedule_results = {'completed': completed, 'missed': missed}
                except Exception as e:
                    st.error(f"Error running model: {e}")

    # Show results
    if st.session_state.schedule_results:
        results = st.session_state.schedule_results
        completed = results['completed']
        missed = results['missed']

        st.subheader("DQN Optimised Schedule")

        # Summary metrics
        total = len(completed) + len(missed)
        on_time = sum(1 for t in completed if t.get('on_time', False))
        late = sum(1 for t in completed if not t.get('on_time', False))
        tcr = (on_time / total * 100) if total > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Scheduled", len(completed))
        col2.metric("On Time", on_time)
        col3.metric("Late", late)
        col4.metric("Missed", len(missed))
        col5.metric("TCR", f"{tcr:.1f}%")

        # Gantt chart
        if completed:
            fig_gantt = go.Figure()

            for i, task in enumerate(completed):
                color = '#1D9E75' if task.get('on_time') else '#e05c4a'
                start_hour = task['start_min'] / 60 + 8
                end_hour = task['finish_time'] / 60 + 8
                deadline_hour = task['deadline_min'] / 60 + 8

                fig_gantt.add_trace(go.Bar(
                    y=[task['name']],
                    x=[task['duration_min'] / 60],
                    base=[start_hour],
                    orientation='h',
                    marker_color=color,
                    name=task['name'],
                    showlegend=False,
                    text=f"{'On time' if task.get('on_time') else 'Late'} ({task['duration_min']}min)",
                    textposition='inside',
                    hovertemplate=(
                        f"<b>{task['name']}</b><br>"
                        f"Start: {min_to_time(task['start_min'])}<br>"
                        f"End: {min_to_time(task['finish_time'])}<br>"
                        f"Deadline: {min_to_time(task['deadline_min'])}<br>"
                        f"Priority: {task['priority']}/5<br>"
                        f"Status: {'On time' if task.get('on_time') else 'Late'}"
                        "<extra></extra>"
                    )
                ))

                # Deadline marker
                fig_gantt.add_trace(go.Scatter(
                    x=[deadline_hour], y=[task['name']],
                    mode='markers', marker=dict(symbol='diamond', size=10, color='red'),
                    showlegend=False,
                    hovertemplate=f"Deadline: {min_to_time(task['deadline_min'])}<extra></extra>"
                ))

            fig_gantt.update_layout(
                xaxis_title="Time of Day",
                xaxis=dict(range=[8, 18], dtick=1, tickformat=".0f"),
                height=max(300, len(completed) * 50),
                template="plotly_white",
                bargap=0.3,
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

            # Legend
            st.markdown("🟩 On time | 🟥 Late | 🔴 Deadline marker")

        if missed:
            st.subheader("Missed Tasks")
            for t in missed:
                st.markdown(f"- {t['name']} (P{t['priority']}, deadline {min_to_time(t['deadline_min'])})")
