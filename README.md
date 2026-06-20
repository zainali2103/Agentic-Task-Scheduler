# Autonomous Task Planner
**Author:** Zain Ali | **Year:** 2026

> *"To what extent does LLM-based dynamic task re-prioritisation outperform static scheduling algorithms in multi-constraint personal productivity environments?"*

---

## 🎯 Key Results

| Scheduler | Test TCR | vs FIFO |
|---|---|---|
| **DQN Agent** | **41.96%** | **+3.47 pp** |
| FIFO Baseline | 38.49% | — |
| Priority Queue | 37.79% | −0.70 pp |

**DQN advantage on high-complexity scenarios: +11.78 pp over FIFO**

---

## 📁 Project Structure

```
autonomous-task-planner-/
│
├── notebooks/
│   ├── 01_Dataset_Generator.ipynb   # Synthetic dataset generation + EDA
│   ├── 02_NLP_Pipeline.ipynb        # Mistral 7B task parsing pipeline
│   ├── 03_Baselines.ipynb           # FIFO + priority queue schedulers
│   ├── 04_RL_Environment.ipynb      # MDP formulation + gym environment
│   └── 05_DQN_Training.ipynb        # DQN agent training + evaluation
│
├── data/
│   ├── scenarios.json               # Full dataset (1,000 scenarios)
│   ├── train_scenarios.json         # Training split (800 scenarios)
│   ├── test_scenarios.json          # Test split (200 scenarios)
│   ├── dataset_summary.csv          # One row per scenario (for EDA)
│   └── baseline_results.csv         # FIFO + PQ evaluation results
│
├── dashboard/
│   └── dashboard.html               # Interactive web dashboard (no install needed)
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/autonomous-task-planner-msc.git
cd autonomous-task-planner-msc
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama (for NLP pipeline only)

Download Ollama from https://ollama.com/download and install it.

Then pull the Mistral model:

```bash
ollama pull mistral
```

Start Ollama (it runs as a background service on Windows — check your system tray).

### 4. Open Jupyter

```bash
jupyter notebook
```

Navigate to the `notebooks/` folder and run the notebooks **in order** from 01 to 05.

### 5. Open the dashboard

Simply double-click `dashboard/dashboard.html` — no server or install required.

---

## 📓 Notebook Guide

| Notebook | Description | Run time | Requires Ollama? |
|---|---|---|---|
| `01_Dataset_Generator.ipynb` | Generates 1,000 synthetic scheduling scenarios, splits into train/test, runs EDA | ~30 seconds | No |
| `02_NLP_Pipeline.ipynb` | Implements NLP parsing pipeline using Mistral 7B, evaluates on 100 test cases | ~60–90 minutes | **Yes** |
| `03_Baselines.ipynb` | Implements FIFO and priority queue schedulers, evaluates on 200 test scenarios | ~30 seconds | No |
| `04_RL_Environment.ipynb` | Defines MDP, implements TaskSchedulerEnv, validates with random agent | ~5 minutes | No |
| `05_DQN_Training.ipynb` | Trains DQN agent for 5,000 episodes, evaluates on test set | ~10 min (GPU) / ~2 hrs (CPU) | No |

> **Note:** Notebooks 01, 03, 04, and 05 can be run without Ollama. Only notebook 02 requires the Mistral model.

---

## 🧠 System Architecture

```
User Input (natural language)
        │
        ▼
NLP Pipeline (Mistral 7B via Ollama)
        │  extracts: name, deadline, duration, priority, dependencies
        ▼
Task Queue (structured JSON)
        │
        ▼
DQN Scheduling Agent
        │  observes: 140-dim state vector (20 tasks × 7 features)
        │  actions:  select next task to execute (discrete 0–19)
        │  reward:   +10 on-time, −15 missed, −2 reorder, +5 dep. resolved
        ▼
Optimised Schedule
```

---

## 🤖 DQN Architecture

```
Input (140)  →  Dense(256, ReLU)  →  Dense(256, ReLU)  →  Dense(128, ReLU)  →  Output(20)
```

| Hyperparameter | Value |
|---|---|
| Training episodes | 5,000 |
| Replay buffer | 10,000 transitions |
| Batch size | 64 |
| Learning rate | 0.001 (Adam) |
| Epsilon decay | 1.0 → 0.05 over 50,000 steps |
| Target network update | Every 100 episodes |
| Discount factor γ | 0.95 |
| Algorithm | Double DQN + Huber loss |

---

## 📊 Dataset

The synthetic dataset was generated procedurally using `01_Dataset_Generator.ipynb`.

| Property | Value |
|---|---|
| Total scenarios | 1,000 |
| Training split | 800 (80%) |
| Test split | 200 (20%) |
| Complexity levels | Low (5–8 tasks), Medium (9–14), High (15–20) |
| Workday duration | 600 minutes (08:00–18:00) |
| Infeasibility rate | 80.5% of scenarios |
| Disruption types | Urgent insertion (30%), Deadline shift (40%), Cancellation (30%) |
| Random seed | 42 (fully reproducible) |

---

## 🔧 Requirements

```
Python 3.11+
torch>=2.0
numpy
pandas
matplotlib
seaborn
scipy
nbformat
jupyter
ollama          # for NLP pipeline only
```

Install everything at once:

```bash
pip install -r requirements.txt
```

For the NLP pipeline specifically, also install Ollama and pull the Mistral model as described in the Quick Start section above.

---

## 📈 Reproducing the Results

To reproduce the exact results reported in the thesis:

1. Run `01_Dataset_Generator.ipynb` with `RANDOM_SEED = 42` — this regenerates the identical dataset
2. Run `03_Baselines.ipynb` — this reproduces the FIFO and priority queue results (TCR: 38.49% and 37.79%)
3. Run `05_DQN_Training.ipynb` with `SEED = 42` — this retrains the DQN agent (results may vary slightly due to GPU non-determinism)
4. The test evaluation at the end of notebook 05 should produce TCR ≈ 41.96%

> **Note:** Exact DQN results may vary by ±1–2 pp depending on hardware and PyTorch version due to floating-point non-determinism in GPU operations. The baseline results (notebooks 01 and 03) are fully deterministic.

---

## 🖥️ Dashboard

The interactive dashboard (`dashboard/dashboard.html`) requires no installation.

**To open:** double-click the file. Works in Chrome, Firefox, and Edge.

**Features:**
- Overview tab — key results, system architecture, MDP summary
- Training tab — interactive DQN learning curve over 5,000 episodes
- Compare tab — FIFO vs Priority Queue vs DQN across all complexity levels
- Live Demo tab — type tasks in natural language, run DQN, see Gantt chart

---

## 📚 Citation

If you use this dataset or code in your own research, please cite:

```
Ali, Z. (2025). Autonomous Task Planner: LLM-Powered Dynamic Scheduling
with Deep Reinforcement Learning. MSc Data Science Thesis,
Arden University Berlin Campus.
```

---

## 📄 Licence

This project is released for academic and research purposes.
The synthetic dataset is released as an open artefact under the MIT Licence.

---

