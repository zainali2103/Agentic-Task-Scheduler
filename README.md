# NeuroPlanner: Autonomous Task Planner

> *"To what extent does LLM-based dynamic task re-prioritisation outperform static scheduling algorithms in multi-constraint personal productivity environments?"*

---

## Key Results

| Scheduler | Test TCR | vs FIFO |
|---|---|---|
| **DQN Agent** | **41.96%** | **+3.47 pp** |
| FIFO Baseline | 38.49% | - |
| Priority Queue | 37.79% | -0.70 pp |

**DQN advantage on high-complexity scenarios: +11.78 pp over FIFO (p < 0.001)**

---

## Project Structure

```
Agentic-Task-Scheduler/
|
├── Data_Generation/
│   ├── 01_Dataset_Generator.ipynb
│   ├── dataset_generator.py
│   ├── scenarios.json
│   ├── train_scenarios.json
│   ├── test_scenarios.json
│   └── dataset_summary.csv
|
├── NLP_Pipeline/
│   └── 02_NLP_Pipeline.ipynb
|
├── Baseline/
│   └── 03_Baselines.ipynb
|
├── RL_Env/
│   └── 04_RL_Environment_v2.ipynb
|
├── DQN_Training/
│   └── 05_DQN_Training.ipynb
|
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── data/
│       ├── dqn_model.pth
│       ├── training_results.csv
│       ├── dqn_test_results.csv
│       ├── baseline_results.csv
│       ├── scenarios.json
│       └── test_scenarios.json
|
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/zainali2103/Agentic-Task-Scheduler.git
cd Agentic-Task-Scheduler
```

### 2. Install dependencies

```bash
pip install -r dashboard/requirements.txt
```

### 3. Install and start Ollama (for NLP pipeline)

Download Ollama from https://ollama.com/download and install it.

Then pull the Mistral model:

```bash
ollama pull mistral
```

### 4. Open Jupyter

```bash
jupyter notebook
```

Navigate to the notebook folders and run the notebooks in order from 01 to 05.

### 5. Run the dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open in your browser at http://localhost:8501.

---

## Notebook Guide

| Notebook | Description | Run time | Requires Ollama? |
|---|---|---|---|
| 01_Dataset_Generator.ipynb | Generates 1,000 synthetic scheduling scenarios, splits into train/test, runs EDA | ~30 seconds | No |
| 02_NLP_Pipeline.ipynb | Implements NLP parsing pipeline using Mistral 7B, evaluates on 100 test cases | ~60 to 90 minutes | **Yes** |
| 03_Baselines.ipynb | Implements FIFO and priority queue schedulers, evaluates on 200 test scenarios | ~30 seconds | No |
| 04_RL_Environment.ipynb | Defines MDP, implements TaskSchedulerEnv, validates with random agent | ~5 minutes | No |
| 05_DQN_Training.ipynb | Trains DQN agent for 5,000 episodes, evaluates on test set | ~10 min (GPU) / ~2 hrs (CPU) | No |

Notebooks 01, 03, 04, and 05 can be run without Ollama. Only notebook 02 requires the Mistral model.

---

## System Architecture

```
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
        |  actions:  select next task to execute (discrete 0 to 19)
        |  reward:   +10 on-time, -15 missed, -2 reorder, +5 dep. resolved
        v
Optimised Schedule
```

---

## DQN Architecture

```
Input (140) > Dense(256, ReLU) > Dense(256, ReLU) > Dense(128, ReLU) > Output(20)
```

| Hyperparameter | Value |
|---|---|
| Training episodes | 5,000 |
| Replay buffer | 10,000 transitions |
| Batch size | 64 |
| Learning rate | 0.001 (Adam) |
| Epsilon decay | 1.0 to 0.05 over 50,000 steps |
| Target network update | Every 100 episodes |
| Discount factor | 0.95 |
| Algorithm | Double DQN + Huber loss |

---

## Dataset

The synthetic dataset was generated procedurally using 01_Dataset_Generator.ipynb.

| Property | Value |
|---|---|
| Total scenarios | 1,000 |
| Training split | 800 (80%) |
| Test split | 200 (20%) |
| Complexity levels | Low (5 to 8 tasks), Medium (9 to 14), High (15 to 20) |
| Workday duration | 600 minutes (08:00 to 18:00) |
| Infeasibility rate | 80.5% of scenarios |
| Disruption types | Urgent insertion (30%), Deadline shift (40%), Cancellation (30%) |
| Random seed | 42 (fully reproducible) |

---

## Dashboard

The interactive Streamlit dashboard loads real data, runs the trained DQN model, and connects to Mistral 7B via Ollama for live NLP parsing.

**To run:**

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

**Pages:**

- **Overview** : key metrics computed from real evaluation data
- **Training** : interactive training curve from 5,000 logged episodes
- **Compare** : FIFO vs Priority Queue vs DQN with live statistical testing
- **Live Demo** : real NLP parsing (Mistral 7B) and real DQN model inference with Gantt chart output

---

## Requirements

```
Python 3.11+
torch>=2.0
numpy
pandas
matplotlib
seaborn
scipy
streamlit
plotly
requests
jupyter
ollama
```

---

## Reproducing the Results

1. Run 01_Dataset_Generator.ipynb with RANDOM_SEED = 42 to regenerate the identical dataset
2. Run 03_Baselines.ipynb to reproduce the FIFO and priority queue results (TCR: 38.49% and 37.79%)
3. Run 05_DQN_Training.ipynb with SEED = 42 to retrain the DQN agent
4. The test evaluation at the end of notebook 05 should produce TCR of approximately 41.96%

Exact DQN results may vary by 1 to 2 pp depending on hardware and PyTorch version due to floating-point non-determinism in GPU operations. The baseline results are fully deterministic.

---

## Licence

This project is released for academic and research purposes.
The synthetic dataset is released as an open artefact under the MIT Licence.
