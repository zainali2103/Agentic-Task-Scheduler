# Autonomous Task Planning with LLM-Powered Dynamic Scheduling

> *"To what extent does LLM-based dynamic task re-prioritisation outperform static scheduling algorithms in multi-constraint personal productivity environments?"*

---

## Key Results

| Scheduler | Test TCR | vs FIFO | vs EDD |
|---|---|---|---|
| **DQN Agent (single seed)** | **41.96%** | **+3.47 pp** | **+1.52 pp** |
| **DQN Agent (5-seed mean)** | **42.26%** | **+3.77 pp** | **+1.82 pp** |
| EDD Baseline | 40.44% | +1.95 pp | - |
| FIFO Baseline | 38.49% | - | -1.95 pp |
| Priority Queue | 37.79% | -0.70 pp | -2.65 pp |
| Oracle (theoretical ceiling) | 48.40% | +9.91 pp | +7.96 pp |

- **DQN advantage on high-complexity scenarios: +11.78 pp over FIFO (p < 0.001)**
- **5-seed mean TCR: 42.26% (SD 3.09%, 95% CI [38.42%, 46.09%])**
- **DQN achieves 87.3% of the oracle ceiling**

---

## System Architecture

```
                    User Input (natural language)
                              |
                              v
                +---------------------------------+
                |     NLP Parsing Pipeline         |
                |   (Mistral 7B via Ollama)        |
                |   Extracts: name, deadline,      |
                |   duration, priority, deps       |
                +---------------------------------+
                              |
                              v
                +---------------------------------+
                |     Task Queue (JSON)            |
                +---------------------------------+
                     /        |        \
                    v         v         v
            +--------+  +--------+  +--------+
            |  FIFO  |  |Priority|  |  EDD   |
            |        |  | Queue  |  |Earliest|
            |        |  |0.7d+   |  |Due Date|
            |        |  |0.3p    |  |        |
            +--------+  +--------+  +--------+
                     \        |        /
                      v       v       v
                +---------------------------------+
                |  RL Scheduling Environment       |
                |  MDP: 140-dim state, 20 actions  |
                |  Reward: +10 on-time, -15 missed |
                +---------------------------------+
                              |
                              v
                +---------------------------------+
                |     DQN Scheduling Agent          |
                |  Double DQN + Huber Loss          |
                |  5-seed training for robustness   |
                +---------------------------------+
                              |
                              v
                +---------------------------------+
                |   Comparative Evaluation          |
                |   200 test scenarios              |
                |   TCR / DD / SRT metrics          |
                |   Paired t-tests, Wilcoxon,       |
                |   Cohen's d, Shapiro-Wilk         |
                +---------------------------------+
                              |
                              v
                +---------------------------------+
                |   Streamlit Dashboard             |
                |   4 pages: Overview, Training,    |
                |   Compare, Live Demo              |
                +---------------------------------+
```

---

## DQN Architecture

```
Input (140) --> Dense(256, ReLU) --> Dense(256, ReLU) --> Dense(128, ReLU) --> Output(20)
```

**State encoding:** 20 task slots x 7 features = 140-dimensional observation vector

| Feature | Description |
|---|---|
| Normalised deadline | Minutes until deadline / 600 |
| Normalised duration | Task duration / max duration |
| Priority level | 1 (low) to 4 (critical), normalised |
| Dependency status | 1 if all dependencies resolved, 0 otherwise |
| Time pressure | Duration / time remaining until deadline |
| Is active | 1 if slot contains a task, 0 if empty |
| Slack ratio | (Deadline - current time - duration) / 600 |

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Training episodes | 5,000 |
| Replay buffer | 10,000 transitions |
| Batch size | 64 |
| Learning rate | 0.001 (Adam) |
| Epsilon decay | 1.0 to 0.05 over 50,000 steps |
| Target network update | Every 100 episodes |
| Discount factor (gamma) | 0.95 |
| Algorithm | Double DQN + Huber loss |
| Training seeds | 42, 7, 17, 31, 53 |

**Reward function:**

| Component | Value | Rationale |
|---|---|---|
| On-time completion | +10 | Primary objective |
| Missed deadline | -15 | 1.5x asymmetric penalty |
| Dependency resolution | +5 | Encourages unblocking |
| Unnecessary reorder | -2 | Discourages thrashing |

---

## Baselines

### FIFO (First-In-First-Out)
Executes tasks in arrival order. Provides a lower bound representing zero scheduling intelligence.

### Priority Queue
Weighted static sort: 0.7 x deadline urgency + 0.3 x priority level. Represents a rule-based heuristic combining two scheduling criteria.

### Earliest Due Date (EDD)
Sorts all available tasks by deadline in ascending order and always executes the task with the nearest deadline first. Re-sorts after each disruption event. O(n log n) complexity. A well-established operations research heuristic that minimises maximum lateness in single-machine scheduling problems.

### Oracle Scheduler
Processes tasks with full knowledge of all future disruptions from the start. Establishes the theoretical TCR ceiling (48.40%) for the benchmark dataset, confirming the 75% target was structurally unreachable.

---

## NLP Pipeline

The NLP parsing component uses **Mistral 7B** (via Ollama) to extract structured task attributes from free-text input.

**Example input:**
```
Finish quarterly report by 2pm, high priority. Call mom at 3pm. 
Team standup at 9:30am, 30 minutes. Submit expenses before end of day.
```

**Extracted fields:** task name, deadline, estimated duration, priority level (1-4), dependencies

**Evaluation (100 annotated test cases):**

| Metric | Score |
|---|---|
| Field-level accuracy | 92.8% |
| Exact match rate | 66% |
| Partial match rate | 34% |

---

## Dataset

The synthetic dataset is generated procedurally with controlled complexity levels and stochastic disruptions.

| Property | Value |
|---|---|
| Total scenarios | 1,000 |
| Training split | 800 (80%) |
| Test split | 200 (20%) |
| Complexity levels | Low (5-8 tasks), Medium (9-14), High (15-20) |
| Workday duration | 600 minutes (08:00 to 18:00) |
| Infeasibility rate | 80.5% of scenarios |
| Disruption types | Urgent insertion (30%), Deadline shift (40%), Cancellation (30%) |
| Random seed | 42 (fully reproducible) |

---

## Statistical Validation

All results are validated with multi-seed training (5 seeds) and rigorous statistical testing:

| Test | DQN vs FIFO | DQN vs PQ | DQN vs EDD |
|---|---|---|---|
| Paired t-test | p < 0.001 | p < 0.001 | p = 0.028 |
| Wilcoxon signed-rank | p < 0.001 | p < 0.001 | p = 0.031 |
| Cohen's d (effect size) | 0.31 (small-medium) | 0.38 (small-medium) | 0.13 (small) |
| Shapiro-Wilk normality | p = 0.41 (normal) | p = 0.38 (normal) | p = 0.44 (normal) |

**5-seed training results:**

| Seed | TCR |
|---|---|
| 42 | 41.96% |
| 7 | 45.21% |
| 17 | 38.89% |
| 31 | 44.12% |
| 53 | 41.13% |
| **Mean (SD)** | **42.26% (3.09%)** |
| **95% CI** | **[38.42%, 46.09%]** |

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
| 03_Baselines.ipynb | Implements FIFO, Priority Queue, and EDD schedulers, evaluates on 200 test scenarios | ~30 seconds | No |
| 04_RL_Environment.ipynb | Defines MDP, implements TaskSchedulerEnv, validates with random agent | ~5 minutes | No |
| 05_DQN_Training.ipynb | Trains DQN agent (5 seeds x 5,000 episodes), evaluates on test set with statistical testing | ~50 min (GPU) / ~10 hrs (CPU) | No |

Notebooks 01, 03, 04, and 05 can be run without Ollama. Only notebook 02 requires the Mistral model.

---

## Dashboard

The interactive Streamlit dashboard loads real evaluation data, runs the trained DQN model live, and connects to Mistral 7B via Ollama for NLP parsing.

**To run:**

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

**Pages:**

- **Overview**: key metrics (TCR, deadline deviation, SRT), oracle ceiling comparison, and summary statistics computed from real evaluation data
- **Training**: interactive training curve from 5,000 logged episodes with moving average, epsilon decay overlay, and convergence analysis
- **Compare**: FIFO vs Priority Queue vs EDD vs DQN with grouped bar charts by complexity level, live paired t-tests, Wilcoxon signed-rank tests, and Cohen's d effect sizes computed via scipy
- **Live Demo**: real-time NLP parsing (Mistral 7B) of user-typed tasks, DQN model inference, and interactive Gantt chart output showing the optimised schedule

---

## Reproducing the Results

1. Run `01_Dataset_Generator.ipynb` with `RANDOM_SEED = 42` to regenerate the identical dataset
2. Run `03_Baselines.ipynb` to reproduce the FIFO, Priority Queue, and EDD results (TCR: 38.49%, 37.79%, and 40.44%)
3. Run `05_DQN_Training.ipynb` with seeds `[42, 7, 17, 31, 53]` to retrain the DQN agent across all 5 seeds
4. The test evaluation should produce a 5-seed mean TCR of approximately 42.26%

Exact DQN results may vary by 1 to 2 pp depending on hardware and PyTorch version due to floating-point non-determinism in GPU operations. The baseline results are fully deterministic.

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

## Licence

This project is released for academic and research purposes.
The synthetic dataset is released as an open artefact under the MIT Licence.
