# NeuroPlanner Dashboard

Interactive Streamlit dashboard for the autonomous task planning system.
Loads real data, runs the trained DQN model, and connects to Mistral 7B via Ollama.

## Setup

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Organise data files

Place the following files in the `data/` folder:

```
data/
  dqn_model.pth           # Trained DQN model weights
  training_results.csv    # Training logs (5,000 episodes)
  dqn_test_results.csv    # DQN test evaluation (200 scenarios)
  baseline_results.csv    # FIFO and PQ results (400 rows)
  scenarios.json          # Full dataset (1,000 scenarios)
  test_scenarios.json     # Test split (200 scenarios)
```

### 3. Start Ollama (optional, for real NLP parsing)

```
ollama serve
ollama pull mistral
```

If Ollama is not running, the Live Demo page will use keyword-based parsing as a fallback.

### 4. Run the dashboard

```
streamlit run app.py
```

The dashboard will open in your browser at http://localhost:8501.

## Pages

- **Overview**: Key metrics, research question, system architecture, findings
- **Training**: Interactive training curve, reward trajectory, loss, epsilon decay
- **Compare**: TCR comparison by complexity, statistical significance testing, detailed results
- **Live Demo**: Real NLP parsing (Mistral 7B) and real DQN model inference with Gantt chart output
