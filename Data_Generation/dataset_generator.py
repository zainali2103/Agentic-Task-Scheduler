"""
Synthetic Task Scheduling Dataset Generator
============================================
MSc Data Science Thesis — Zain Ali
Generates 1,000 task scenarios for training/evaluating the RL scheduling agent.

Output files:
  - scenarios.json       Full dataset (all 1,000 scenarios)
  - train_scenarios.json 800 scenarios (training split)
  - test_scenarios.json  200 scenarios (test split)
  - dataset_summary.csv  One row per scenario (for quick EDA)

Scenario structure:
  - 5–20 tasks per scenario
  - Each task has: id, name, deadline, duration, priority, dependencies
  - 1–3 disruption events per scenario
  - Simulated working day: 08:00–18:00 (600 min)
"""

import json
import random
import uuid
import csv
from datetime import datetime, timedelta
from copy import deepcopy

random.seed(42)

# ── Task name templates ──────────────────────────────────────────────────────

TASK_TEMPLATES = [
    "Write {doc} report", "Review {doc} document", "Prepare {doc} presentation",
    "Attend {meeting} meeting", "Send {doc} email", "Analyze {data} data",
    "Fix {issue} bug", "Update {doc} documentation", "Complete {doc} draft",
    "Research {topic}", "Schedule {meeting} call", "Finalize {doc} proposal",
    "Submit {doc} form", "Respond to {topic} feedback", "Plan {topic} strategy",
    "Code {feature} module", "Test {feature} feature", "Deploy {feature} update",
    "Review {topic} pull request", "Create {topic} dashboard",
]

DOC_WORDS    = ["quarterly", "project", "client", "budget", "technical", "annual", "summary"]
MEETING_WORDS= ["team", "client", "sprint", "board", "stakeholder", "1:1", "sync"]
DATA_WORDS   = ["sales", "user", "performance", "market", "financial", "traffic"]
ISSUE_WORDS  = ["critical", "login", "UI", "API", "database", "auth", "cache"]
TOPIC_WORDS  = ["product", "marketing", "security", "roadmap", "pricing", "onboarding"]
FEATURE_WORDS= ["search", "payment", "notification", "export", "analytics", "settings"]

def random_task_name():
    template = random.choice(TASK_TEMPLATES)
    return template.format(
        doc=random.choice(DOC_WORDS),
        meeting=random.choice(MEETING_WORDS),
        data=random.choice(DATA_WORDS),
        issue=random.choice(ISSUE_WORDS),
        topic=random.choice(TOPIC_WORDS),
        feature=random.choice(FEATURE_WORDS),
    )

# ── Complexity buckets ───────────────────────────────────────────────────────

COMPLEXITY_CONFIGS = {
    "low":    {"task_range": (5, 8),   "weight": 0.33},
    "medium": {"task_range": (9, 14),  "weight": 0.34},
    "high":   {"task_range": (15, 20), "weight": 0.33},
}

# ── Disruption types ─────────────────────────────────────────────────────────

DISRUPTION_TYPES = {
    "urgent_insertion": 0.30,   # A new high-priority task is added mid-day
    "deadline_shift":   0.40,   # An existing task's deadline moves ±2h
    "task_cancellation":0.30,   # An existing task is cancelled
}

# ── Core generators ──────────────────────────────────────────────────────────

def generate_task(task_index, total_tasks, workday_start_min, workday_duration_min):
    """Generate a single task with realistic attributes."""
    task_id   = f"T{task_index:03d}"
    name      = random_task_name()
    duration  = random.choice([15, 20, 30, 45, 60, 90, 120, 150, 180, 240])
    priority  = random.choices([1, 2, 3, 4, 5], weights=[0.10, 0.20, 0.35, 0.25, 0.10])[0]

    # Deadline: somewhere within the workday, at least duration minutes from start
    earliest_deadline_min = workday_start_min + duration
    latest_deadline_min   = workday_start_min + workday_duration_min
    deadline_offset_min   = random.randint(earliest_deadline_min, latest_deadline_min)

    base_date    = datetime(2025, 1, 6, 8, 0, 0)   # Monday 08:00
    deadline_dt  = base_date + timedelta(minutes=deadline_offset_min)
    deadline_iso = deadline_dt.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "id":           task_id,
        "name":         name,
        "deadline":     deadline_iso,
        "deadline_min": deadline_offset_min,          # minutes from workday start (internal)
        "duration_min": duration,
        "priority":     priority,
        "dependencies": [],                           # filled in later
        "status":       "pending",
    }


def add_dependencies(tasks):
    """Add realistic dependency edges (DAG — no cycles)."""
    n = len(tasks)
    if n < 3:
        return tasks
    # ~25% of tasks have one dependency on an earlier task
    for i in range(1, n):
        if random.random() < 0.25:
            dep_idx = random.randint(0, i - 1)
            tasks[i]["dependencies"].append(tasks[dep_idx]["id"])
    return tasks


def generate_disruptions(tasks, n_disruptions):
    """Generate disruption events for a scenario."""
    disruptions = []
    cancellable = [t for t in tasks if not t["dependencies"]]  # don't cancel depended-on tasks

    for _ in range(n_disruptions):
        dtype = random.choices(
            list(DISRUPTION_TYPES.keys()),
            weights=list(DISRUPTION_TYPES.values())
        )[0]

        if dtype == "urgent_insertion":
            base_date   = datetime(2025, 1, 6, 8, 0, 0)
            arrival_min = random.randint(60, 480)   # arrives 1–8h into the day
            deadline_min= arrival_min + random.randint(60, 180)
            deadline_dt = base_date + timedelta(minutes=deadline_min)
            disruptions.append({
                "type":         "urgent_insertion",
                "arrival_min":  arrival_min,
                "task": {
                    "id":           f"URGENT_{uuid.uuid4().hex[:6].upper()}",
                    "name":         f"URGENT: {random_task_name()}",
                    "deadline":     deadline_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    "deadline_min": deadline_min,
                    "duration_min": random.choice([15, 30, 45, 60]),
                    "priority":     5,
                    "dependencies": [],
                    "status":       "pending",
                }
            })

        elif dtype == "deadline_shift" and tasks:
            target = random.choice(tasks)
            shift  = random.choice([-120, -90, -60, 60, 90, 120])
            base_date    = datetime(2025, 1, 6, 8, 0, 0)
            new_dl_min   = max(target["duration_min"], target["deadline_min"] + shift)
            new_dl_min   = min(new_dl_min, 600)
            new_deadline = (base_date + timedelta(minutes=new_dl_min)).strftime("%Y-%m-%dT%H:%M:%S")
            disruptions.append({
                "type":          "deadline_shift",
                "target_task_id": target["id"],
                "shift_minutes":  shift,
                "new_deadline":   new_deadline,
                "new_deadline_min": new_dl_min,
            })

        elif dtype == "task_cancellation" and cancellable:
            target = random.choice(cancellable)
            disruptions.append({
                "type":           "task_cancellation",
                "target_task_id": target["id"],
            })

    return disruptions


def complexity_label(n_tasks):
    if n_tasks <= 8:
        return "low"
    elif n_tasks <= 14:
        return "medium"
    return "high"


def generate_scenario(scenario_id):
    """Generate one complete scenario."""
    # Pick complexity
    complexity  = random.choices(
        list(COMPLEXITY_CONFIGS.keys()),
        weights=[c["weight"] for c in COMPLEXITY_CONFIGS.values()]
    )[0]
    task_range  = COMPLEXITY_CONFIGS[complexity]["task_range"]
    n_tasks     = random.randint(*task_range)

    workday_start_min    = 0     # 08:00
    workday_duration_min = 600   # 10 hours

    # Generate tasks
    tasks = [generate_task(i, n_tasks, workday_start_min, workday_duration_min)
             for i in range(n_tasks)]
    tasks = add_dependencies(tasks)

    # Sort by deadline for a natural initial ordering
    tasks.sort(key=lambda t: t["deadline_min"])

    # Disruptions
    n_disruptions = random.randint(1, 3)
    disruptions   = generate_disruptions(tasks, n_disruptions)

    # Feasibility flag: can all tasks fit if scheduled optimally?
    total_duration = sum(t["duration_min"] for t in tasks)
    feasible       = total_duration <= workday_duration_min

    return {
        "scenario_id":      scenario_id,
        "complexity":       complexity_label(n_tasks),
        "n_tasks":          n_tasks,
        "n_disruptions":    len(disruptions),
        "workday_minutes":  workday_duration_min,
        "total_duration_min": total_duration,
        "feasible":         feasible,
        "tasks":            tasks,
        "disruptions":      disruptions,
    }


# ── Main generation loop ─────────────────────────────────────────────────────

def generate_dataset(n_scenarios=1000):
    print(f"Generating {n_scenarios} scenarios...")
    scenarios = []
    for i in range(1, n_scenarios + 1):
        scenario = generate_scenario(f"SCN_{i:04d}")
        scenarios.append(scenario)
        if i % 100 == 0:
            print(f"  {i}/{n_scenarios} done")
    return scenarios


def split_dataset(scenarios, train_ratio=0.8):
    random.shuffle(scenarios)
    split = int(len(scenarios) * train_ratio)
    return scenarios[:split], scenarios[split:]


def save_summary_csv(scenarios, path):
    fields = [
        "scenario_id", "complexity", "n_tasks", "n_disruptions",
        "total_duration_min", "feasible", "workday_minutes"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in scenarios:
            writer.writerow({k: s[k] for k in fields})


def print_stats(scenarios, label="Dataset"):
    complexities = [s["complexity"] for s in scenarios]
    n_tasks_list = [s["n_tasks"] for s in scenarios]
    feasible     = sum(1 for s in scenarios if s["feasible"])

    print(f"\n{'='*50}")
    print(f"  {label} Statistics ({len(scenarios)} scenarios)")
    print(f"{'='*50}")
    print(f"  Complexity — low:    {complexities.count('low'):>4}  ({complexities.count('low')/len(scenarios)*100:.1f}%)")
    print(f"  Complexity — medium: {complexities.count('medium'):>4}  ({complexities.count('medium')/len(scenarios)*100:.1f}%)")
    print(f"  Complexity — high:   {complexities.count('high'):>4}  ({complexities.count('high')/len(scenarios)*100:.1f}%)")
    print(f"  Avg tasks/scenario:  {sum(n_tasks_list)/len(n_tasks_list):.1f}")
    print(f"  Min/Max tasks:       {min(n_tasks_list)} / {max(n_tasks_list)}")
    print(f"  Feasible scenarios:  {feasible} ({feasible/len(scenarios)*100:.1f}%)")

    disc_types = {"urgent_insertion": 0, "deadline_shift": 0, "task_cancellation": 0}
    for s in scenarios:
        for d in s["disruptions"]:
            disc_types[d["type"]] += 1
    total_d = sum(disc_types.values())
    print(f"  Disruption breakdown:")
    for dtype, count in disc_types.items():
        print(f"    {dtype:<25} {count:>4}  ({count/total_d*100:.1f}%)")
    print(f"{'='*50}\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("/home/claude/dataset", exist_ok=True)

    all_scenarios = generate_dataset(1000)
    train, test   = split_dataset(all_scenarios)

    # Save full dataset
    with open("/home/claude/dataset/scenarios.json", "w") as f:
        json.dump(all_scenarios, f, indent=2)

    # Save splits
    with open("/home/claude/dataset/train_scenarios.json", "w") as f:
        json.dump(train, f, indent=2)

    with open("/home/claude/dataset/test_scenarios.json", "w") as f:
        json.dump(test, f, indent=2)

    # Save CSV summary
    save_summary_csv(all_scenarios, "/home/claude/dataset/dataset_summary.csv")

    print_stats(all_scenarios, "Full Dataset")
    print_stats(train,         "Train Split (800)")
    print_stats(test,          "Test Split  (200)")

    print("Files saved:")
    print("  /home/claude/dataset/scenarios.json")
    print("  /home/claude/dataset/train_scenarios.json")
    print("  /home/claude/dataset/test_scenarios.json")
    print("  /home/claude/dataset/dataset_summary.csv")
