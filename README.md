# Snake DQN - Reinforcement Learning Dashboard

A modular Streamlit application where a neural network learns to play Snake via Deep Q-Learning (DQN), with live interactive visualizations of the learning dynamics, decision-making, and neural network internals.

## Features

### 🎮 Live Game Arena
- Real-time rendering of the Snake board with live counters
- Playback controls: Start/Pause, Speed Slider, Reset, Step-by-Step
- Manual control buttons for testing

### 🧠 Neural Network Internals & Decision Visualizer
- **Active State Vector**: Visual breakdown of the 16 inputs with color-coded indicators
- **Network Architecture Diagram**: Dynamic node/edge activation diagram
- **Action Probabilities / Q-Values**: Real-time bar charts with chosen action highlighted
- **Bellman Update Explanation**: Visual breakdown of `y = r + γⁿ(1-d)max Q'` and TD error

### 📈 Learning Analytics & Metrics
- **Performance Chart**: Score per Episode with Rolling Average (50-episode window)
- **Loss Curve**: Real-time plot of Bellman loss (log scale)
- **Exploration vs. Exploitation**: Epsilon decay tracking
- **Episode Outcomes**: Horizontal stacked bar of termination reasons

### ⚡ Execution Policies
- **Watch**: Deliberately paced, readable movement
- **FastTraining**: Multiple transitions per UI update within time budget
- **Evaluate**: Greedy policy, no learning
- **Manual**: Human-controlled snake

## Quickstart

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run the dashboard
streamlit run app.py
```

Then open `http://localhost:8501`

## Project Structure

```
snake/
├── app.py                    # Streamlit dashboard entry point
├── pyproject.toml            # Single source of truth (deps, tools, config)
├── LICENSE
├── README.md
├── docs/                     # Documentation
│   ├── IMPROVEMENT_PLAN.md   # Implementation roadmap
│   ├── REFACTORING_REPORT.md # Original refactoring notes
│   └── ...
├── legacy/                   # Old flat modules (deprecated)
├── snake_qdn/                # NEW: Clean package structure
│   ├── __init__.py           # Public API
│   ├── config.py             # EnvConfig, AgentConfig, NetworkConfig, TrainingConfig
│   ├── events.py             # EventType, TrainingEvent, TrainingSnapshot, PhaseInfo
│   ├── env.py                # SnakeEnv (16D obs, O(1) lookups, curriculum)
│   ├── network.py            # QNet (64 hidden, no dropout)
│   ├── replay.py             # ReplayBuffer (n=3 step returns, preallocated arrays)
│   ├── trainer.py            # QTrainer (Double DQN, vectorized soft update)
│   ├── agent.py              # Agent (unified replay schedule, step-based epsilon)
│   ├── controller.py         # TrainingController (framework-independent)
│   ├── headless.py           # train_headless(), evaluate_policy()
│   └── persistence.py        # CheckpointManager (v3, atomic saves, exact resume)
├── tests/                    # 125 passing tests
│   ├── test_agent.py
│   ├── test_checkpoint.py
│   ├── test_controller.py
│   ├── test_env.py
│   ├── test_experiments.py
│   └── ...
└── ui/                       # Dashboard components (unchanged)
```

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

### Headless Training (Programmatic)
```python
from snake_qdn import (
    TrainingController, train_headless, ActivityMode, 
    EnvConfig, AgentConfig, NetworkConfig
)

# Configure
controller = TrainingController(
    env_config=EnvConfig(grid_size=20),
    agent_config=AgentConfig(network=NetworkConfig(hidden_size=64))
)

# Train
history = train_headless(controller, n_episodes=1000)

# Or step manually
controller.set_mode(ActivityMode.TRAIN)
controller.set_paused(False)
controller.restart_board()
while not controller.game_over:
    snapshot = controller.step()
    # Use snapshot for logging/visualization
```

### Checkpoint Management
```python
from snake_qdn import Agent, AgentConfig, NetworkConfig

config = AgentConfig(network=NetworkConfig(hidden_size=64))
agent = Agent(config)
# ... train ...

# Full training state (exact resume)
agent.save_checkpoint("checkpoints/full.pth")
info = agent.load_checkpoint("checkpoints/full.pth")

# Weights only (for inference/deployment)
agent.save_weights_only("checkpoints/weights.pth")
config = agent.load_weights_only("checkpoints/weights.pth")
```

### Controlled Experiments
```python
from experiments import ExperimentRunner, ExperimentConfig, ExperimentBudget
from snake_qdn import AgentConfig, EnvConfig

config = ExperimentConfig(
    name="baseline",
    description="Baseline DQN",
    agent_config=AgentConfig(network=NetworkConfig(hidden_size=64)),
    env_config=EnvConfig(grid_size=20),
    budget=ExperimentBudget(max_env_transitions=500_000),
    seed=42,
)

runner = ExperimentRunner("results")
result = runner.run_experiment(config, eval_every=10_000, eval_episodes=5)
runner.compare_results([result1, result2, ...])
```

## Key Improvements (v2.0.0)

| Area | Change |
|------|--------|
| **Network** | 11→256→256→3 → **16→64→64→3** (13× fewer FLOPs) |
| **Dropout** | Removed (anti-pattern in DQN) |
| **Batch/Train** | 256→64, train_every=1→2 (2-4× speedup) |
| **Reward Shaping** | +0.1 × Manhattan distance delta (potential-based) |
| **n-step Returns** | n=3 with γⁿ target |
| **Epsilon** | Step-based decay (50k transitions) by default |
| **Curriculum** | 8×8 → 12×12 → 20×20 on rolling avg ≥ 10 |
| **Observation** | 11→16 features (2-step danger, food distance, tail direction) |
| **Double DQN** | Online chooses, target evaluates |
| **Device** | Default CPU, CUDA opt-in |
| **Vectorization** | `torch._foreach_lerp_`, `torch.from_numpy` |
| **Env Speed** | O(1) collision via `_occupied` set |

## Environment Details

### Observation Space (16-dimensional)
| Index | Feature | Description |
|-------|---------|-------------|
| 0-2 | Danger (1-step) | Straight, Right, Left |
| 3-5 | Danger (2-step) | Straight, Right, Left |
| 6-9 | Direction | Left, Right, Up, Down (one-hot) |
| 10-13 | Food | Left, Right, Up, Down |
| 14 | Food Dist Norm | Normalized Manhattan distance |
| 15 | Tail Dir Rel | Relative tail direction |

### Action Space (3 discrete)
| Action | Description |
|--------|-------------|
| 0 | Straight |
| 1 | Turn Right |
| 2 | Turn Left |

### Reward Function
- **Eating food**: +10
- **Game Over** (wall/self collision): -10
- **Step penalty**: -0.01
- **Timeout** (max steps without food): -10 (truncated, not terminated)
- **Shaping**: +0.1 × (prev_dist - new_dist) Manhattan distance

## Testing

```bash
# All tests (125 pass)
python -m pytest tests/ -v

# Linting
ruff check .

# Formatting
ruff format --check .

# Type checking
mypy snake_qdn/
```

## License

MIT License - Feel free to use for learning and research!

[LICENSE](LICENSE)