# Snake DQN - Reinforcement Learning Dashboard

A modular Streamlit application where a neural network learns to play Snake via Deep Q-Learning (DQN), with live interactive visualizations of the learning dynamics, decision-making, and neural network internals.

**Live Demo:** https://snakeqdn.streamlit.app/

## Features

### 🎮 Live Game Arena
- Real-time rendering of the Snake board (grid, snake head/body, food)
- Live counters: Current Score, High Score, Episode Count, Current Epsilon
- Playback controls: Start/Pause, Speed Slider, Reset, Step-by-Step Mode
- Manual control buttons for testing

### 🧠 Neural Network Internals & Decision Visualizer
- **Active State Vector**: Visual breakdown of the 11 inputs with color-coded indicators
- **Network Architecture Diagram**: Dynamic node/edge activation diagram showing layer weights and firing signals
- **Action Probabilities / Q-Values**: Real-time bar charts showing predicted values for [Straight, Right, Left], highlighting the chosen action
- **Decision Snapshot**: Pre-action observation, Q-values, and selection reason (exploring/exploiting/manual)

### 📈 Learning Analytics & Metrics
- **Performance Chart**: Dynamic plot showing Score per Episode alongside Rolling Average (50-episode window)
- **Loss Curve**: Real-time plot of Bellman loss over training steps (log scale)
- **Exploration vs. Exploitation**: Epsilon decay tracking over episodes
- **Training Metrics**: Environment steps, optimizer updates, replay buffer size, rolling averages
- **Bellman Update Explanation**: Visual breakdown of `y = r + γ(1-d)max Q'` and TD error for sampled replay transitions

### ⚡ Execution Policies
- **Watch**: Deliberately paced, readable movement (configurable speed)
- **FastTraining**: Multiple transitions per UI update within a bounded compute-time budget
- **Evaluate**: Greedy policy, no learning, separate metrics
- **Manual**: Human-controlled snake for testing

## Architecture

```
snake/
├── app.py              # Main Streamlit application (UI, callbacks, fragments)
├── training.py         # TrainingController - framework-independent training logic
├── persistence.py      # Checkpoint persistence with versioning & validation
├── snake_env.py        # Gymnasium-compatible Snake environment
├── model.py            # PyTorch neural network models (Linear_QNet, QTrainer)
├── agent.py            # DQN Agent with experience replay
├── experiments.py      # Controlled experiment framework
├── tests/              # Focused test suites (129 tests)
│   ├── test_agent.py
│   ├── test_characterization.py
│   ├── test_checkpoint.py
│   ├── test_controller.py
│   ├── test_env.py
│   ├── test_episode_continuation.py
│   ├── test_modules.py
│   ├── test_rendering.py
│   └── test_rendering_smoke.py
├── pyproject.toml      # Tool configuration (pytest, Ruff, mypy)
├── requirements.txt    # Python dependencies
└── checkpoints/        # Model checkpoints (auto-created)
```

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (for testing/linting)
pip install -e ".[dev]"
```

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Run Tests
```bash
# All tests (128 pass, 1 flaky, 4 env failures)
python -m pytest tests/ -v

# Run linting
ruff check .

# Run formatting check
ruff format --check .

# Type checking
mypy training.py agent.py model.py
```

### Headless Training (Programmatic)
```python
from training import TrainingController, train_headless, ActivityMode, EnvConfig, AgentConfig

# Configure
controller = TrainingController(
    env_config=EnvConfig(grid_size=20), agent_config=AgentConfig(hidden_size=256, lr=0.001)
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
from agent import Agent

agent = Agent()
# ... train ...

# Full training state (for exact resume)
agent.save_checkpoint("checkpoints/full.pth")
info = agent.load_checkpoint("checkpoints/full.pth")

# Weights only (for inference/deployment)
agent.save_weights_only("checkpoints/weights.pth")
config = agent.load_weights_only("checkpoints/weights.pth")
```

### Controlled Experiments (Budget-Matched)
```python
from experiments import ExperimentRunner, ExperimentConfig, ExperimentBudget, AgentConfig, EnvConfig

# Define experiment
config = ExperimentConfig(
    name="baseline",
    description="Baseline DQN",
    agent_config=AgentConfig(hidden_size=256, lr=0.001, dropout=0.1),
    env_config=EnvConfig(grid_size=20),
    budget=ExperimentBudget(max_env_transitions=500_000),
    seed=42,
)

# Run
runner = ExperimentRunner("results")
result = runner.run_experiment(config, eval_every=10_000, eval_episodes=5, verbose=True)

# Compare results
runner.compare_results([result1, result2, ...])
```

## Environment Details

### Observation Space (11-dimensional)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Danger Straight | Collision within 1 step ahead |
| 1 | Danger Right | Collision within 1 step to the right |
| 2 | Danger Left | Collision within 1 step to the left |
| 3 | Direction Left | Currently moving left (one-hot) |
| 4 | Direction Right | Currently moving right (one-hot) |
| 5 | Direction Up | Currently moving up (one-hot) |
| 6 | Direction Down | Currently moving down (one-hot) |
| 7 | Food Left | Food is to the left of head |
| 8 | Food Right | Food is to the right of head |
| 9 | Food Up | Food is above head |
| 10 | Food Down | Food is below head |

### Action Space (3 discrete)
| Action | Description |
|--------|-------------|
| 0 | Straight (continue current direction) |
| 1 | Turn Right (90° clockwise) |
| 2 | Turn Left (90° counter-clockwise) |

### Reward Function
- **Eating food**: +10
- **Game Over** (wall/self collision): -10
- **Step penalty**: -0.01 (configurable, discourages wandering)
- **Timeout** (max steps without food): -10 (truncated, not terminated)

## Key Improvements (v0.2.0)

### P0 — Trustworthy Behavior
| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Configuration wiring** | `dropout`, `tau` now properly passed through `AgentConfig` → `Agent` → `Linear_QNet`/`QTrainer` |
| 2 | **Optimizer metrics** | Separate `env_transitions`, `training_transitions`, `optimizer_updates` with session/active time rates |
| 3 | **Clear history** | Display-only operation preserving lifetime counters (`n_games`, `total_steps`, `epsilon`, replay) |
| 4 | **Episode outcomes** | `termination_reason` from env, `recent_terminations` deque (maxlen=50), correct denominators |
| 5 | **Checkpoints** | Schema v3, environment RNG state, mid-episode resume, bounded replay/loss history |

### P1 — Lower Overhead
| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Unified replay schedule** | `maybe_optimize()` with `learning_starts`, `train_every`, `gradient_steps` replaces dual `train_short/long_memory` |
| 2 | **Diagnostic gating** | `compute_q_values=False` during FastTraining exploration (~200x speedup in action selection) |
| 3 | **Compact replay buffer** | Preallocated numpy arrays, circular buffer, O(1) sampling, backward-compatible checkpoints |
| 4 | **Game rules** | `allow_tail_entry` config, action validation (0/1/2), optimized food placement via precomputed empty cells |

### P2 — Educational Dashboard
| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Bellman update visualization** | `render_bellman_update()` shows `y = r + γ(1-d)max Q'`, TD error for sampled replay transitions |
| 2 | **Experiment framework** | `ExperimentRunner` with `ExperimentBudget` (transitions/time/optimizer steps/episodes), eval scheduling, ablation configs |
| 3 | **Step-based epsilon decay** | New `epsilon_decay_steps` parameter with linear decay over training transitions |

## Neural Network Architecture

```
Input (11) → Linear(256) → ReLU → Dropout(0.1)
         → Linear(256) → ReLU → Dropout(0.1)
         → Linear(3)   → Q-values [Straight, Right, Left]
```

- **Optimizer**: Adam (lr=0.001)
- **Loss**: Huber (SmoothL1Loss) - more stable than MSE
- **Discount factor (γ)**: 0.99
- **Target Network**: Soft updates (τ=0.005)

## Training Hyperparameters (Configurable via Sidebar)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Hidden Size | 256 | Neurons in hidden layers |
| Learning Rate | 0.001 | Adam optimizer learning rate |
| Gamma | 0.99 | Discount factor for future rewards |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Minimum exploration rate |
| Epsilon Decay | 0.995 | Multiplicative decay per episode |
| Epsilon Decay Steps | 0 | If > 0, linear decay over training transitions |
| Memory Size | 100,000 | Experience replay buffer capacity |
| Batch Size | 256 | Training batch size |
| Learning Starts | 1,000 | Transitions before training begins |
| Train Every | 1 | Train every N environment transitions |
| Gradient Steps | 1 | Optimizer steps per training call |
| Tau | 0.005 | Soft target update coefficient |
| Dropout | 0.1 | Dropout rate for hidden layers (0 = disabled) |

## Epsilon Decay Modes

### Per-Episode (Legacy)
```python
# Default: multiplicative decay per episode
ε(n) = max(0.01, 1.0 × 0.995ⁿ)
```

### Per-Training-Transition (New)
```python
# If epsilon_decay_steps > 0: linear decay over training transitions
progress = min(1.0, training_transitions / epsilon_decay_steps)
ε = ε_start + (ε_end - ε_start) × progress
```

## Controls

### Sidebar
- **Environment**: Grid size, reward shaping
- **Agent**: All hyperparameters (recreate agent to apply)
- **Training**: Start/Stop, Reset stats, Execution policy (Watch/FastTraining), Step mode
- **Checkpoints**: Save/Load model weights
- **Display**: Grid toggle, New game

### Main View
- **Play/Pause**: Toggle automatic stepping
- **Step**: Single frame advance (in step mode)
- **Reset**: Restart current game
- **Manual Controls**: Arrow buttons when paused (Manual mode)

## Checkpoint Details

### Full Training State (exact resume)
- Model weights (policy + target networks)
- Optimizer state
- Training statistics (episode, epsilon, high score, loss history)
- Model configuration
- Replay buffer contents
- RNG states (Python, NumPy, PyTorch, CUDA)
- Environment state (snake, food, direction, starvation counter, local RNG)

### Weights-Only (inference)
- Model weights only
- Model configuration

```python
from agent import Agent

agent = Agent()

# Full training state (exact resume)
agent.save_checkpoint("my_model.pth")
info = agent.load_checkpoint("my_model.pth")  # Returns episode, epsilon, high_score

# Weights only (inference)
agent.save_weights_only("my_weights.pth")
config = agent.load_weights_only("my_weights.pth")
```

**Note**: Exact training session resumption requires restoring all counters, RNG state, replay buffer, and configuration. The full checkpoint supports this; weights-only does not.

## Performance Notes

The dashboard supports two execution policies:

| Policy | Description | Use Case |
|--------|-------------|----------|
| **Watch** | One transition per UI update (configurable 0-100ms delay) | Watching the agent learn, debugging |
| **FastTraining** | Multiple transitions per UI update (bounded by step cap & time budget) | Maximum training throughput |

**Preliminary benchmarks** (2000 transitions, grid_size=20):
- **Watch mode**: ~350 trans/s (includes UI overhead)
- **FastTraining mode**: ~1000+ trans/s, diagnostic gating provides ~200x action selection speedup

## Testing

```bash
# All tests (128 pass, 1 flaky due to test interference, 4 env failures)
python -m pytest tests/ -v

# Run linting
ruff check .

# Run formatting check
ruff format --check .

# Type checking
mypy training.py agent.py model.py
```

## Technical Notes

- **No UI freezes**: Uses Streamlit's fragment system with `st.session_state` for state management
- **Auto-reset**: During training, episodes automatically restart on game over
- **Efficient rendering**: Plotly coordinate-only rendering (no RGB/PNG/base64 on normal path)
- **Modular design**: Each component (env, model, agent, controller, persistence) is independently testable
- **Shared training logic**: `TrainingController` used by both dashboard and headless training
- **Epsilon-greedy policy**: Action selection uses ε-greedy; softmax(Q) is shown for visualization only, not the actual behavior policy
- **Terminated vs Truncated**: Terminated (game over) suppresses bootstrap; Truncated (timeout) retains bootstrap
- **Diagnostic gating**: During FastTraining exploration, Q-value computation is skipped (~200x speedup)
- **Compact replay buffer**: Preallocated numpy arrays with circular buffer, O(1) push and sample
- **Environment-local RNG**: Each environment has isolated random generators for reproducibility

## Limitations & Known Issues

- **Hosted training is resource-dependent**: CPU-only on Streamlit Cloud; no GPU acceleration
- **Server-side Save checkpoint**: Does not download a file to the browser; saves to server filesystem
- **Flaky test**: `test_auto_episode_continuation_fast_training` passes in isolation but fails in suite due to test interference (random state from watch mode test affects fast training test)
- **Hosted session is not durable**: State resets on redeploy or idle timeout
- **No pretrained checkpoint provided**: Train from scratch or use weights-only for inference

## License

MIT License - Feel free to use for learning and research!

[LICENSE](LICENSE)

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Interactive web apps
- [PyTorch](https://pytorch.org/) - Deep learning
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Gymnasium](https://gymnasium.farama.org/) - RL environment API