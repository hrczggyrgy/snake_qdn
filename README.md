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
├── tests/              # Focused test suites
│   ├── test_env.py
│   ├── test_agent.py
│   ├── test_controller.py
│   ├── test_checkpoint.py
│   └── test_rendering.py
├── test_modules.py     # Legacy module verification tests
├── test_characterization.py  # Behavior characterization tests
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Tool configuration (pytest, Ruff, mypy)
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
# All tests
python -m pytest tests/ -v

# Legacy module tests
python test_modules.py

# Characterization tests (behavior verification)
python test_characterization.py

# Run linting
ruff check .

# Run formatting check
ruff format --check .
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
| Memory Size | 100,000 | Experience replay buffer capacity |
| Batch Size | 1,000 | Training batch size |

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

## Epsilon Decay

The exploration rate ε follows:
```
ε(n) = max(0.01, 1.0 × 0.995ⁿ)
```

Where n is the number of completed training episodes.

Approximate values:
- 100 episodes: 0.606
- 200 episodes: 0.367
- 500 episodes: 0.082
- First reaches floor (0.01) after 919 decay calls

**Note**: ε reaching its floor does not imply successful learning; it only means exploration has minimized.

## Checkpoint Details

### Full Training State (exact resume)
- Model weights (policy + target networks)
- Optimizer state
- Training statistics (episode, epsilon, high score, loss history)
- Model configuration
- Replay buffer contents
- RNG states (Python, NumPy, PyTorch, CUDA)

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

**Preliminary benchmarks** (5 episodes, grid_size=20):
- Config A (hidden=256, batch=1000): ~54 trans/s, replay not yet active
- Config B (hidden=256, batch=256): ~24 trans/s, 7.6 opt/s
- Config C (hidden=128, batch=256): ~44 trans/s, 4.1 opt/s

*These are preliminary results from 5 episodes; not sufficient for convergence claims.*

## Technical Notes

- **No UI freezes**: Uses Streamlit's fragment system with `st.session_state` for state management
- **Auto-reset**: During training, episodes automatically reset on game over
- **Efficient rendering**: Canvas uses coordinate-only Plotly rendering (no RGB/PNG/base64 on normal path)
- **Modular design**: Each component (env, model, agent, controller, persistence) is independently testable
- **Shared training logic**: `TrainingController` used by both dashboard and headless training
- **Epsilon-greedy policy**: Action selection uses ε-greedy; softmax(Q) is shown for visualization only, not the actual behavior policy
- **Terminated vs Truncated**: Terminated (game over) suppresses bootstrap; Truncated (timeout) retains bootstrap

## Limitations & Known Issues

- **Hosted training is resource-dependent**: CPU-only on Streamlit Cloud; no GPU acceleration
- **Server-side Save checkpoint**: Does not download a file to the browser; saves to server filesystem
- **Checkpoints don't include replay memory/RNG by default**: Use `agent.save_checkpoint()` for full state
- **Loading into incompatible architecture**: Unsupported unless explicitly handled (hidden size must match)
- **Hosted session is not durable**: State resets on redeploy or idle timeout
- **No pretrained checkpoint provided**: Train from scratch or use weights-only for inference

## Testing

```bash
# Unit tests
python -m pytest tests/ -v

# Legacy module tests
python test_modules.py

# Behavior characterization tests
python test_characterization.py

# Linting
ruff check .

# Format check
ruff format --check .
```

## License

MIT License - Feel free to use for learning and research!

[LICENSE](LICENSE)

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Interactive web apps
- [PyTorch](https://pytorch.org/) - Deep learning
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Gymnasium](https://gymnasium.farama.org/) - RL environment API