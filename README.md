# Snake DQN - Reinforcement Learning Dashboard

A complete, modular Streamlit application where a neural network learns to play Snake via Deep Q-Learning (DQN), with live interactive visualizations of the learning dynamics, decision-making, and neural network internals.

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

### 📈 Learning Analytics & Metrics
- **Performance Chart**: Dynamic plot showing Score per Episode alongside Rolling Average (50-episode window)
- **Loss Curve**: Real-time plot of Bellman loss over training steps (log scale with smoothing)
- **Exploration vs. Exploitation**: Epsilon decay tracking over episodes
- **Score Distribution**: Histogram of achieved scores

## Architecture

```
snake/
├── app.py              # Main Streamlit application
├── snake_env.py        # Gymnasium-compatible Snake environment
├── model.py            # PyTorch neural network models (Linear_QNet, QTrainer)
├── agent.py            # DQN Agent with experience replay
├── test_modules.py     # Module verification tests
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
```

Or install system-wide (with override):
```bash
pip install --break-system-packages -r requirements.txt
```

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Run Tests
```bash
python test_modules.py
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
- **Training**: Start/Stop, Reset stats, Speed control, Step mode
- **Checkpoints**: Save/Load model weights
- **Display**: Grid toggle, New game

### Main View
- **Play/Pause**: Toggle automatic stepping
- **Step**: Single frame advance (in step mode)
- **Reset**: Restart current game
- **Manual Controls**: Arrow buttons when paused

## Training Tips

1. **Start with defaults** - they work well for 20x20 grid
2. **Watch epsilon decay** - should reach ~0.01 after ~500 episodes
3. **Monitor loss** - should decrease and stabilize
4. **Rolling average (50)** - target >10 for competent play
5. **Save checkpoints** - use the sidebar to preserve progress

## Expected Performance

| Episodes | Avg Score (50) | Behavior |
|----------|----------------|----------|
| 0-50 | 1-3 | Random exploration |
| 50-200 | 3-8 | Learning basic survival |
| 200-500 | 8-15 | Competent food seeking |
| 500+ | 15+ | Expert play, long games |

## Saving/Loading Models

Checkpoints save:
- Model weights (policy + target networks)
- Optimizer state
- Training statistics (episode, epsilon, high score, loss history)
- Model configuration

```python
# Programmatic usage
from agent import Agent
agent = Agent()
agent.save_checkpoint("my_model.pth")
info = agent.load_checkpoint("my_model.pth")
```

## Technical Notes

- **No UI freezes**: Uses Streamlit's rerun lifecycle with `st.session_state` for state management
- **Auto-reset**: During training, episodes automatically reset on game over
- **Efficient rendering**: Plotly charts update incrementally
- **Modular design**: Each component (env, model, agent) is independently testable

## Troubleshooting

### "Module not found" errors
```bash
pip install --break-system-packages -r requirements.txt
```

### Streamlit doesn't start
```bash
streamlit run app.py --server.port 8501
```

### Slow training
- Reduce grid size in sidebar
- Increase game speed (lower ms/frame)
- Use smaller hidden size (128)

### CUDA/GPU
The app auto-detects CUDA. For CPU-only:
```python
# In agent.py, change device='auto' to device='cpu'
```

## License

MIT License - Feel free to use for learning and research!

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Interactive web apps
- [PyTorch](https://pytorch.org/) - Deep learning
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Gymnasium](https://gymnasium.farama.org/) - RL environment API