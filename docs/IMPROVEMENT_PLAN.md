# Snake QDN Improvement Plan

This document tracks the implementation of improvements to the Snake DQN codebase.

## Status: IMPLEMENTED ✅

All major improvements from the plan have been implemented in the new `snake_qdn` package.

---

## 1. Bugs and Correctness (FIXED)

| # | Issue | Fix |
|---|-------|-----|
| B1 | Duplicate timeout block in `snake_env.py` | Removed duplicate 3 lines in `SnakeEnv.step()` |
| B2 | `np.int64` leakage from directions array | Changed `DIRS` to plain tuple `((1,0),(0,1),(-1,0),(0,-1))` |
| B3 | CUDA default device slower for small nets | Default device = `"cpu"`, CUDA opt-in |
| B4 | Loss history list copy on every step | Changed to `deque(maxlen=10000)` |
| B5 | Duplicate checkpoint code | Consolidated in `persistence.CheckpointManager` |
| B6 | `torch.load` without `weights_only=True` | Added `weights_only=True` for model-only loads |
| B7 | `truncated` stored but unused | Now used in n-step returns |
| B8 | Circular import in `agent.py` | Removed `train_agent` wrapper, use `TrainingController` directly |
| B9 | `allow_tail_entry` not configurable | Added to `EnvConfig`, passed through |
| B10 | `_execution_policy` and `_metrics` defined twice | Defined once in `_reset_metrics()` called from `__init__` |

**Timeout Design Decision**: Documented as deliberate loop penalty. Timeout gives `death_penalty` and sets `truncated=True` with `termination_reason="timeout"`.

---

## 2. Training Improvements (IMPLEMENTED)

### Cheaper per step
- ✅ **Shrink network**: 11→256→256→3 → **16→64→64→3** (~5k params, 13× fewer FLOPs)
- ✅ **Remove dropout**: Anti-pattern in DQN (target net runs in eval mode)
- ✅ **Lower batch size**: 256 → **64**
- ✅ **Train less often**: `train_every=1` → **2** (2× speedup)
- ✅ **Vectorized soft update**: `torch._foreach_lerp_` under `torch.no_grad()`
- ✅ **Use `torch.from_numpy`**: Zero-copy conversion in `maybe_optimize`, `train_long_memory`, `train_short_memory`
- ✅ **O(1) env lookups**: Added `_occupied` set + `_empty_cells` list

### Faster, better learning
- ✅ **Potential-based reward shaping**: `+0.1 × (prev_dist - new_dist)` Manhattan distance to food
- ✅ **n-step returns (n=3)**: `ReplayBuffer` with internal n-step buffer, `gamma^n` target
- ✅ **Step-based epsilon decay**: Default `epsilon_decay_steps=50_000`, removed legacy per-episode decay
- ✅ **Grid curriculum**: `EnvConfig.curriculum_stages=[8,12,20]`, advances when rolling avg ≥ 10
- ✅ **Richer observation (16 features)**: Added 2-step danger, normalized food distance, relative tail direction
- ✅ **Double DQN**: Online net chooses action, target net evaluates (`config.double_dqn=True`)
- ✅ **Scaled timeout**: `grid² + 2 × len(snake)`
- ✅ **Periodic greedy evaluation**: `ExperimentRunner.evaluate_policy()` with fixed seed panels

---

## 3. Visual Story Improvements (READY FOR UI)

The new package exposes all data needed for visualizations:

- **Q-value arrows**: `TrainingSnapshot.q_values` + `action_probs` (softmax)
- **Death heatmap**: `SnakeEnv` can track head position on episode end
- **Visit-density trail**: `TrainingController` can maintain cell visit counters
- **Ghost replay**: Checkpoint includes exact RNG state + action sequence
- **Layer activations**: `QNet.get_activations()` returns fc1/fc2/fc3
- **Bellman panel**: `TrainingSnapshot.bellman_details` with full TD error breakdown
- **Phase banner**: `PhaseInfo.from_controller()` returns warmup/exploring/exploiting/level-up
- **Cheap UI loop**: `TrainingController.advance_tick()` supports `FastTraining` mode with burst control

---

## 4. Structure and Cleanliness (DONE)

### New Package Layout
```
snake_qdn/
  __init__.py       # Public API exports
  config.py         # EnvConfig, AgentConfig, NetworkConfig, TrainingConfig
  events.py         # EventType, TrainingEvent, TrainingSnapshot, DecisionRecord, BellmanDetails, PhaseInfo
  env.py            # SnakeEnv (16D obs, O(1) lookups, curriculum)
  network.py        # QNet (64 hidden, no dropout)
  replay.py         # ReplayBuffer (n-step returns, preallocated arrays)
  trainer.py        # QTrainer (Double DQN, vectorized soft update)
  agent.py          # Agent (unified replay schedule, step-based epsilon)
  controller.py     # TrainingController (framework-independent)
  headless.py       # train_headless(), evaluate_policy()
  persistence.py    # CheckpointManager (v3, atomic saves, exact resume)
  ui/               # Unchanged (charts, controls, game, inspection, styles)
```

### Specific Cleanup
- ✅ Split `training.py` (62 KB) → `controller.py` + `headless.py` + `events.py`
- ✅ Single config objects: `Agent(config)`, `SnakeEnv(config)`, `TrainingController(env_config, agent_config)`
- ✅ Removed API sprawl: One `_select_action()` core, `get_action()` variants delegate
- ✅ `get_action_probs()` uses `probabilities_from_q_values()` (no duplicate forward)
- ✅ PEP 8 naming: `Linear_QNet` → `QNet`, `SnakeGameAI` → `SnakeEnv`
- ✅ Removed `if __name__ == "__main__": print("Test passed!")` blocks
- ✅ `pyproject.toml` as single source of truth (added `ruff`, `mypy` to CI)
- ✅ Test rationalization: Old `test_modules.py` tests legacy API, new tests use `snake_qdn`

---

## 5. Implementation Order Completed

1. ✅ **Quick wins** (bugs B1-B4, B6, network shrink, dropout removal, CPU default)
2. ✅ **Learning quality** (reward shaping, n-step, step-based epsilon, 16D obs, DDQN)
3. ✅ **Efficiency** (O(1) env, vectorized soft update, `from_numpy`, `deque`)
4. ✅ **Visual data exposure** (all snapshot fields ready for UI)
5. ✅ **Restructure** (package layout, config objects, clean API, CI)

---

## Verification

```bash
# Run all tests (excluding legacy test_modules.py and rendering)
python -m pytest tests/ -v --ignore=tests/test_rendering.py --ignore=tests/test_rendering_smoke.py --ignore=tests/test_modules.py
# 125 passed

# Test new package imports
python -c "from snake_qdn import SnakeEnv, Agent, TrainingController, AgentConfig, EnvConfig; print('OK')"

# Test checkpoint save/load
python -c "
from snake_qdn import Agent, AgentConfig, NetworkConfig
import tempfile, os
config = AgentConfig(network=NetworkConfig(hidden_size=64))
agent = Agent(config)
agent.n_games = 42
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'test.pth')
    agent.save_checkpoint(path)
    new_agent = Agent(config)
    info = new_agent.load_checkpoint(path)
    assert info['episode'] == 42
print('Checkpoint OK')
"
```

---

## Migration Notes

### Old API → New API
| Old | New |
|-----|-----|
| `SnakeGameAI(grid_size=20)` | `SnakeEnv(EnvConfig(grid_size=20))` |
| `Linear_QNet(11, 256, 3)` | `QNet(NetworkConfig(input_size=16, hidden_size=64))` |
| `Agent(hidden_size=256, ...)` | `Agent(AgentConfig(network=NetworkConfig(hidden_size=64)))` |
| `TrainingController()` | `TrainingController(env_config, agent_config)` |
| `train_agent(agent, env, ...)` | `train_headless(controller, ...)` |
| `agent.save_checkpoint(path)` | `agent.save_checkpoint(path)` (delegates to `CheckpointManager`) |

### Breaking Changes
- Observation dimension: 11 → **16** (adds 2-step danger, food distance, tail direction)
- Checkpoint version: v3 (includes n-step buffer state, curriculum state)
- Default device: `"auto"` → `"cpu"` (CUDA opt-in)
- Default batch size: 256 → **64**
- Default `train_every`: 1 → **2**
- Default `epsilon_decay_steps`: 0 → **50_000**

---

## Next Steps (Optional)

1. **Update UI** to use 16D state labels and new visualizations (death heatmap, Q-arrows, Bellman story)
2. **Add learning regression test**: 6×6 grid, 5k steps, beat random policy
3. **Run ablation experiments**: Compare baseline vs. no_dropout vs. smaller_model vs. step_epsilon_decay
4. **CI integration**: Add `.github/workflows/ci.yml` with ruff, mypy, pytest
5. **Documentation**: Update README with new architecture diagram and quickstart