# Refactoring Plan: Snake DQN Dashboard

## Baseline Assessment (Commit 2353f7d)

### Current State
- **Files**: app.py (1211 lines), agent.py (437 lines), model.py (370 lines), snake_env.py (406 lines), test_modules.py (450 lines)
- **Architecture**: Monolithic Streamlit app with game logic, training loop, and UI in app.py
- **Tests**: All 9 test suites pass (env, model, agent, integration, timeout, bootstrap, victory, checkpoint, epsilon decay)
- **Key Issues Identified**:
  1. Training logic duplicated between `train_agent()` (headless) and dashboard fragments
  2. Streamlit container handles stored in session state (chart slots)
  3. Canvas rendering uses RGB→PNG→base64 path instead of coordinate-only
  4. Epsilon decay order discrepancy between dashboard and `train_agent()`
  5. No shared TrainingController abstraction
  6. Configuration scattered as dicts in session state
  7. Metrics ownership ambiguous (agent vs dashboard vs trainer)
  8. `get_action()` and `get_action_with_reason()` duplicate decision logic
  9. Checkpoint persistence mixed into Agent/QTrainer
  10. Global RNG seeding in environment

---

## Implementation Checklist

### Stage A: Characterization Tests
- [x] Train, evaluate, manual, pause/resume, single-step behavior tests
- [x] Terminal transitions suppressing bootstrap; truncation retaining bootstrap
- [x] Epsilon decaying once per completed training episode (not during evaluation)
- [x] Document epsilon logging/update order discrepancy as explicit fix
- [x] Board restart, history reset, new-run behavior tests
- [x] Checkpoint round trips and incompatible architecture handling
- [x] Zero-valued loss recording (not discarded by truthiness)
- [x] Food=None, full-board victory, small grids, game-over states
- [x] ReplayMemory.sample compatibility with Python version
- [x] Probability helpers not mutating caller-owned input arrays
- [x] Rendering after unchanged data, repeated full runs, control interactions

### Stage B: Extract Shared Application Logic
- [x] Create `training.py` with `TrainingController` class
- [x] Owns: environment, agent, current observation, mode, episode lifecycle
- [x] Exposes: step, restart_board, clear_history, new_run, get_snapshot
- [x] One transition/episode implementation shared by headless and dashboard
- [x] No Streamlit imports or rendering calls
- [x] Small enum for TRAIN/EVALUATE/MANUAL + separate paused flag
- [x] Keep app.py focused on page setup, UI, callbacks, fragment scheduling

### Stage C: Configuration, State, and Metrics
- [x] Validated configuration dataclasses (EnvConfig, AgentConfig, DisplayConfig)
- [x] One authoritative owner per metric (agent for training stats, trainer for loss)
- [x] Distinguish env steps, optimizer updates, completed episodes
- [x] Distinguish completed-episode epsilon from live epsilon
- [x] Publish copied snapshots (frozen dataclass with copied arrays)
- [x] Keep pre-action obs, actions, Q-values, reasons aligned
- [x] Use run/episode/step identifiers for UI cache invalidation
- [x] Keep Streamlit container handles out of Session State

### Stage D: Performance Without Algorithm Changes
- [x] Unify `get_action()` and `get_action_with_reason()` around one decision path
- [x] Reuse Q-values for probabilities and diagnostics
- [x] Compute optional activations only when requested
- [x] Preserve/restore model training/eval mode around diagnostics
- [ ] Canvas rendering coordinate-only end-to-end
- [ ] Retain explicit RGB export if still used
- [x] Bounded chart windows with documented complete-history policy
- [x] Avoid converting entire growing histories before slicing tail
- [ ] Profile replay sampling; document O(buffer-size) conversion tradeoff
- [ ] Remove unused fields/imports/figures/fallback paths/dependencies

### Stage E: Rendering
- [ ] Preserve stable canvas component identity
- [ ] Updates don't unnecessarily recreate DOM/frontend state
- [ ] Retain latest valid board while waiting for new data
- [ ] Chart layout inside its fragment
- [ ] Render all chart elements on every analytics-fragment execution
- [ ] Preserve stable keys, sensible dimensions, Plotly interaction settings
- [ ] Verify Streamlit version supports APIs used
- [ ] Backend smoke tests and browser checks (if tooling available)

### Stage F: Checkpoints and Randomness
- [x] Move persistence to dedicated `persistence.py` module
- [x] Version checkpoint schema, validate model config on load
- [x] Preserve legacy loading with explicit compatibility handling
- [x] Distinguish loading weights from resuming exact training session
- [x] Atomic saves where appropriate
- [ ] Replace global RNG seeding with instance-local RNG (tested separately)
- [x] No unsafe deserialization options

### Stage G: Tooling, CI, and Documentation
- [x] Add pyproject.toml for pytest and Ruff
- [x] Split tests: test_env.py, test_agent.py, test_controller.py, test_checkpoint.py, test_rendering.py
- [x] Use pytest tmp_path for checkpoint files
- [x] Preserve import paths or provide compatibility exports
- [ ] Clear runtime vs dev dependencies
- [ ] Remove obsolete patch artifacts and auto-commit workflow
- [x] Correct README claims to match implementation
- [x] Don't describe softmax(Q) as epsilon-greedy behavior policy

---

## Validation Commands
```bash
# Run all tests
python -m pytest tests/ -v

# Run lint
ruff check .

# Run format check
ruff format --check .

# Run module verification
python test_modules.py

# Start app
streamlit run app.py
```

---

## Progress Log

*Will be updated during implementation*