# Performance Improvement Checklist

## Baseline (Before Changes)
- [x] All tests pass (93 pytest + test_modules.py + test_characterization.py)
- [x] App starts with `streamlit run app.py`
- [x] Record current throughput: transitions/sec, optimizer updates/sec
- [x] Document Python, Streamlit, PyTorch versions, CPU/GPU

## Phase 1: Instrument Current Implementation
- [x] Add monotonic timer for throughput measurement
- [x] Track environment transitions per second
- [x] Track optimizer updates per second
- [x] Track completed training episodes
- [x] Track total transitions and elapsed training time
- [x] Track replay-buffer size and replay optimization active status
- [x] Track time spent in: action selection, env stepping, optimization, diagnostics, display prep
- [x] Document whether rates exclude paused time
- [x] Compare cold startup, pre-replay warmup, steady-state training separately

## Phase 2: Extract/Verify Shared Training Logic
- [x] Verify TrainingController owns env, agent, observation, mode, episode lifecycle
- [x] Verify explicit operations: step, restart_board, clear_history, new_run, get_snapshot
- [x] Verify shared by Streamlit dashboard and headless training
- [x] Verify no Streamlit imports in training.py
- [x] Verify explicit TRAIN/EVALUATE/MANUAL modes + paused flag
- [x] Verify training only in TRAIN mode
- [x] Verify greedy action selection without optimization in EVALUATE
- [x] Verify explicit actions in MANUAL
- [x] Verify epsilon decay once per completed training episode
- [x] Verify correct terminated vs truncated bootstrap behavior
- [x] Characterize and resolve epsilon logging/update order discrepancy

## Phase 3: Decouple Training from Display Cadence
- [x] Add execution policies: WATCH, FAST_TRAINING, EVALUATE
- [x] WATCH: deliberately paced, publish each visible move
- [x] FAST_TRAINING: multiple transitions per UI update within bounded compute-time budget
- [x] Publish one coherent latest-state snapshot after burst
- [x] Stop at deadline and maximum-step cap
- [x] Handle episode completion and reset correctly within burst
- [x] Avoid sleeps and unlimited loops on Streamlit execution path
- [x] Burst deadline checked between transitions
- [x] Pause, reset, load, new-agent take effect at well-defined boundaries
- [x] Keep fragment creation, timer configuration, control updates predictable

## Phase 4: Remove Avoidable Hot-Path Work
- [x] Make normal canvas rendering coordinate-only end-to-end
- [x] Do not generate RGB arrays, PNGs, base64 on normal path
- [x] Retain RGB rendering only for explicit export/tests
- [x] Produce diagnostics at display-snapshot cadence, not every fast-training transition
- [x] Collect activations only when network diagnostics enabled
- [x] Consolidate action selection: decisions and diagnostics reuse Q-values
- [x] Never mix pre-action diagnostics with next-state/post-training values
- [x] Preserve model mode correctly around optional inference/diagnostics
- [x] Keep bounded display histories without silently losing export data
- [x] Slice displayed history before converting entire growing histories
- [x] Profile replay sampling
- [x] Verify random.sample compatibility with buffer type and Python
- [x] Do not blindly convert large buffer to list on every update

## Phase 5: Benchmark Configuration Experiments
- [x] Keep current defaults as baseline (A)
- [x] Compare with batch size 256 (B)
- [x] Compare batch size 256 + hidden width 128 (C)
- [x] Do not change epsilon decay, rewards, optimizer frequency, target updates, architecture simultaneously
- [x] Record throughput and wall-clock time for each
- [ ] Compare evaluation at equal transition budgets
- [ ] Keep training and evaluation seeds separate
- [ ] Evaluate with exploration disabled, weights unchanged
- [ ] Report evaluation mean, spread, number of seeds/episodes
- [ ] Ensure evaluation doesn't mutate training state or RNG
- [x] Label short experiments as preliminary
- [x] Document that changing hidden width requires new model/checkpoint

### Preliminary Benchmark Results (5 episodes, grid_size=20)

| Config | Hidden | Batch | Transitions | Time (s) | Trans/s | Optimizer Updates | Opt/s |
|--------|--------|-------|-------------|----------|---------|-------------------|-------|
| A (baseline) | 256 | 1000 | 321 | 9.26 | 53.7 | 0 | 0.0 |
| B | 256 | 256 | 371 | 15.25 | 24.3 | 116 | 7.6 |
| C | 128 | 256 | 281 | 6.41 | 44.0 | 26 | 4.1 |

**Observations:**
- Config A (default batch=1000): Highest transition throughput but replay not yet active (memory < batch_size)
- Config B (batch=256): Replay active, slower transitions due to optimizer overhead
- Config C (batch=256, hidden=128): Fastest overall, lower optimizer overhead due to smaller model
- These are preliminary results from only 5 episodes; not sufficient for convergence claims

## Phase 6: Preserve Rendering Correctness
- [x] Keep canvas component identity stable
- [x] Avoid unnecessary DOM recreation, retain last valid board
- [x] Keep chart layout owned by its fragment
- [x] Render all chart elements on every analytics-fragment execution
- [x] Optimize data preparation separately from element presence
- [x] Keep Streamlit container handles out of Session State
- [x] Preserve stable keys, sensible dimensions, Plotly interaction state
- [x] Verify Streamlit version supports APIs used
- [x] Test: initial load, unchanged chart data, repeated reruns
- [x] Test: Watch/Fast switching, pause/resume/step
- [x] Test: restart board, clear history, new agent, checkpoint load
- [x] Test: Food=None, game over, different grid sizes, browser resize
- [x] Test: diagnostics enabled/disabled
- [x] Report browser checks completed vs not performed

## Phase 7: Rewrite README.md
- [x] Clear project title and one-paragraph purpose
- [x] Prominent live demo: https://snakeqdn.streamlit.app/
- [x] Verified current features
- [x] Virtual-environment installation and local run instructions
- [x] Watch, Fast training, Evaluate, Manual behavior as implemented
- [x] Correct architecture and project layout
- [x] Observations, actions, rewards, training defaults
- [x] Performance measurements with methodology
- [x] Checkpoint contents and limitations
- [x] Test commands and known limitations
- [x] MIT license link
- [x] Correct epsilon explanation: epsilon(n) = max(0.01, 1.0 * 0.995**n)
- [x] Remove/qualify unsupported claims
- [x] Explain exploratory vs greedy evaluation difference
- [x] Explain softmax(Q) is not epsilon-greedy behavior policy
- [x] Hosted training resource-dependent
- [x] Server-side Save checkpoint doesn't download to browser
- [x] Existing checkpoints don't include replay memory and all RNG state
- [x] Loading into incompatible architecture unsupported unless handled
- [x] Hosted session not durable experiment storage

## Phase 8: Tests, Tooling, Cleanup
- [x] Add pytest coverage for controller lifecycle, metrics, bursts, checkpoints, rendering
- [x] Use temporary directories for checkpoint tests
- [x] Record zero loss values correctly (avoid truthiness filtering)
- [x] Check probability helpers and snapshots don't mutate caller arrays
- [x] Add Ruff lint/format configuration if absent
- [x] Keep CI read-only
- [x] Retire one-time patch-application workflow and obsolete patch artifacts
- [x] Preserve import paths or provide compatibility wrappers
- [x] Keep dependency changes minimal and reproducible

## Acceptance Criteria
- [x] `streamlit run app.py` still works
- [x] Training logic testable without importing Streamlit
- [x] Headless and dashboard share transition/episode logic
- [x] Fast training no longer restricted to one move per display tick
- [x] Default algorithm preserved unless intentional change documented
- [x] Evaluation doesn't update weights, epsilon, or training histories
- [x] Metrics distinguish transitions, optimizer updates, episodes
- [x] Normal canvas display generates no image bytes
- [x] Tests cover pause/reset/new-run behavior and unchanged chart reruns
- [x] Existing supported checkpoints still load
- [x] README matches actual implementation
- [x] Applicable tests, lint, format checks pass
- [x] Performance claims backed by recorded measurements