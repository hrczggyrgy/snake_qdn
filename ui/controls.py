"""
Control components: Primary toolbar, sidebar configuration forms.
"""

import streamlit as st

from training import ActivityMode, AgentConfig, EnvConfig, TrainingController

# =============================================================================
# Primary Toolbar (Main Content Area)
# =============================================================================


def render_primary_toolbar(controller: TrainingController) -> None:
    """Render the single primary control toolbar in the main content area.

    Uses native Streamlit containers/columns for proper layout.
    """
    with st.container():
        # Row 1: Mode selector + Execution policy + Speed
        mode_col, policy_col, speed_col = st.columns([1, 1, 1], gap="small")

        with mode_col:
            mode_options = ["Train", "Evaluate", "Manual"]
            current_mode = controller.mode.value.title()
            try:
                mode_index = mode_options.index(current_mode)
            except ValueError:
                mode_index = 0

            new_mode = st.selectbox(
                "Mode",
                mode_options,
                index=mode_index,
                key="toolbar_mode",
                help="Train: Agent learns while playing\nEvaluate: Greedy policy, no learning\nManual: You control the snake",
            )

            if new_mode != current_mode:
                mode_map = {
                    "Train": ActivityMode.TRAIN,
                    "Evaluate": ActivityMode.EVALUATE,
                    "Manual": ActivityMode.MANUAL,
                }
                controller.set_mode(mode_map[new_mode])
                controller.set_paused(True)
                st.rerun()

        with policy_col:
            if controller.mode != ActivityMode.MANUAL:
                # Internal policy values (must match TrainingController.advance_tick expectations)
                policy_values = ["Watch", "FastTraining"]
                # Display labels for UI
                policy_labels = ["Watch", "Train fast"]

                current_policy = st.session_state.get("execution_policy", "Watch")
                try:
                    policy_index = policy_values.index(current_policy)
                except ValueError:
                    policy_index = 0

                new_policy = st.selectbox(
                    "Execution",
                    policy_labels,
                    index=policy_index,
                    key="toolbar_policy",
                    help="Watch: Deliberately paced, readable movement\nTrain fast: Multiple transitions per UI update",
                )
                # Map display label back to internal value
                st.session_state.execution_policy = policy_values[policy_labels.index(new_policy)]
            else:
                st.selectbox("Execution", ["Manual"], disabled=True, key="toolbar_policy_manual")

        with speed_col:
            if controller.mode != ActivityMode.MANUAL:
                policy = st.session_state.get("execution_policy", "Watch")
                if policy == "Watch":
                    speed_options = ["Slow", "Normal", "Fast"]
                    current_speed = st.session_state.get("speed_preset", "Normal")
                    try:
                        speed_index = speed_options.index(current_speed)
                    except ValueError:
                        speed_index = 1

                    new_speed = st.selectbox(
                        "Speed",
                        speed_options,
                        index=speed_index,
                        key="toolbar_speed",
                        help="Slow: 0.1s/step\nNormal: 0.05s/step\nFast: 0.01s/step (burst-like)",
                    )
                    st.session_state.speed_preset = new_speed
                    st.session_state.speed = {"Slow": 0.1, "Normal": 0.05, "Fast": 0.01}[new_speed]
                else:
                    st.session_state.speed = 0.0
                    st.selectbox("Speed", ["N/A (burst)"], disabled=True, key="toolbar_speed_burst")
            else:
                st.selectbox("Speed", ["N/A"], disabled=True, key="toolbar_speed_manual")

        # Row 2: Action buttons grouped together
        is_paused = controller.paused
        btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1], gap="small")

        with btn_col1:
            # Start/Resume/Pause Button
            if controller.mode == ActivityMode.TRAIN:
                if is_paused:
                    if controller.agent is None or controller.agent.n_games == 0:
                        btn_label, btn_type = "▶️ Start Training", "primary"
                    else:
                        btn_label, btn_type = "▶️ Resume Training", "primary"
                else:
                    btn_label, btn_type = "⏸️ Pause Training", "secondary"
            elif controller.mode == ActivityMode.EVALUATE:
                if is_paused:
                    if controller.agent is None or controller.agent.n_games == 0:
                        btn_label, btn_type = "▶️ Start Evaluation", "primary"
                    else:
                        btn_label, btn_type = "▶️ Resume Evaluation", "primary"
                else:
                    btn_label, btn_type = "⏸️ Pause Evaluation", "secondary"
            else:  # MANUAL
                if is_paused:
                    btn_label, btn_type = "▶️ Resume Manual", "primary"
                else:
                    btn_label, btn_type = "⏸️ Pause Manual", "secondary"

            if st.button(btn_label, type=btn_type, key="toolbar_play_pause", width="stretch"):
                if is_paused:
                    if controller.game_state is None or controller.game_over:
                        controller.restart_board()
                    controller.set_paused(False)
                else:
                    controller.set_paused(True)
                st.rerun()

        with btn_col2:
            # Step Once (only when paused)
            if is_paused:
                if st.button(
                    "⏭️ Step Once",
                    key="toolbar_step",
                    width="stretch",
                    help="Advance exactly one transition",
                ):
                    _handle_step_once(controller)
                    st.rerun()
            else:
                st.button(
                    "⏭️ Step Once", key="toolbar_step_disabled", width="stretch", disabled=True
                )

        with btn_col3:
            # Restart Board
            if st.button(
                "🔄 Restart Board",
                key="toolbar_restart",
                width="stretch",
                help="Restart current game, keep learned weights and training history",
            ):
                controller.restart_board()
                if "last_decision" in st.session_state:
                    del st.session_state.last_decision
                st.rerun()


def _handle_step_once(controller: TrainingController) -> None:
    """Handle Step Once button press with explicit terminal board behavior."""
    if controller.game_over:
        if controller.mode == ActivityMode.TRAIN:
            # In training, step once on terminal board restarts the board
            controller.restart_board()
        elif controller.mode == ActivityMode.EVALUATE:
            # In evaluation, pause and stay on terminal board
            controller.set_paused(True)
        else:  # MANUAL
            # In manual, restart board for new game
            controller.restart_board()
    else:
        # Normal step
        controller.step()


def render_status_badge_inline(controller: TrainingController) -> str:
    """Render inline status badge HTML."""
    mode = controller.mode
    paused = controller.paused

    if paused:
        status_text = f"{mode.value.title()} · Paused"
        css_class = "sqdn-status-paused"
    else:
        status_text = f"{mode.value.title()} · Running"
        css_class = "sqdn-status-running"

    mode_class = f"sqdn-status-{mode.value}"
    return f'<span class="sqdn-status-badge {mode_class} {css_class}">{status_text}</span>'


# =============================================================================
# Manual Control Buttons (when in Manual mode and paused)
# =============================================================================


def render_manual_controls(controller: TrainingController, snapshot) -> None:
    """Render manual control buttons when in Manual mode and paused."""
    if controller.mode != ActivityMode.MANUAL or not controller.paused or controller.game_over:
        return

    st.caption("Manual Controls")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        if st.button("⬆️ Straight", key="manual_straight", width="stretch"):
            # Straight relative to current direction
            snapshot.game_state.get("direction", 0) if snapshot.game_state else 0
            # Action 0 = straight
            controller.step(action=0)
            st.rerun()
    with mc2:
        if st.button("➡️ Right", key="manual_right", width="stretch"):
            controller.step(action=1)
            st.rerun()
    with mc3:
        if st.button("⬅️ Left", key="manual_left", width="stretch"):
            controller.step(action=2)
            st.rerun()


# =============================================================================
# Sidebar Configuration Forms
# =============================================================================


def render_sidebar_environment(controller: TrainingController) -> None:
    """Render Environment configuration form in sidebar."""
    with st.expander("🌍 Environment", expanded=True):
        with st.form("env_config_form"):
            grid_size = st.slider(
                "Grid Size",
                10,
                30,
                controller.env_config.grid_size,
                key="sidebar_grid_size",
            )
            step_penalty = st.number_input(
                "Step Penalty",
                -1.0,
                0.0,
                controller.env_config.step_penalty,
                0.001,
                format="%.3f",
                key="sidebar_step_penalty",
            )
            food_reward = st.number_input(
                "Food Reward",
                0.0,
                100.0,
                controller.env_config.food_reward,
                1.0,
                key="sidebar_food_reward",
            )
            death_penalty = st.number_input(
                "Death Penalty",
                -100.0,
                0.0,
                controller.env_config.death_penalty,
                1.0,
                key="sidebar_death_penalty",
            )

            if st.form_submit_button("Apply Environment Settings", width="stretch"):
                new_config = EnvConfig(
                    grid_size=grid_size,
                    step_penalty=step_penalty,
                    food_reward=food_reward,
                    death_penalty=death_penalty,
                    render_mode="rgb_array",
                )
                controller.update_env_config(new_config)
                st.success("Environment updated — agent and replay reset")
                st.rerun()


def render_sidebar_agent(controller: TrainingController) -> None:
    """Render Agent Hyperparameters form in sidebar."""
    with st.expander("🧠 Agent Hyperparameters", expanded=True):
        with st.form("agent_config_form"):
            hidden_size = st.selectbox(
                "Hidden Size",
                [128, 256, 512],
                index=[128, 256, 512].index(controller.agent_config.hidden_size),
                key="sidebar_hidden_size",
            )
            lr = st.number_input(
                "Learning Rate",
                1e-5,
                1e-1,
                controller.agent_config.lr,
                1e-4,
                format="%.5f",
                key="sidebar_lr",
            )
            gamma = st.number_input(
                "Gamma",
                0.8,
                0.999,
                controller.agent_config.gamma,
                0.001,
                format="%.3f",
                key="sidebar_gamma",
            )
            epsilon_start = st.number_input(
                "Epsilon Start",
                0.0,
                1.0,
                controller.agent_config.epsilon_start,
                0.01,
                key="sidebar_eps_start",
            )
            epsilon_end = st.number_input(
                "Epsilon End",
                0.0,
                0.5,
                controller.agent_config.epsilon_end,
                0.001,
                format="%.3f",
                key="sidebar_eps_end",
            )
            epsilon_decay = st.number_input(
                "Epsilon Decay",
                0.9,
                0.9999,
                controller.agent_config.epsilon_decay,
                0.0001,
                format="%.4f",
                key="sidebar_eps_decay",
            )
            memory_size = st.selectbox(
                "Memory Size",
                [10000, 50000, 100000, 200000],
                index=[10000, 50000, 100000, 200000].index(controller.agent_config.memory_size),
                key="sidebar_memory_size",
            )
            batch_size = st.selectbox(
                "Batch Size",
                [256, 512, 1000, 2048],
                index=[256, 512, 1000, 2048].index(controller.agent_config.batch_size),
                key="sidebar_batch_size",
            )
            tau = st.number_input(
                "Target Tau (Soft Update)",
                0.001,
                1.0,
                controller.agent_config.tau,
                0.001,
                format="%.3f",
                key="sidebar_tau",
                help="Soft update coefficient for target network. Lower = slower updates.",
            )
            dropout = st.number_input(
                "Dropout",
                0.0,
                0.5,
                controller.agent_config.dropout,
                0.05,
                format="%.2f",
                key="sidebar_dropout",
                help="Dropout rate for hidden layers. 0 = disabled.",
            )
            learning_starts = st.number_input(
                "Learning Starts",
                0,
                100000,
                controller.agent_config.learning_starts,
                500,
                key="sidebar_learning_starts",
                help="Number of transitions before training begins (replay warm-up).",
            )
            train_every = st.number_input(
                "Train Every",
                1,
                100,
                controller.agent_config.train_every,
                1,
                key="sidebar_train_every",
                help="Train every N environment transitions.",
            )
            gradient_steps = st.number_input(
                "Gradient Steps",
                1,
                10,
                controller.agent_config.gradient_steps,
                1,
                key="sidebar_gradient_steps",
                help="Optimizer steps per training call.",
            )

            if st.form_submit_button("Create New Agent", width="stretch", type="primary"):
                new_config = AgentConfig(
                    hidden_size=hidden_size,
                    lr=lr,
                    gamma=gamma,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay=epsilon_decay,
                    memory_size=memory_size,
                    batch_size=batch_size,
                    tau=tau,
                    dropout=dropout,
                    learning_starts=learning_starts,
                    train_every=train_every,
                    gradient_steps=gradient_steps,
                )
                controller.update_agent_config(new_config)
                st.success("New agent created with fresh replay buffer!")
                st.rerun()


def render_sidebar_checkpoints(controller: TrainingController) -> None:
    """Render Checkpoint operations in sidebar."""
    with st.expander("💾 Checkpoints", expanded=False):
        save_path = st.text_input(
            "Checkpoint Path", "checkpoints/snake_dqn.pth", key="sidebar_ckpt_path"
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "💾 Save",
                width="stretch",
                key="sidebar_save",
                help="Save full training state (weights, optimizer, replay, RNG)",
            ):
                if controller.agent:
                    controller.save_checkpoint(save_path)
                    st.success("Checkpoint saved!")
                else:
                    st.error("No agent to save")

        with c2:
            if st.button(
                "📂 Load",
                width="stretch",
                key="sidebar_load",
                help="Load checkpoint — replaces current agent and replay",
            ):
                try:
                    info = controller.load_checkpoint(save_path)
                    st.success(
                        f"Loaded: ep {info.get('episode', 0)}, ε={info.get('epsilon', 1.0):.3f}"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {e}")


def render_sidebar_destructive(controller: TrainingController) -> None:
    """Render destructive operations with clear disclosure."""
    with st.expander("⚠️ Destructive Operations", expanded=False):
        st.warning("These actions cannot be undone. They discard data permanently.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "🗑️ Clear History",
                width="stretch",
                key="sidebar_clear_hist",
                help="Clear displayed charts and statistics, KEEP model weights and replay buffer",
            ):
                controller.clear_history()
                st.success("Training history cleared")
                st.rerun()

        with c2:
            if st.button(
                "🆕 New Training Run",
                width="stretch",
                type="primary",
                key="sidebar_new_run",
                help="Create FRESH agent and replay buffer, DISCARD current model weights",
            ):
                if st.checkbox(
                    "I understand this discards the current model", key="confirm_new_run"
                ):
                    controller.new_run()
                    st.success("New training run started")
                    st.rerun()
                else:
                    st.info("Check the box to confirm")


def render_sidebar_display(controller: TrainingController) -> None:
    """Render Display settings in sidebar."""
    with st.expander("🎨 Display", expanded=False):
        st.session_state.show_activations = st.checkbox(
            "Show Network Activations (Inspect Learning)",
            st.session_state.get("show_activations", False),
            key="sidebar_show_activations",
            help="Enable expensive activation visualization in Inspect Learning section",
        )
        st.session_state.show_grid = st.checkbox(
            "Show Grid Lines",
            st.session_state.get("show_grid", True),
            key="sidebar_show_grid",
        )

        if st.button(
            "🎮 New Game",
            width="stretch",
            key="sidebar_new_game",
            help="Same as Restart Board in toolbar",
        ):
            controller.restart_board()
            st.rerun()


def render_sidebar_help() -> None:
    """Render Help/About in sidebar."""
    with st.expander("ℹ️ Help & Info", expanded=False):
        st.markdown("""
**Snake DQN Dashboard**

**Modes:**
- **Train**: Agent explores (ε-greedy) and learns from experience
- **Evaluate**: Agent acts greedily (ε=0), no learning
- **Manual**: You control the snake via buttons

**Execution Policies (Train/Eval):**
- **Watch**: Fixed delay per step, human-readable
- **FastTraining**: Max throughput, multiple steps per UI frame

**Controls:**
- **Start/Resume**: Begin or continue execution
- **Pause**: Stop execution, keep state
- **Step Once**: Advance exactly one transition (when paused)
- **Restart Board**: New episode, keep learned weights

**Destructive:**
- **Clear History**: Reset statistics only
- **New Run**: Fresh agent, discard weights & replay

**Inspect Learning:**
- State vector (11D): Danger / Direction / Food
- Q-Values: ▲ Executed action, ◆ Greedy action
- Q-gap: Top-two Q-value separation
- Network: Activations recomputed with current weights
""")


# =============================================================================
# Full Sidebar Renderer
# =============================================================================


def render_sidebar(controller: TrainingController) -> None:
    """Render complete sidebar with all configuration sections."""
    with st.sidebar:
        st.markdown("### 🐍 Snake DQN")
        st.caption("Live Training Dashboard")
        st.divider()

        render_sidebar_environment(controller)
        render_sidebar_agent(controller)
        st.divider()
        render_sidebar_checkpoints(controller)
        render_sidebar_destructive(controller)
        st.divider()
        render_sidebar_display(controller)
        render_sidebar_help()
