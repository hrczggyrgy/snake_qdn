"""
Streamlit Application for Snake DQN Training & Visualization.

A comprehensive dashboard with three main views:
1. Live Game Arena - Real-time game rendering and controls
2. Neural Network Internals - State vector, architecture, Q-values
3. Learning Analytics - Scores, loss curves, exploration tracking
"""

import logging

import streamlit as st

from training import ActivityMode, TrainingController
from ui.charts import render_score_chart
from ui.controls import (
    render_manual_controls,
    render_primary_toolbar,
    render_sidebar,
    render_status_badge_inline,
)
from ui.game import (
    render_decision_strip_with_disclosure,
    render_game_board,
    render_latest_event,
    render_metrics_row,
    render_outcome_summary,
)
from ui.inspection import render_inspect_learning, render_training_diagnostics
from ui.styles import BOARD_SIZE, inject_global_styles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Streamlit Version Compatibility
# =============================================================================
try:
    _ST_VERSION = tuple(map(int, st.__version__.split(".")[:2]))
except Exception:
    _ST_VERSION = (1, 28)


def btn_full(label, **kwargs):
    """Button with full-width compatibility across Streamlit versions."""
    return st.button(label, width="stretch", **kwargs)


# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Snake DQN - Live Training Dashboard",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global styles
inject_global_styles()


# =============================================================================
# Session State - TrainingController
# =============================================================================
def get_controller() -> TrainingController:
    """Get or create the TrainingController from session state."""
    if "controller" not in st.session_state:
        st.session_state.controller = TrainingController()
    return st.session_state.controller


def init_ui_state():
    """Initialize UI-only session state (not managed by controller)."""
    defaults = {
        "speed": 0.1,
        "speed_preset": "Watch",
        "execution_policy": "Watch",
        "show_activations": False,
        "show_grid": True,
        "max_burst_steps": 50,
        "burst_time_budget": 0.05,
        # Event tracking
        "last_event_id": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_ui_state()


# =============================================================================
# Fragment 1: Execution + Game Board (Game Frequency: 50-100ms)
# =============================================================================
def create_execution_fragment():
    """Create the execution fragment that owns advance_tick() and renders the game board.
    
    This is the ONLY fragment that calls advance_tick(). It runs at game frequency
    (50ms Watch, 100ms FastTraining) and renders the game board, decision strip,
    latest event, and status badge.
    """
    controller = get_controller()

    running = (
        not controller.paused
        and controller.mode != ActivityMode.MANUAL
    )

    # For Watch mode, use a reasonable minimum interval (not 0)
    # FastTraining uses burst logic inside the fragment
    if running:
        if st.session_state.get("execution_policy", "Watch") == "FastTraining":
            interval = 0.1  # Refresh UI every 100ms, burst inside fragment
        else:
            interval = max(0.05, float(st.session_state.get("speed", 0.1)))
    else:
        interval = None

    @st.fragment(run_every=interval)
    def execution_fragment():
        ctl = get_controller()

        # =========================================================================
        # EXECUTION: Single call to controller.advance_tick()
        # =========================================================================
        try:
            if not ctl.paused and ctl.mode != ActivityMode.MANUAL:
                policy = st.session_state.get("execution_policy", "Watch")
                max_steps = st.session_state.get("max_burst_steps", 50)
                time_budget = st.session_state.get("burst_time_budget", 0.05)

                ctl.advance_tick(
                    execution_policy=policy,
                    max_burst_steps=max_steps,
                    burst_time_budget=time_budget,
                )
        except Exception:
            logger.exception(
                "Live execution failed: mode=%s paused=%s game_over=%s episode=%s",
                ctl.mode.value,
                ctl.paused,
                ctl.game_over,
                ctl.episode,
            )
            raise

        # =========================================================================
        # RENDERING: Get fresh snapshot AFTER execution
        # =========================================================================
        snapshot = ctl.get_snapshot()

        # ----- STATUS BADGE (updates with fragment) -----
        status_html = render_status_badge_inline(ctl)
        st.markdown(status_html, unsafe_allow_html=True)

        # ----- MAIN ROW: Game Board (left) + Progress Chart (right) -----
        # Game board gets ~40%, progress chart gets ~60%
        game_col, progress_col = st.columns([1, 1.5], gap="medium")

        with game_col:
            # Game Board
            if snapshot.game_state:
                fig = render_game_board(snapshot, show_grid=st.session_state.get("show_grid", True))
                st.plotly_chart(
                    fig,
                    width=BOARD_SIZE,
                    height=BOARD_SIZE,
                    config={"displayModeBar": False, "staticPlot": True, "responsive": False},
                    key="live_game_board",
                )

            # Last Decision Strip (compact)
            decision_html = render_decision_strip_with_disclosure(snapshot)
            if decision_html:
                st.markdown(decision_html, unsafe_allow_html=True)

            # Latest Event (single line)
            if hasattr(snapshot, "recent_events") and snapshot.recent_events:
                event_html = render_latest_event([e.to_dict() for e in snapshot.recent_events])
                if event_html:
                    st.markdown(event_html, unsafe_allow_html=True)

        with progress_col:
            # Score Chart (main progress indicator) - rebuilt but only updates on episode changes
            fig_score = render_score_chart(
                snapshot.recent_scores,
                snapshot.recent_mean_scores,
                snapshot.episode,
            )
            st.plotly_chart(
                fig_score,
                width="stretch",
                config={"displayModeBar": False},
                key="live_score_chart",
            )

            # Episode Outcome Summary
            if ctl.agent and hasattr(ctl.agent, "termination_history"):
                outcome_html = render_outcome_summary(ctl.agent.termination_history, window=50)
                if outcome_html:
                    st.markdown(outcome_html, unsafe_allow_html=True)

    # Render the fragment
    execution_fragment()


# =============================================================================
# Fragment 2: Metrics Row (500ms cadence)
# =============================================================================
@st.fragment(run_every=0.5)
def metrics_fragment():
    """Render the metrics row at 500ms cadence (read-only)."""
    ctl = get_controller()
    render_metrics_row(ctl)


# =============================================================================
# Fragment 3: Inspect Learning (300-500ms cadence)
# =============================================================================
@st.fragment(run_every=0.5)
def inspect_learning_fragment():
    """Render the Inspect Learning section at 500ms cadence (read-only).
    
    Gets a fresh coherent snapshot on each refresh. Uses the retained
    last-decision record from the snapshot (pre-action data).
    """
    controller = get_controller()
    snapshot = controller.get_snapshot()

    st.markdown("---")
    st.markdown("### 🔬 Inspect Learning")

    # State Vector
    st.caption("State Representation (11D) — grouped by Danger / Direction / Food")
    from ui.charts import render_state_vector
    fig_sv = render_state_vector(snapshot.observation)
    if fig_sv:
        st.plotly_chart(
            fig_sv,
            width="stretch",
            height=360,  # Explicit element height matching figure height
            config={"displayModeBar": False},
            key="inspect_state_vector",
        )
    else:
        st.info("No observation available")

    # Q-Values
    st.caption("Action Q-Values — ▲ Executed  ◆ Preferred (greedy)  Q-gap = top-two separation")
    from ui.charts import render_q_values
    import numpy as np
    fig_q = render_q_values(
        q_values=snapshot.q_values,
        action_executed=snapshot.action,
        action_preferred=int(np.argmax(snapshot.q_values)) if snapshot.q_values is not None else None,
        selection_reason=snapshot.selection_reason,
    )
    if fig_q:
        st.plotly_chart(
            fig_q,
            width="stretch",
            height=320,  # Explicit element height matching figure height
            config={"displayModeBar": False},
            key="inspect_q_values",
        )
    else:
        st.info("No Q-values available")

    # Network Activations (gated)
    if st.session_state.get("show_activations", False):
        st.caption("Network Activations — recomputed with current weights")
        from ui.charts import render_network_activations
        fig_net = render_network_activations(snapshot, controller)
        if fig_net:
            st.plotly_chart(
                fig_net,
                width="stretch",
                height=360,  # Explicit element height matching figure height
                config={"displayModeBar": False},
                key="inspect_network",
            )
        else:
            st.info("Network activations unavailable")
    else:
        # Show a placeholder with activation toggle
        if controller.agent is not None:
            if st.button("🔬 Show Network Activations", key="toggle_activations", width="stretch"):
                st.session_state.show_activations = True
                st.rerun()
        else:
            st.caption("Network activations unavailable (no agent)")


# =============================================================================
# Main Application
# =============================================================================
def main():
    controller = get_controller()

    # Render Sidebar (Configuration only - no primary controls)
    render_sidebar(controller)

    # =============================================================================
    # PAGE HEADER
    # =============================================================================
    st.markdown('<div class="sqdn-page">', unsafe_allow_html=True)

    # Title
    st.markdown("## 🐍 Snake DQN Live Training Dashboard")

    # =============================================================================
    # PRIMARY CONTROL TOOLBAR (Single Authority)
    # =============================================================================
    render_primary_toolbar(controller)

    # Manual controls when in Manual mode and paused
    render_manual_controls(controller, controller.get_snapshot())

    # =============================================================================
    # FRAGMENT 1: Execution + Game Board (Game Frequency)
    # =============================================================================
    create_execution_fragment()

    # =============================================================================
    # FRAGMENT 2: Metrics Row (500ms)
    # =============================================================================
    metrics_fragment()

    # =============================================================================
    # FRAGMENT 3: Inspect Learning (500ms)
    # =============================================================================
    inspect_learning_fragment()

    # =============================================================================
    # TRAINING DIAGNOSTICS (Collapsible) - outside fragments, renders on interaction
    # Contains loss/epsilon/replay details - no auto-refresh needed
    # =============================================================================
    st.markdown("---")
    render_training_diagnostics(controller, controller.get_snapshot())

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()