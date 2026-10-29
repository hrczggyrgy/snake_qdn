"""
Streamlit Application for Snake DQN Training & Visualization.

A comprehensive dashboard with three main views:
1. Live Game Arena - Real-time game rendering and controls
2. Neural Network Internals - State vector, architecture, Q-values
3. Learning Analytics - Scores, loss curves, exploration tracking
"""

import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from training import (
    ActivityMode,
    AgentConfig,
    EnvConfig,
    TrainingController,
    TrainingSnapshot,
)

# Import UI modules
from ui.styles import inject_global_styles, BOARD_SIZE
from ui.controls import (
    render_primary_toolbar,
    render_manual_controls,
    render_sidebar,
    render_status_badge_inline,
)
from ui.game import (
    render_game_board,
    render_decision_strip_with_disclosure,
    render_latest_event,
    render_outcome_summary,
    render_metrics_row,
)
from ui.inspection import render_inspect_learning, render_training_diagnostics
from ui.charts import render_score_chart, render_loss_chart, render_epsilon_chart


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

    # Title and status line
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown("## 🐍 Snake DQN Live Training Dashboard")
    with col_status:
        status_html = render_status_badge_inline(controller)
        st.markdown(status_html, unsafe_allow_html=True)

    # =============================================================================
    # PRIMARY CONTROL TOOLBAR (Single Authority)
    # =============================================================================
    render_primary_toolbar(controller)

    # Manual controls when in Manual mode and paused
    render_manual_controls(controller, controller.get_snapshot())

    # =============================================================================
    # LIVE DASHBOARD FRAGMENT
    # =============================================================================
    # This fragment handles BOTH execution (stepping) AND rendering of all
    # live-updating elements: board, metrics, charts, decision strip, events.
    # This ensures everything stays in sync and refreshes together.
    
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
    def live_dashboard():
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

        # ----- LEFT COLUMN: Game Board + Decision Strip + Latest Event -----
        left_col, right_col = st.columns([3, 2], gap="medium")

        with left_col:
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

            # Status caption
            if ctl.agent is not None and snapshot.game_state:
                mode_icon = {"train": "🧠", "evaluate": "📊", "manual": "👤"}.get(ctl.mode.value, "❓")
                status = "Running" if not ctl.paused else "Paused"
                st.caption(
                    f"{mode_icon} {ctl.mode.value.title()} · {status} · "
                    f"Ep {ctl.agent.n_games} · "
                    f"Score {snapshot.game_state['score']} · "
                    f"Best {ctl.agent.high_score} · "
                    f"ε {ctl.agent.epsilon:.3f}"
                )

            # Last Decision Strip
            st.markdown("---")
            decision_html = render_decision_strip_with_disclosure(snapshot)
            if decision_html:
                st.markdown(decision_html, unsafe_allow_html=True)

            # Latest Event
            if hasattr(snapshot, "recent_events") and snapshot.recent_events:
                event_html = render_latest_event([e.to_dict() for e in snapshot.recent_events])
                if event_html:
                    st.markdown(event_html, unsafe_allow_html=True)

        # ----- RIGHT COLUMN: Score Chart + Episode Outcome Summary -----
        with right_col:
            # Score Chart
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

            # Loss & Epsilon Charts (compact)
            lcol, ecol = st.columns(2, gap="small")
            with lcol:
                fig_loss = render_loss_chart(snapshot.recent_loss, snapshot.optimizer_steps)
                st.plotly_chart(
                    fig_loss,
                    width="stretch",
                    config={"displayModeBar": False},
                    key="live_loss_chart",
                )
            with ecol:
                fig_eps = render_epsilon_chart(snapshot.recent_epsilon, snapshot.episode)
                st.plotly_chart(
                    fig_eps,
                    width="stretch",
                    config={"displayModeBar": False},
                    key="live_epsilon_chart",
                )

    # Render the live dashboard fragment
    live_dashboard()

    # =============================================================================
    # STATIC SECTIONS (below the live fragment)
    # =============================================================================

    # FULL-WIDTH INSPECT LEARNING
    render_inspect_learning(
        controller,
        controller.get_snapshot(),
        show_activations=st.session_state.get("show_activations", False),
    )

    # TRAINING DIAGNOSTICS (Collapsible)
    render_training_diagnostics(controller, controller.get_snapshot())

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()