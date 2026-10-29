"""
Inspect Learning components: State Vector, Q-Values, Network Activations.
Full-width section below the main game/progress row.
"""

import numpy as np
import streamlit as st

from training import TrainingController, TrainingSnapshot
from ui.charts import (
    render_network_activations,
    render_q_values,
    render_state_vector,
)


def render_inspect_learning(
    controller: TrainingController,
    snapshot: TrainingSnapshot,
    show_activations: bool,
    model_revision: int | None = None,
) -> None:
    """Render the full-width Inspect Learning section.

    Args:
        controller: TrainingController instance
        snapshot: Current training snapshot (contains pre-action data)
        show_activations: Whether to show network activations (expensive)
        model_revision: Model revision ID for activation fidelity tracking
    """
    st.markdown("---")
    st.markdown("### 🔬 Inspect Learning")

    # State Vector
    st.caption("State Representation (11D) — grouped by Danger / Direction / Food")
    fig_sv = render_state_vector(snapshot.observation)
    if fig_sv:
        st.plotly_chart(
            fig_sv, width="stretch", config={"displayModeBar": False}, key="inspect_state_vector"
        )
    else:
        st.info("No observation available")

    # Q-Values
    st.caption("Action Q-Values — ▲ Executed  ◆ Preferred (greedy)  Q-gap = top-two separation")
    fig_q = render_q_values(
        q_values=snapshot.q_values,
        action_executed=snapshot.action,
        action_preferred=int(np.argmax(snapshot.q_values))
        if snapshot.q_values is not None
        else None,
        selection_reason=snapshot.selection_reason,
    )
    if fig_q:
        st.plotly_chart(
            fig_q, width="stretch", config={"displayModeBar": False}, key="inspect_q_values"
        )
    else:
        st.info("No Q-values available")

    # Network Activations (gated)
    if show_activations:
        st.caption("Network Activations — recomputed with current weights")
        fig_net = render_network_activations(snapshot, controller, model_revision)
        if fig_net:
            st.plotly_chart(
                fig_net, width="stretch", config={"displayModeBar": False}, key="inspect_network"
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


def render_inspect_learning_simple(
    controller: TrainingController,
    snapshot: TrainingSnapshot,
    show_activations: bool,
) -> None:
    """Simplified version without model_revision tracking."""
    render_inspect_learning(controller, snapshot, show_activations, model_revision=None)


# =============================================================================
# Training Diagnostics
# =============================================================================


def render_training_diagnostics(controller: TrainingController, snapshot: TrainingSnapshot) -> None:
    """Render the lower Training Diagnostics area (collapsible)."""
    from ui.charts import (
        render_bellman_update,
        render_epsilon_chart,
        render_loss_chart,
        render_score_chart,
        render_termination_summary,
        render_throughput_chart,
    )

    st.markdown("---")
    with st.expander("📊 Training Diagnostics", expanded=False):
        # Row 1: Score + Loss
        col1, col2 = st.columns(2)
        with col1:
            fig_score = render_score_chart(
                snapshot.recent_scores,
                snapshot.recent_mean_scores,
                snapshot.episode,
            )
            st.plotly_chart(
                fig_score, width="stretch", config={"displayModeBar": False}, key="diag_scores"
            )

        with col2:
            fig_loss = render_loss_chart(
                snapshot.recent_loss,
                snapshot.optimizer_steps,
            )
            st.plotly_chart(
                fig_loss, width="stretch", config={"displayModeBar": False}, key="diag_loss"
            )

        # Row 2: Epsilon + Throughput
        col1, col2 = st.columns(2)
        with col1:
            fig_eps = render_epsilon_chart(snapshot.recent_epsilon, snapshot.episode)
            st.plotly_chart(
                fig_eps, width="stretch", config={"displayModeBar": False}, key="diag_epsilon"
            )

        with col2:
            # Throughput - placeholder for now, could be enhanced with history
            metrics = controller.get_metrics()
            tps_history = getattr(controller, "_tps_history", [])
            ops_history = getattr(controller, "_ops_history", [])
            steps_history = getattr(controller, "_throughput_steps", [])

            if tps_history:
                fig_tp = render_throughput_chart(tps_history, ops_history, steps_history)
                st.plotly_chart(
                    fig_tp, width="stretch", config={"displayModeBar": False}, key="diag_throughput"
                )
            else:
                # Show current metrics as static display
                st.markdown("**Current Throughput**")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Transitions/s", f"{metrics.get('transitions_per_sec', 0):.1f}")
                with m2:
                    st.metric("Opt Updates/s", f"{metrics.get('optimizer_updates_per_sec', 0):.1f}")

        # Row 3: Episode Termination Summary
        if controller.agent and hasattr(controller.agent, "termination_history"):
            term_counts = controller.agent.termination_history
            if term_counts:
                recent_terms = list(controller.agent.recent_terminations) if hasattr(controller.agent, "recent_terminations") else None
                fig_term = render_termination_summary(term_counts, window=50, recent_terminations=recent_terms)
                if fig_term:
                    st.plotly_chart(
                        fig_term,
                        width="stretch",
                        config={"displayModeBar": False},
                        key="diag_termination",
                    )

        # Row 4: Bellman Update Explanation
        if snapshot.bellman_details is not None:
            st.markdown("---")
            st.caption("**Bellman Update** — sampled from replay (may differ from visible board)")
            fig_bellman = render_bellman_update(snapshot.bellman_details)
            if fig_bellman:
                st.plotly_chart(
                    fig_bellman, width="stretch", config={"displayModeBar": False}, key="diag_bellman"
                )

        # Row 5: Event History (if available)
        if hasattr(st.session_state, "event_history") and st.session_state.event_history:
            with st.expander("📜 Event History", expanded=False):
                for event in reversed(st.session_state.event_history[-50:]):
                    event_type = event.get("type", "info")
                    icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(
                        event_type, "•"
                    )
                    st.caption(f"{icon} {event.get('message', '')}")



