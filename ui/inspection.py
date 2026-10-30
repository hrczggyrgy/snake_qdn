"""
Training Diagnostics components: Loss, epsilon, replay, Bellman update.
Full-width section below the main game/progress row.
"""

# =============================================================================
# Training Diagnostics
# =============================================================================


def render_training_diagnostics(controller: "TrainingController", snapshot: "TrainingSnapshot") -> None:
    """Render the lower Training Diagnostics area (collapsible)."""
    from training import TrainingController, TrainingSnapshot
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

        # Row 5: Event History (if available)
        if hasattr(st.session_state, "event_history") and st.session_state.event_history:
            with st.expander("📜 Event History", expanded=False):
                for event in reversed(st.session_state.event_history[-50:]):
                    event_type = event.get("type", "info")
                    icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}.get(
                        event_type, "•"
                    )
                    st.caption(f"{icon} {event.get('message', '')}")



