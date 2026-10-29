"""
Game rendering components: board, decision strip, event feed, outcome summary.
"""

import plotly.graph_objects as go
import streamlit as st

from training import TrainingSnapshot
from ui.styles import (
    ACTION_NAMES,
    ACTION_NAMES_SHORT,
    BOARD_MARGIN,
    BOARD_SIZE,
)

# =============================================================================
# Game Board (Plotly Coordinate Renderer)
# =============================================================================


def render_game_board(snapshot: TrainingSnapshot, show_grid: bool = True) -> go.Figure:
    """Render the game board as a fixed-size Plotly figure from snapshot."""
    game_state = snapshot.game_state
    if not game_state:
        fig = go.Figure()
        fig.update_layout(
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            margin=BOARD_MARGIN,
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            xaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
                "range": [0, 20],
                "fixedrange": True,
            },
            yaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
                "range": [20, 0],
                "fixedrange": True,
                "scaleanchor": "x",
                "scaleratio": 1,
            },
        )
        return fig

    grid_size = game_state["grid_size"]
    snake = game_state["snake"]
    food = game_state["food"]
    direction = game_state["direction"]
    game_over = game_state["game_over"]

    fig = go.Figure()

    # Grid background
    if show_grid:
        for i in range(grid_size + 1):
            fig.add_shape(
                type="line",
                x0=0,
                y0=i,
                x1=grid_size,
                y1=i,
                line={"color": "rgba(100,100,100,0.2)", "width": 0.5},
            )
            fig.add_shape(
                type="line",
                x0=i,
                y0=0,
                x1=i,
                y1=grid_size,
                line={"color": "rgba(100,100,100,0.2)", "width": 0.5},
            )

    # Snake body
    for i, (x, y) in enumerate(snake):
        color = "#32CD32" if i == 0 else "#228B22"
        opacity = 1.0 if i == 0 else max(0.4, 1.0 - i * 0.03)

        fig.add_shape(
            type="rect",
            x0=x,
            y0=y,
            x1=x + 1,
            y1=y + 1,
            fillcolor=color,
            opacity=opacity,
            line={"color": "#1a5c1a", "width": 1},
            layer="above",
        )

        # Eyes on head
        if i == 0:
            eye_size = 0.2
            if direction == 0:  # Right
                eyes = [(0.65, 0.2), (0.65, 0.6)]
            elif direction == 2:  # Left
                eyes = [(0.15, 0.2), (0.15, 0.6)]
            elif direction == 3:  # Up
                eyes = [(0.2, 0.15), (0.6, 0.15)]
            else:  # Down
                eyes = [(0.2, 0.65), (0.6, 0.65)]

            for ex, ey in eyes:
                fig.add_shape(
                    type="circle",
                    x0=x + ex,
                    y0=y + ey,
                    x1=x + ex + eye_size,
                    y1=y + ey + eye_size,
                    fillcolor="white",
                    line={"width": 0},
                    layer="above",
                )

    # Food
    if food is not None:
        fx, fy = food
        fig.add_shape(
            type="circle",
            x0=fx + 0.1,
            y0=fy + 0.1,
            x1=fx + 0.9,
            y1=fy + 0.9,
            fillcolor="#FF4444",
            line={"color": "#CC0000", "width": 2},
            layer="above",
        )
        fig.add_shape(
            type="circle",
            x0=fx + 0.2,
            y0=fy + 0.2,
            x1=fx + 0.4,
            y1=fy + 0.4,
            fillcolor="#FFAAAA",
            line={"width": 0},
            layer="above",
        )

    # Game over overlay
    if game_over:
        fig.add_annotation(
            x=grid_size / 2,
            y=grid_size / 2,
            text="GAME OVER",
            showarrow=False,
            font={"size": 28, "color": "red", "family": "Arial Black"},
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="red",
            borderwidth=2,
            borderpad=8,
        )

    fig.update_layout(
        xaxis={
            "range": [0, grid_size],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "autorange": False,
            "fixedrange": True,
            "constrain": "domain",
        },
        yaxis={
            "range": [grid_size, 0],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "autorange": False,
            "fixedrange": True,
            "constrain": "domain",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        width=BOARD_SIZE,
        height=BOARD_SIZE,
        margin=BOARD_MARGIN,
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font={"color": "white"},
        showlegend=False,
        autosize=False,
    )
    return fig


def render_game_board_fragment(controller, show_grid: bool, key: str = "live_game_board"):
    """Render game board inside a fragment for auto-refresh.

    This should be called inside an @st.fragment decorated function.
    """
    snapshot = controller.get_snapshot()
    fig = render_game_board(snapshot, show_grid)
    st.plotly_chart(
        fig,
        width=BOARD_SIZE,
        height=BOARD_SIZE,
        config={"displayModeBar": False, "staticPlot": True, "responsive": False},
        key=key,
    )

    # Status caption
    if controller.agent and snapshot.game_state:
        mode_icon = {"train": "🧠", "evaluate": "📊", "manual": "👤"}.get(
            controller.mode.value, "❓"
        )
        status = "Running" if not controller.paused else "Paused"
        st.caption(
            f"{mode_icon} {controller.mode.value.title()} · {status} · "
            f"Ep {controller.agent.n_games} · "
            f"Score {snapshot.game_state['score']} · "
            f"Best {controller.agent.high_score} · "
            f"ε {controller.agent.epsilon:.3f}"
        )


# =============================================================================
# Decision Strip
# =============================================================================


def render_decision_strip(snapshot: TrainingSnapshot) -> str | None:
    """Render the compact 'Last decision' strip as HTML.

    Uses pre-action data from snapshot (pre_observation, pre_q_values, etc.)
    Falls back to current snapshot values if pre-action not available.
    """
    # Use pre-action data if available (set by TrainingController.step)
    q_values = snapshot.q_values
    action = snapshot.action
    reason = snapshot.selection_reason
    epsilon = snapshot.epsilon

    if q_values is None or action is None:
        return None

    action_name = ACTION_NAMES[action]

    if reason == "exploring":
        label = "🎲 Exploration"
        detail = f"{action_name} chosen randomly (ε={epsilon:.3f})"
        color = "#4488ff"
        reason_class = "sqdn-reason-exploring"
    elif reason == "manual":
        label = "👤 Manual"
        detail = f"{action_name} chosen by user"
        color = "#ffd700"
        reason_class = "sqdn-reason-manual"
    elif reason == "exploiting":
        label = "🎯 Exploiting"
        q_gap = abs(sorted(q_values)[-1] - sorted(q_values)[-2]) if len(q_values) >= 2 else 0
        detail = f"{action_name} = argmax Q (Q-gap: {q_gap:.3f})"
        color = "#44cc44"
        reason_class = "sqdn-reason-exploiting"
    else:
        label = "❓ Unknown"
        detail = f"{action_name} (reason: {reason or 'unknown'})"
        color = "#888888"
        reason_class = "sqdn-reason-unknown"

    # Q-values mini display
    q_parts = "  ".join(
        f"{ACTION_NAMES_SHORT[i]}: {v:+.2f}" + (" ●" if i == action else "")
        for i, v in enumerate(q_values)
    )

    html = f"""
    <div class="sqdn-decision-strip" style="border-left: 3px solid {color};">
        <div class="sqdn-decision-strip-row">
            <div style="min-width: 0;">
                <span class="sqdn-label {reason_class}">{label}</span>
                <span class="sqdn-detail">{detail}</span>
            </div>
            <div class="sqdn-decision-q-values">
                Q: {q_parts}
            </div>
        </div>
    </div>
    """
    return html


def render_decision_strip_with_disclosure(snapshot: TrainingSnapshot) -> str | None:
    """Render decision strip with disclosure notice when board shows post-action state."""
    strip_html = render_decision_strip(snapshot)
    if not strip_html:
        return None

    # Add disclosure when game_over or when board shows result of action
    disclosure = """
    <div class="sqdn-disclosure">
        Inspection describes the last action; the board shows its result.
    </div>
    """
    return strip_html + disclosure


# =============================================================================
# Event Feed
# =============================================================================

EVENT_ICONS = {
    "info": "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "food": "🍎",
    "episode_end": "🏁",
    "replay": "🔄",
    "record": "🏆",
}

EVENT_COLORS = {
    "info": "#4488ff",
    "success": "#44cc44",
    "warning": "#ffd700",
    "error": "#ff4444",
    "food": "#ff4444",
    "episode_end": "#ff8800",
    "replay": "#aa44ff",
    "record": "#ffd700",
}

EVENT_CLASSES = {
    "info": "sqdn-event-info",
    "success": "sqdn-event-success",
    "warning": "sqdn-event-warning",
    "error": "sqdn-event-error",
    "food": "sqdn-event-success",
    "episode_end": "sqdn-event-warning",
    "replay": "sqdn-event-info",
    "record": "sqdn-event-success",
}


def format_event(event: dict) -> str:
    """Format a single event dict as HTML."""
    event_type = event.get("type", "info")
    icon = EVENT_ICONS.get(event_type, "•")
    color = EVENT_COLORS.get(event_type, "#888")
    css_class = EVENT_CLASSES.get(event_type, "sqdn-event-info")
    message = event.get("message", "")
    event_id = event.get("id", "")

    return f"""
    <div class="sqdn-event {css_class}" data-event-id="{event_id}">
        <span class="sqdn-event-icon" style="color: {color};">{icon}</span>
        <span class="sqdn-event-message">{message}</span>
    </div>
    """


def render_event_feed(events: list[dict], max_events: int = 5) -> str | None:
    """Render the event feed as HTML."""
    if not events:
        return None

    html_parts = ['<div class="sqdn-event-feed">']
    for event in events[-max_events:]:
        html_parts.append(format_event(event))
    html_parts.append("</div>")
    return "\n".join(html_parts)


def render_latest_event(events: list[dict]) -> str | None:
    """Render only the latest meaningful event in a stable-height area."""
    if not events:
        return None

    # Filter out low-priority events for the "latest" slot
    priority_order = [
        "record",
        "episode_end",
        "replay",
        "food",
        "warning",
        "success",
        "info",
        "error",
    ]
    filtered = [e for e in events if e.get("type") in priority_order]
    if not filtered:
        filtered = events

    latest = filtered[-1]
    return format_event(latest)


# =============================================================================
# Episode Outcome Summary (Horizontal Stacked Bar)
# =============================================================================


def render_outcome_summary(termination_counts: dict[str, int], window: int = 50) -> str | None:
    """Render horizontal stacked bar of recent episode outcomes.

    Args:
        termination_counts: Dict with keys wall, self, timeout, victory
        window: Maximum episodes in the window (denominator is actual completed)
    """
    if not termination_counts or sum(termination_counts.values()) == 0:
        return None

    colors = {
        "wall": "#ff4444",
        "self": "#ff8800",
        "timeout": "#ffd700",
        "victory": "#44cc44",
    }
    labels = {
        "wall": "Wall",
        "self": "Self",
        "timeout": "Timeout",
        "victory": "Victory",
    }

    completed = sum(termination_counts.values())
    denominator = max(completed, window) if completed >= window else completed

    # Build stacked bar segments
    segments = []
    for reason in ["wall", "self", "timeout", "victory"]:
        count = termination_counts.get(reason, 0)
        if count > 0:
            pct = count / denominator * 100
            segments.append(
                f'<div class="sqdn-outcome-segment" style="background: {colors[reason]}; width: {pct}%;">'
                f"{labels[reason]}: {count}"
                f"</div>"
            )

    # Remaining (incomplete to fill window)
    remaining = max(0, window - completed)
    if remaining > 0:
        pct = remaining / denominator * 100
        segments.append(
            f'<div class="sqdn-outcome-segment" style="background: rgba(100,100,100,0.3); width: {pct}%;">'
            f"Pending: {remaining}"
            f"</div>"
        )

    bar_html = "".join(segments)

    # Legend
    legend_items = []
    for reason in ["wall", "self", "timeout", "victory"]:
        count = termination_counts.get(reason, 0)
        if count > 0:
            legend_items.append(
                f'<span class="sqdn-outcome-legend-item">'
                f'<span class="sqdn-outcome-legend-swatch" style="background: {colors[reason]};"></span>'
                f"{labels[reason]}: {count}"
                f"</span>"
            )
    if remaining > 0:
        legend_items.append(
            f'<span class="sqdn-outcome-legend-item">'
            f'<span class="sqdn-outcome-legend-swatch" style="background: rgba(100,100,100,0.3);"></span>'
            f"Pending: {remaining}"
            f"</span>"
        )

    legend_html = "".join(legend_items)

    html = f"""
    <div class="sqdn-outcome-summary">
        <div class="sqdn-outcome-header">Recent Episode Outcomes (last {window})</div>
        <div class="sqdn-outcome-bar">{bar_html}</div>
        <div class="sqdn-outcome-legend">{legend_html}</div>
        <div style="font-size: 0.75rem; color: var(--sqdn-text-muted); margin-top: 0.25rem;">
            Completed: {completed} · Showing termination reasons for completed episodes
        </div>
    </div>
    """
    return html


# =============================================================================
# Metrics Row
# =============================================================================


def render_metrics_row(controller) -> None:
    """Render compact summary metrics row (5 key metrics)."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Episode", f"{controller.episode}")
    with col2:
        st.metric("Score", f"{controller.score}")
    with col3:
        st.metric("Best Score", f"{controller.high_score}")
    with col4:
        st.metric(
            "Avg (50)",
            f"{controller.agent.mean_scores_history[-1] if controller.agent and controller.agent.mean_scores_history else 0:.1f}",
        )
    with col4:
        st.metric("ε (Epsilon)", f"{controller.epsilon:.3f}")
