"""
Streamlit Application for Snake DQN Training & Visualization.

A comprehensive dashboard with three main views:
1. Live Game Arena - Real-time game rendering and controls
2. Neural Network Internals - State vector, architecture, Q-values
3. Learning Analytics - Scores, loss curves, exploration tracking
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

# PIL for inline PNG encoding (fallback)
try:
    import importlib.util

    _PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
except ImportError:
    _PIL_AVAILABLE = False

# Import our modules
from training import (
    ActivityMode,
    AgentConfig,
    EnvConfig,
    TrainingController,
    TrainingSnapshot,
)

# ============================================================================
# Streamlit Version Compatibility
# ============================================================================
try:
    _ST_VERSION = tuple(map(int, st.__version__.split(".")[:2]))
except Exception:
    _ST_VERSION = (1, 28)


def btn_full(label, **kwargs):
    """Button with full-width compatibility across Streamlit versions."""
    return st.button(label, width="stretch", **kwargs)


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Snake DQN - Live Training Dashboard",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.75rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.25rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.4rem 0.8rem;
    }
    .stMetric {
        background: rgba(30, 30, 46, 0.8);
        padding: 0.5rem;
        border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .game-container {
        border: 2px solid #444;
        border-radius: 8px;
        overflow: hidden;
        background: #1e1e2e;
    }
    .decision-strip {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .event-message {
        background: rgba(30, 30, 46, 0.9);
        border: 1px solid #444;
        border-radius: 6px;
        padding: 0.4rem 0.75rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    .outcome-summary {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stPlotlyChart"] > div {
        background: transparent !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# Session State - TrainingController
# ============================================================================
def get_controller() -> TrainingController:
    """Get or create the TrainingController from session state."""
    if "controller" not in st.session_state:
        st.session_state.controller = TrainingController()
    return st.session_state.controller


def init_ui_state():
    """Initialize UI-only session state (not managed by controller)."""
    defaults = {
        "speed": 50,
        "speed_preset": "Watch",
        "activity": "Train",
        "execution_policy": "Watch",
        "show_network": False,
        "show_grid": True,
        "max_burst_steps": 50,
        "burst_time_budget": 0.05,
        # Event tracking
        "last_event_id": 0,
        "pending_events": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_ui_state()


# ============================================================================
# Helper Functions - Sync UI to Controller
# ============================================================================
def sync_activity_to_controller():
    """Sync UI activity selection to controller mode."""
    controller = get_controller()
    activity_map = {
        "Train": ActivityMode.TRAIN,
        "Evaluate": ActivityMode.EVALUATE,
        "Manual": ActivityMode.MANUAL,
    }
    controller.set_mode(activity_map[st.session_state.activity])
    st.session_state.training = st.session_state.activity == "Train"


def sync_paused_to_controller():
    """Sync UI paused state to controller."""
    controller = get_controller()
    controller.set_paused(st.session_state.paused)


def push_event(message: str, event_type: str = "info"):
    """Add an event message to the pending events queue."""
    st.session_state.last_event_id += 1
    st.session_state.pending_events.append(
        {
            "id": st.session_state.last_event_id,
            "message": message,
            "type": event_type,
        }
    )
    # Keep only last 10 events
    if len(st.session_state.pending_events) > 10:
        st.session_state.pending_events = st.session_state.pending_events[-10:]


def check_new_events(controller: TrainingController, snapshot: TrainingSnapshot):
    """Check for new events based on controller state changes."""
    # Check for food eaten (score increased)
    if hasattr(st.session_state, "_last_score"):
        if snapshot.score > st.session_state._last_score:
            push_event(f"Food eaten! Score: {snapshot.score}", "success")
    st.session_state._last_score = snapshot.score

    # Check for episode completion
    if hasattr(st.session_state, "_last_episode"):
        if snapshot.episode > st.session_state._last_episode:
            if snapshot.game_over:
                reason = (
                    "board completed" if snapshot.game_state.get("food") is None else "collision"
                )
                push_event(
                    f"Episode {snapshot.episode} ended: {reason} · Score: {snapshot.score}",
                    "warning",
                )
    st.session_state._last_episode = snapshot.episode

    # Check for new high score
    if hasattr(st.session_state, "_last_high_score"):
        if snapshot.high_score > st.session_state._last_high_score:
            push_event(f"New record! {snapshot.high_score}", "success")
    st.session_state._last_high_score = snapshot.high_score

    # Check replay warmup completion
    if hasattr(st.session_state, "_replay_warmed_up"):
        if (
            not st.session_state._replay_warmed_up
            and controller.agent
            and controller.agent.memory.is_ready(controller.agent.batch_size)
        ):
            push_event("Replay buffer ready — training started", "info")
            st.session_state._replay_warmed_up = True
    else:
        st.session_state._replay_warmed_up = False


def render_game_grid(snapshot: TrainingSnapshot, show_grid: bool = True):
    """Render the game as a compact Plotly figure from snapshot."""
    game_state = snapshot.game_state
    if not game_state:
        fig = go.Figure()
        fig.update_layout(
            width=380,
            height=380,
            margin={"l": 5, "r": 5, "t": 5, "b": 5},
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
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
            if direction == 0:  # Right
                fig.add_shape(
                    type="circle",
                    x0=x + 0.65,
                    y0=y + 0.2,
                    x1=x + 0.85,
                    y1=y + 0.4,
                    fillcolor="white",
                    line={"width": 0},
                )
                fig.add_shape(
                    type="circle",
                    x0=x + 0.65,
                    y0=y + 0.6,
                    x1=x + 0.85,
                    y1=y + 0.8,
                    fillcolor="white",
                    line={"width": 0},
                )
            elif direction == 2:  # Left
                fig.add_shape(
                    type="circle",
                    x0=x + 0.15,
                    y0=y + 0.2,
                    x1=x + 0.35,
                    y1=y + 0.4,
                    fillcolor="white",
                    line={"width": 0},
                )
                fig.add_shape(
                    type="circle",
                    x0=x + 0.15,
                    y0=y + 0.6,
                    x1=x + 0.35,
                    y1=y + 0.8,
                    fillcolor="white",
                    line={"width": 0},
                )
            elif direction == 3:  # Up
                fig.add_shape(
                    type="circle",
                    x0=x + 0.2,
                    y0=y + 0.15,
                    x1=x + 0.4,
                    y1=y + 0.35,
                    fillcolor="white",
                    line={"width": 0},
                )
                fig.add_shape(
                    type="circle",
                    x0=x + 0.6,
                    y0=y + 0.15,
                    x1=x + 0.8,
                    y1=y + 0.35,
                    fillcolor="white",
                    line={"width": 0},
                )
            else:  # Down
                fig.add_shape(
                    type="circle",
                    x0=x + 0.2,
                    y0=y + 0.65,
                    x1=x + 0.4,
                    y1=y + 0.85,
                    fillcolor="white",
                    line={"width": 0},
                )
                fig.add_shape(
                    type="circle",
                    x0=x + 0.6,
                    y0=y + 0.65,
                    x1=x + 0.8,
                    y1=y + 0.85,
                    fillcolor="white",
                    line={"width": 0},
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
        },
        yaxis={
            "range": [0, grid_size],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        width=380,
        height=380,
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font={"color": "white"},
        showlegend=False,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def render_decision_strip(snapshot: TrainingSnapshot):
    """Render the compact 'Last decision' strip."""
    if snapshot.action is None or snapshot.q_values is None:
        return None

    action_names = ["Straight", "Right", "Left"]
    action_name = action_names[snapshot.action]
    reason = snapshot.selection_reason or "unknown"
    epsilon = snapshot.epsilon

    if reason == "exploring":
        label = "🎲 Exploration"
        detail = f"{action_name} chosen randomly (ε={epsilon:.3f})"
        color = "#4488FF"
    elif reason == "manual":
        label = "👤 Manual"
        detail = f"{action_name} chosen by user"
        color = "#44FF88"
    else:
        label = "🎯 Exploiting"
        q_gap = (
            abs(sorted(snapshot.q_values)[-1] - sorted(snapshot.q_values)[-2])
            if len(snapshot.q_values) >= 2
            else 0
        )
        detail = f"{action_name} = argmax Q (gap: {q_gap:.3f})"
        color = "#44FF44"

    # Q-values mini display
    action_names_short = ["S", "R", "L"]
    q_parts = "  ".join(
        f"{action_names_short[i]}: {v:+.2f}" + (" ●" if i == snapshot.action else "")
        for i, v in enumerate(snapshot.q_values)
    )

    html = f"""
    <div class="decision-strip" style="border-left: 3px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="color: {color}; font-weight: 600;">{label}</span>
                <span style="color: #aaa; margin-left: 0.5rem;">{detail}</span>
            </div>
            <div style="color: #888; font-size: 0.8rem;">
                Q: {q_parts}
            </div>
        </div>
    """
    return html


def render_event_feed():
    """Render the brief event feed."""
    events = st.session_state.pending_events
    if not events:
        return None

    html_parts = []
    for event in events[-5:]:  # Show last 5
        color = {"info": "#4488FF", "success": "#44FF88", "warning": "#FFD700"}.get(
            event["type"], "#888"
        )
        icon = {"info": "ℹ️", "success": "✅", "warning": "⚠️"}.get(event["type"], "•")
        html_parts.append(
            f'<div class="event-message" style="border-left: 3px solid {color};">'
            f'<span style="color: {color};">{icon}</span> '
            f"<span>{event['message']}</span>"
            f"</div>"
        )
    return "\n".join(html_parts)


def render_outcome_summary(snapshot: TrainingSnapshot):
    """Render the recent episode outcome summary (horizontal stacked bar)."""
    # We need access to the agent's recent termination reasons
    # For now, we'll show a placeholder that will be enhanced
    controller = get_controller()
    if not controller.agent or not controller.agent.scores_history:
        return None

    # Get recent termination reasons from the environment
    # For now, show a simple summary of recent scores
    recent_scores = controller.agent.scores_history[-50:]
    if not recent_scores:
        return None

    avg_score = sum(recent_scores) / len(recent_scores)
    max_score = max(recent_scores)

    html = f"""
    <div class="outcome-summary">
        <div style="font-weight: 600; margin-bottom: 0.25rem;">Recent Episode Outcomes (last 50)</div>
        <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; font-size: 0.85rem;">
            <span style="background: #333; padding: 0.15rem 0.5rem; border-radius: 4px;">Avg: {avg_score:.1f}</span>
            <span style="background: #333; padding: 0.15rem 0.5rem; border-radius: 4px;">Max: {max_score}</span>
            <span style="background: #333; padding: 0.15rem 0.5rem; border-radius: 4px;">Count: {len(recent_scores)}</span>
        </div>
    """
    return html


def render_state_vector(snapshot: TrainingSnapshot):
    """Render the 11-dimensional state vector as compact horizontal bars."""
    obs = snapshot.observation
    if obs is None:
        return None

    labels = [
        "Danger→",
        "Danger↘",
        "Danger↙",
        "Dir←",
        "Dir→",
        "Dir↑",
        "Dir↓",
        "Food←",
        "Food→",
        "Food↑",
        "Food↓",
    ]

    colors = []
    for i, val in enumerate(obs):
        if i < 3:  # Danger
            colors.append("#FF4444" if val > 0.5 else "#44FF44")
        elif i < 7:  # Direction
            colors.append("#4488FF" if val > 0.5 else "#333333")
        else:  # Food
            colors.append("#FFD700" if val > 0.5 else "#444444")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=obs,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}" for v in obs],
            textposition="inside",
            textfont={"color": "white", "size": 10, "family": "Arial Black"},
            showlegend=False,
            hovertemplate="%{y}: %{x:.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=220,
        margin={"l": 50, "r": 5, "t": 5, "b": 5},
        xaxis={"range": [0, 1.1], "showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "autorange": "reversed", "tickfont": {"size": 9}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 9},
    )
    return fig


def render_q_values(snapshot: TrainingSnapshot):
    """Render Q-values and action probabilities as compact bar charts."""
    q_vals = snapshot.q_values
    action_probs = snapshot.action_probs
    action_idx = snapshot.action

    if q_vals is None:
        return None, None

    action_names = ["Straight", "Right", "Left"]

    # Q-values
    q_colors = ["#44FF44" if i == action_idx else "#666666" for i in range(3)]
    fig_q = go.Figure()
    fig_q.add_trace(
        go.Bar(
            x=action_names,
            y=q_vals,
            marker_color=q_colors,
            text=[f"{v:.2f}" for v in q_vals],
            textposition="outside",
            textfont={"color": "white", "size": 11},
            showlegend=False,
        )
    )
    fig_q.update_layout(
        title={"text": "Q-Values", "font": {"size": 12}},
        height=180,
        margin={"l": 20, "r": 10, "t": 30, "b": 10},
        yaxis={"showgrid": True, "gridcolor": "rgba(100,100,100,0.2)"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 9},
    )

    # Probabilities
    if action_probs is not None:
        probs = action_probs
        p_colors = ["#FFD700" if i == action_idx else "#555555" for i in range(3)]
        fig_p = go.Figure()
        fig_p.add_trace(
            go.Bar(
                x=action_names,
                y=probs,
                marker_color=p_colors,
                text=[f"{p:.0%}" for p in probs],
                textposition="outside",
                textfont={"color": "white", "size": 11},
                showlegend=False,
            )
        )
        fig_p.update_layout(
            title={"text": "Probabilities", "font": {"size": 12}},
            height=180,
            margin={"l": 20, "r": 10, "t": 30, "b": 10},
            yaxis={"range": [0, 1.1], "showgrid": True, "gridcolor": "rgba(100,100,100,0.2)"},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white", "size": 9},
        )
    else:
        fig_p = None

    return fig_q, fig_p


def render_network_architecture(snapshot: TrainingSnapshot, controller: TrainingController):
    """Render compact neural network architecture."""
    if not st.session_state.show_network or controller.agent is None:
        return None

    device = controller.agent.trainer.device
    obs_tensor = torch.FloatTensor(snapshot.observation).unsqueeze(0).to(device)
    model = controller.agent.model
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            acts = model.get_activations(obs_tensor)
    finally:
        model.train(was_training)

    layer_sizes = [
        model.fc1.in_features,
        model.fc1.out_features,
        model.fc2.out_features,
        model.fc3.out_features,
    ]
    layer_names = [
        f"Input ({layer_sizes[0]})",
        f"Hidden 1 ({layer_sizes[1]})",
        f"Hidden 2 ({layer_sizes[2]})",
        f"Output ({layer_sizes[3]})",
    ]

    fig = go.Figure()
    layer_x = [0, 1, 2, 3]

    for layer_idx, (size, name) in enumerate(zip(layer_sizes, layer_names, strict=True)):
        x = layer_x[layer_idx]

        if layer_idx == 0:
            act_vals = snapshot.observation
        else:
            key = f"fc{layer_idx}"
            if key in acts:
                act_vals = np.abs(acts[key][0])
                if act_vals.max() > 0:
                    act_vals = act_vals / act_vals.max()
            else:
                act_vals = np.zeros(size)

        if size <= 16:
            indices = np.arange(size)
            display_acts = act_vals
        else:
            indices = np.concatenate([np.arange(min(6, size)), np.arange(max(6, size - 6), size)])
            display_acts = act_vals[indices]

        y_positions = np.linspace(-1, 1, len(indices))

        action_idx = snapshot.action if snapshot.action is not None else -1

        for _j, (idx, y, act) in enumerate(zip(indices, y_positions, display_acts, strict=True)):
            intensity = min(1.0, act * 2)
            if layer_idx == 0:
                color = f"rgba(255, {int(255 * (1 - intensity))}, {int(255 * (1 - intensity))}, 1)"
            elif layer_idx == 3:
                color = (
                    "#FFD700" if idx == action_idx else f"rgba(100,100,255,{0.3 + 0.7 * intensity})"
                )
            else:
                color = f"rgba(100, {int(100 + 155 * intensity)}, 255, {0.3 + 0.7 * intensity})"

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker={"size": 10, "color": color, "line": {"width": 0.5, "color": "white"}},
                    showlegend=False,
                    hovertemplate=f"{name}<br>Neuron {idx}<br>Act: {act:.2f}<extra></extra>",
                )
            )

        fig.add_annotation(
            x=x,
            y=1.15,
            text=name,
            showarrow=False,
            font={"color": "white", "size": 9},
            xanchor="center",
        )

    # Layer connections
    for i in range(3):
        fig.add_shape(
            type="line",
            x0=layer_x[i] + 0.12,
            y0=-1.05,
            x1=layer_x[i + 1] - 0.12,
            y1=-1.05,
            line={"color": "rgba(100,100,100,0.2)", "width": 1.5},
            layer="below",
        )

    fig.update_layout(
        title={"text": "Network Activations", "font": {"size": 12}},
        height=200,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"range": [-0.4, 3.4], "showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={
            "range": [-1.2, 1.25],
            "showgrid": False,
            "showticklabels": False,
            "zeroline": False,
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 9},
    )
    return fig


def render_training_charts(snapshot: TrainingSnapshot, controller: TrainingController):
    """Render compact training analytics charts."""
    # SCORE CHART
    if snapshot.recent_scores:
        display_scores = snapshot.recent_scores
        display_means = snapshot.recent_mean_scores
        episodes = list(range(snapshot.episode - len(display_scores) + 1, snapshot.episode + 1))

        fig_score = go.Figure()
        fig_score.add_trace(
            go.Scatter(
                x=episodes,
                y=display_scores,
                mode="lines",
                name="Score",
                line={"color": "#4488FF", "width": 1},
                opacity=0.7,
            )
        )
        if display_means:
            fig_score.add_trace(
                go.Scatter(
                    x=episodes,
                    y=display_means,
                    mode="lines",
                    name="Avg(50)",
                    line={"color": "#FFD700", "width": 2},
                    opacity=0.9,
                )
            )
    else:
        fig_score = go.Figure()
        fig_score.add_annotation(
            x=0.5,
            y=0.5,
            text="No episodes completed",
            showarrow=False,
            font={"color": "gray", "size": 12},
            xref="paper",
            yref="paper",
        )

    fig_score.update_layout(
        title={"text": "Scores per Episode", "font": {"size": 11}},
        height=180,
        margin={"l": 30, "r": 10, "t": 25, "b": 25},
        xaxis={"showgrid": True, "gridcolor": "rgba(100,100,100,0.15)", "title": ""},
        yaxis={"showgrid": True, "gridcolor": "rgba(100,100,100,0.15)", "title": ""},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 9},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 9},
        },
        uirevision="snake-scores",
    )

    # LOSS CHART
    fig_loss = go.Figure()
    if snapshot.recent_loss:
        values = np.asarray(snapshot.recent_loss, dtype=float)
        if values.size:
            last_step = snapshot.optimizer_steps
            steps = np.arange(last_step - len(values) + 1, last_step + 1)
            finite = np.isfinite(values)

            plotted = np.where(finite & (values > 0), values, np.nan)

            if np.any(np.isfinite(plotted)):
                fig_loss.add_trace(
                    go.Scatter(
                        x=steps,
                        y=plotted,
                        mode="lines",
                        name="Optimizer loss",
                        line={"color": "#FF6666", "width": 1},
                        opacity=0.7,
                    )
                )
            else:
                fig_loss.add_annotation(
                    text="No positive finite losses for log scale",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font={"color": "gray", "size": 11},
                )
    else:
        fig_loss.add_annotation(
            text="No optimizer updates yet",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": "gray", "size": 12},
        )

    fig_loss.update_layout(
        title={"text": "Loss per Optimizer Update (log)", "font": {"size": 11}},
        height=180,
        margin={"l": 40, "r": 10, "t": 25, "b": 30},
        xaxis={"title": "Optimizer Step", "showgrid": True, "gridcolor": "rgba(100,100,100,0.15)"},
        yaxis={
            "type": "log",
            "title": "Loss",
            "showgrid": True,
            "gridcolor": "rgba(100,100,100,0.15)",
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 9},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 9},
        },
        uirevision="snake-loss",
    )

    # EPSILON CHART
    if snapshot.recent_epsilon:
        display_eps = snapshot.recent_epsilon
        episodes = list(range(snapshot.episode - len(display_eps) + 1, snapshot.episode + 1))
        fig_eps = go.Figure()
        fig_eps.add_trace(
            go.Scatter(
                x=episodes,
                y=display_eps,
                mode="lines",
                name="ε",
                line={"color": "#44FF88", "width": 1.5},
                fill="tozeroy",
                fillcolor="rgba(68,255,136,0.1)",
            )
        )
    else:
        fig_eps = go.Figure()
        fig_eps.add_annotation(
            x=0.5,
            y=0.5,
            text="No epsilon history",
            showarrow=False,
            font={"color": "gray", "size": 12},
            xref="paper",
            yref="paper",
        )

    fig_eps.update_layout(
        title={"text": "Exploration Rate (ε)", "font": {"size": 11}},
        height=180,
        margin={"l": 30, "r": 10, "t": 25, "b": 25},
        xaxis={"showgrid": True, "gridcolor": "rgba(100,100,100,0.15)", "title": ""},
        yaxis={
            "range": [0, 1.05],
            "showgrid": True,
            "gridcolor": "rgba(100,100,100,0.15)",
            "title": "",
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "size": 9},
        uirevision="snake-epsilon",
    )

    return fig_score, fig_loss, fig_eps


# ============================================================================
# Main Application
# ============================================================================
def main():
    controller = get_controller()

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🐍 Snake DQN")
        st.caption("Live Training Dashboard")
        st.divider()

        # Environment Settings
        with st.expander("🌍 Environment", expanded=True):
            grid_size = st.slider("Grid Size", 10, 30, controller.env_config.grid_size)
            step_penalty = st.number_input(
                "Step Penalty", -1.0, 0.0, controller.env_config.step_penalty, 0.001, format="%.3f"
            )
            food_reward = st.number_input(
                "Food Reward", 0.0, 100.0, controller.env_config.food_reward, 1.0
            )
            death_penalty = st.number_input(
                "Death Penalty", -100.0, 0.0, controller.env_config.death_penalty, 1.0
            )

            if btn_full("Apply Environment Settings"):
                new_config = EnvConfig(
                    grid_size=grid_size,
                    step_penalty=step_penalty,
                    food_reward=food_reward,
                    death_penalty=death_penalty,
                    render_mode="rgb_array",
                )
                controller.update_env_config(new_config)
                st.success("Environment updated - agent and replay reset")
                st.rerun()

        # Agent Settings
        with st.expander("🧠 Agent Hyperparameters", expanded=True):
            hidden_size = st.selectbox(
                "Hidden Size",
                [128, 256, 512],
                index=[128, 256, 512].index(controller.agent_config.hidden_size),
            )
            lr = st.number_input(
                "Learning Rate", 1e-5, 1e-1, controller.agent_config.lr, 1e-4, format="%.5f"
            )
            gamma = st.number_input(
                "Gamma", 0.8, 0.999, controller.agent_config.gamma, 0.001, format="%.3f"
            )
            epsilon_start = st.number_input(
                "Epsilon Start", 0.0, 1.0, controller.agent_config.epsilon_start, 0.01
            )
            epsilon_end = st.number_input(
                "Epsilon End", 0.0, 0.5, controller.agent_config.epsilon_end, 0.001, format="%.3f"
            )
            epsilon_decay = st.number_input(
                "Epsilon Decay",
                0.9,
                0.9999,
                controller.agent_config.epsilon_decay,
                0.0001,
                format="%.4f",
            )
            memory_size = st.selectbox(
                "Memory Size",
                [10000, 50000, 100000, 200000],
                index=[10000, 50000, 100000, 200000].index(controller.agent_config.memory_size),
            )
            batch_size = st.selectbox(
                "Batch Size",
                [256, 512, 1000, 2048],
                index=[256, 512, 1000, 2048].index(controller.agent_config.batch_size),
            )

            if btn_full("Create New Agent"):
                new_config = AgentConfig(
                    hidden_size=hidden_size,
                    lr=lr,
                    gamma=gamma,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay=epsilon_decay,
                    memory_size=memory_size,
                    batch_size=batch_size,
                )
                controller.update_agent_config(new_config)
                st.success("New agent created!")
                st.rerun()

        # Training Controls
        with st.expander("🏋️ Training", expanded=True):
            # Activity selector
            activity_options = ["Train", "Evaluate", "Manual"]
            activity_labels = {
                "Train": "🧠 Train — Agent learns while playing",
                "Evaluate": "📊 Evaluate — Greedy policy, no learning",
                "Manual": "👤 Manual — You control the snake",
            }
            st.session_state.activity = st.radio(
                "Activity",
                activity_options,
                index=activity_options.index(st.session_state.get("activity", "Train")),
                format_func=lambda x: activity_labels[x],
                horizontal=True,
                on_change=sync_activity_to_controller,
            )

            # Execution policy for Train/Evaluate modes
            if st.session_state.activity != "Manual":
                st.caption("Execution Policy")
                policy_options = {
                    "Watch": "Watch — Deliberately paced, readable movement",
                    "FastTraining": "FastTraining — Multiple transitions per UI update",
                }
                st.session_state.execution_policy = st.radio(
                    "Policy",
                    list(policy_options.keys()),
                    index=list(policy_options.keys()).index(
                        st.session_state.get("execution_policy", "Watch")
                    ),
                    format_func=lambda x: policy_options[x],
                    horizontal=False,
                )

                if st.session_state.execution_policy == "Watch":
                    speed_options = {
                        "Watch": 0.1,
                        "Normal": 0.05,
                        "Fast": 0.01,
                        "Max": 0.0,
                    }
                    st.session_state.speed_preset = st.radio(
                        "Playback speed",
                        list(speed_options.keys()),
                        index=list(speed_options.keys()).index(
                            st.session_state.get("speed_preset", "Watch")
                        ),
                        horizontal=True,
                    )
                    st.session_state.speed = speed_options[st.session_state.speed_preset]
                else:
                    st.session_state.speed = 0.0
                    st.session_state.max_burst_steps = st.number_input(
                        "Max steps per burst",
                        1,
                        1000,
                        st.session_state.get("max_burst_steps", 50),
                        step=10,
                        help="Maximum transitions per UI update burst",
                    )
                    st.session_state.burst_time_budget = (
                        st.number_input(
                            "Burst time budget (ms)",
                            5,
                            200,
                            int(st.session_state.get("burst_time_budget", 0.05) * 1000),
                            step=5,
                            help="Target compute time per burst",
                        )
                        / 1000.0
                    )

            # Main control button
            if st.session_state.activity == "Train":
                if controller.paused and not st.session_state.training:
                    if btn_full("▶️ Start training", type="primary"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        controller.restart_board()
                        st.rerun()
                elif controller.paused:
                    if btn_full("▶️ Resume training", type="primary"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        st.rerun()
                else:
                    if btn_full("⏸️ Pause training"):
                        st.session_state.paused = True
                        sync_paused_to_controller()
                        st.rerun()

            elif st.session_state.activity == "Evaluate":
                if controller.paused and not st.session_state.training:
                    if btn_full("▶️ Start evaluation"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        controller.restart_board()
                        st.rerun()
                elif controller.paused:
                    if btn_full("▶️ Resume evaluation"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        st.rerun()
                else:
                    if btn_full("⏸️ Pause evaluation"):
                        st.session_state.paused = True
                        sync_paused_to_controller()
                        st.rerun()

            else:  # Manual
                if not st.session_state.paused:
                    if btn_full("⏸️ Pause"):
                        st.session_state.paused = True
                        sync_paused_to_controller()
                        st.rerun()

            # Step once button (only when paused)
            if st.session_state.paused:
                if btn_full("⏭️ Step once"):
                    controller.step()
                    st.rerun()

            # Reset buttons
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if btn_full(
                    "🔄 Restart board",
                    help="Restart current game, keep learned weights and training history",
                ):
                    controller.restart_board()
                    st.rerun()
            with c2:
                if btn_full(
                    "🗑️ Clear training history",
                    help="Clear displayed charts and statistics, keep model weights",
                ):
                    controller.clear_history()
                    st.rerun()

            # New agent / environment reset
            st.divider()
            if btn_full(
                "🆕 Start new training run",
                help="Create fresh agent and replay buffer, discard current model",
            ):
                controller.new_run()
                st.success("New training run started")
                st.rerun()

            st.divider()
            save_path = st.text_input("Checkpoint Path", "checkpoints/snake_dqn.pth")
            c1, c2 = st.columns(2)
            with c1:
                if btn_full("💾 Save checkpoint"):
                    if controller.agent:
                        controller.save_checkpoint(save_path)
                        st.success("Saved!")
            with c2:
                if btn_full("📂 Load checkpoint"):
                    try:
                        info = controller.load_checkpoint(save_path)
                        st.success(
                            f"Loaded ep {info.get('episode', 0)}, ε={info.get('epsilon', 1.0):.3f}"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")

        # Display Settings
        with st.expander("🎨 Display", expanded=False):
            st.session_state.show_network = st.checkbox(
                "Show network internals", st.session_state.get("show_network", False)
            )
            st.session_state.show_grid = st.checkbox(
                "Show grid lines", st.session_state.get("show_grid", True)
            )
            if btn_full("🎮 New Game", help="Restart board (same as Restart board above)"):
                controller.restart_board()
                st.rerun()

    # Sync activity on load
    sync_activity_to_controller()
    sync_paused_to_controller()

    # Get current snapshot
    snapshot = controller.get_snapshot()

    # Check for new events
    check_new_events(controller, snapshot)

    # ============================================================================
    # Live Game Fragment (partial updates via st.fragment)
    # ============================================================================
    _running = not controller.paused and controller.mode != ActivityMode.MANUAL

    if _running:
        if st.session_state.get("execution_policy", "Watch") == "FastTraining":
            _refresh_seconds = 0.0
        else:
            _refresh_seconds = st.session_state.speed
    else:
        _refresh_seconds = None

    @st.fragment(run_every=_refresh_seconds)
    def live_game_area():
        """Render game board and auto-step. Runs independently of main page."""
        ctl = get_controller()

        # Auto-step logic (inside fragment)
        if not ctl.paused and ctl.mode != ActivityMode.MANUAL:
            policy = st.session_state.get("execution_policy", "Watch")
            if policy == "FastTraining":
                import time as _time

                max_steps = st.session_state.get("max_burst_steps", 50)
                time_budget = st.session_state.get("burst_time_budget", 0.05)
                burst_start = _time.monotonic()
                steps_executed = 0

                while steps_executed < max_steps and not ctl.game_over:
                    ctl.step()
                    steps_executed += 1

                    if _time.monotonic() - burst_start >= time_budget:
                        break

                if ctl.game_over and ctl.mode == ActivityMode.TRAIN:
                    ctl.restart_board()
            else:
                if not ctl.game_over:
                    ctl.step()
                elif ctl.mode == ActivityMode.TRAIN:
                    ctl.restart_board()
                else:
                    ctl.set_paused(True)
                    st.session_state.paused = True

        # Render game using coordinate-only Plotly figure (no RGB/PNG/base64 on normal path)
        snap = ctl.get_snapshot()
        if snap.game_state:
            fig = render_game_grid(snap, show_grid=st.session_state.get("show_grid", True))
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False, "staticPlot": True},
                key="live_game_canvas",
            )

        # Status caption (updates on fragment refresh)
        if ctl.agent is not None and snap.game_state:
            st.caption(
                f"Ep {ctl.agent.n_games} · "
                f"Score {snap.game_state['score']} · "
                f"Best {ctl.agent.high_score} · "
                f"ε {ctl.agent.epsilon:.3f}"
            )

    # Chart fragment - refreshes every 1 second when running
    _chart_refresh = 1.0 if _running else None

    @st.fragment(run_every=_chart_refresh)
    def live_charts_area():
        """Render learning charts. Updates independently of game fragment."""
        ctl = get_controller()
        snap = ctl.get_snapshot()

        fig_score, fig_loss, fig_eps = render_training_charts(snap, ctl)

        for name, fig, slot in zip(
            ["scores", "loss", "epsilon"],
            [fig_score, fig_loss, fig_eps],
            [score_slot, loss_slot, epsilon_slot],
            strict=True,
        ):
            fig.update_layout(
                height=220 if name == "scores" else 180,
                autosize=True,
                width=None,
                uirevision=f"snake-{name}",
                transition_duration=0,
            )
            slot.plotly_chart(
                fig, width="stretch", key=f"live_{name}", config={"displayModeBar": False}
            )

    # Persistent chart placeholders
    score_slot = st.empty()
    loss_col, epsilon_col = st.columns(2, gap="small")
    loss_slot = loss_col.empty()
    epsilon_slot = epsilon_col.empty()

    # ============================================================================
    # Main Layout: Overview (left) + Inspect Learning (right)
    # ============================================================================
    overview_col, inspect_col = st.columns([2, 3], gap="small")

    # ----- LEFT COLUMN: OVERVIEW -----
    with overview_col:
        # Live game fragment (auto-updates)
        live_game_area()

        # State indicator
        activity = st.session_state.activity
        is_paused = controller.paused

        if activity == "Train":
            if is_paused:
                status_text = (
                    "🟡 Train · Paused"
                    if controller.agent and controller.agent.n_games > 0
                    else "🟡 Train · Ready"
                )
            else:
                status_text = "🟢 Train · Running"
        elif activity == "Evaluate":
            if is_paused:
                status_text = "🔵 Evaluate · Paused"
            else:
                status_text = "🔵 Evaluate · Running"
        else:
            if is_paused:
                status_text = "⚪ Manual · Paused"
            else:
                status_text = "⚪ Manual · Running"

        st.caption(status_text)

        # Controls row
        c1, c2, c3 = st.columns(3)
        with c1:
            if activity == "Train":
                if is_paused and (controller.agent is None or controller.agent.n_games == 0):
                    if btn_full("▶️ Start training", type="primary", key="play_pause"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        controller.restart_board()
                        st.rerun()
                elif is_paused:
                    if btn_full("▶️ Resume", type="primary", key="play_pause"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        st.rerun()
                else:
                    if btn_full("⏸️ Pause", key="play_pause"):
                        st.session_state.paused = True
                        sync_paused_to_controller()
                        st.rerun()
            elif activity == "Evaluate":
                if is_paused and (controller.agent is None or controller.agent.n_games == 0):
                    if btn_full("▶️ Start evaluation", type="primary", key="play_pause"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        controller.restart_board()
                        st.rerun()
                elif is_paused:
                    if btn_full("▶️ Resume", type="primary", key="play_pause"):
                        st.session_state.paused = False
                        sync_paused_to_controller()
                        st.rerun()
                else:
                    if btn_full("⏸️ Pause", key="play_pause"):
                        st.session_state.paused = True
                        sync_paused_to_controller()
                        st.rerun()
            else:  # Manual
                if not is_paused:
                    if btn_full("⏸️ Pause", key="play_pause"):
                        st.session_state.paused = True
                        sync_paused_to_controller()
                        st.rerun()

        with c2:
            if st.session_state.paused:
                if btn_full("⏭️ Step once", key="step_btn"):
                    controller.step()
                    st.rerun()
            else:
                st.button("⏭️ Step once", width="stretch", disabled=True, key="step_btn")

        with c3:
            if btn_full(
                "🔄 Restart board",
                key="reset_btn",
                help="Restart current game, keep learned weights",
            ):
                controller.restart_board()
                st.rerun()

        # Manual controls when paused (Manual activity only)
        if (
            st.session_state.activity == "Manual"
            and st.session_state.paused
            and not controller.game_over
        ):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                if btn_full("⬆️ Straight", key="manual_up"):
                    dir_map = {0: 2, 1: 1, 2: 0, 3: 0}
                    action = dir_map.get(snapshot.game_state.get("direction", 0), 0)
                    controller.step(action=action)
                    st.rerun()
            with mc2:
                if btn_full("➡️ Right", key="manual_right"):
                    controller.step(action=1)
                    st.rerun()
            with mc3:
                if btn_full("⬅️ Left", key="manual_left"):
                    controller.step(action=2)
                    st.rerun()

        # --- NEW: Last Decision Strip ---
        st.divider()
        decision_html = render_decision_strip(snapshot)
        if decision_html:
            st.markdown(decision_html, unsafe_allow_html=True)

        # --- NEW: Event Feed ---
        event_html = render_event_feed()
        if event_html:
            st.markdown(event_html, unsafe_allow_html=True)

        # --- NEW: Episode Outcome Summary ---
        st.divider()
        outcome_html = render_outcome_summary(snapshot)
        if outcome_html:
            st.markdown(outcome_html, unsafe_allow_html=True)

        # Network Details (conditionally shown)
        if st.session_state.show_network:
            st.divider()
            st.subheader("🔬 Inspect Learning")

            # State Vector
            st.caption("State Vector (11D)")
            fig_sv = render_state_vector(snapshot)
            if fig_sv:
                st.plotly_chart(
                    fig_sv, width="stretch", config={"displayModeBar": False}, key="state_vector"
                )

            # Q-Values & Probabilities
            fig_q, fig_p = render_q_values(snapshot)
            qcol, pcol = st.columns(2)
            with qcol:
                if fig_q:
                    st.plotly_chart(
                        fig_q, width="stretch", config={"displayModeBar": False}, key="q_values"
                    )
            with pcol:
                if fig_p:
                    st.plotly_chart(
                        fig_p, width="stretch", config={"displayModeBar": False}, key="action_probs"
                    )

            # Network Architecture
            fig_net = render_network_architecture(snapshot, controller)
            if fig_net:
                st.plotly_chart(
                    fig_net, width="stretch", config={"displayModeBar": False}, key="network_arch"
                )

    # ----- RIGHT COLUMN: LEARNING ANALYTICS -----
    with inspect_col:
        st.subheader("📈 Learning Analytics")

        # Live charts fragment
        live_charts_area()

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Episode", f"{controller.episode}")
        with m2:
            st.metric("Score", f"{snapshot.score}")
        with m3:
            st.metric("High Score", f"{controller.high_score}")
        with m4:
            st.metric("ε (Epsilon)", f"{controller.epsilon:.3f}")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Env Steps", f"{controller.total_env_steps}")
        with m2:
            st.metric("Optimizer Steps", f"{controller.optimizer_steps}")
        with m3:
            avg50 = (
                controller.agent.mean_scores_history[-1]
                if controller.agent and controller.agent.mean_scores_history
                else 0
            )
            st.metric("Avg(50)", f"{avg50:.1f}")
        with m4:
            mem_size = len(controller.agent.memory) if controller.agent else 0
            st.metric("Replay Size", f"{mem_size:,}")

        # Performance metrics
        if controller._metrics:
            m = controller._metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Transitions/s", f"{m.get('transitions_per_sec', 0):.1f}")
            with m2:
                st.metric("Opt Updates/s", f"{m.get('optimizer_updates_per_sec', 0):.1f}")
            with m3:
                st.metric("Replay Size", f"{m.get('replay_buffer_size', 0):,}")
            with m4:
                st.metric("Replay Active", "✅" if m.get("replay_active") else "⏳")


if __name__ == "__main__":
    main()
