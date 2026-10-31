"""
Chart rendering components for Snake DQN Dashboard.

All charts use Plotly with theme-aware styling from ui.styles.
Each function returns a Plotly Figure ready for st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go

from ui.styles import (
    ACTION_NAMES,
    CHART_HEIGHT_MAIN,
    CHART_HEIGHT_NETWORK,
    CHART_HEIGHT_QVALUES,
    CHART_HEIGHT_SMALL,
    CHART_HEIGHT_STATE,
    STATE_LABELS_FLAT,
    get_plotly_template,
)
from observation_schema import OBSERVATION_FEATURES


def apply_template(fig: go.Figure, height: int | None = None, uirevision: str | None = None):
    """Apply consistent template to a figure.

    Uses explicit construction order:
    1. Create figure
    2. Apply base template (explicit layout properties via update_layout for proper merging)
    3. Apply chart-specific layout (already on fig from caller)
    4. Apply final dimensions (height, uirevision)
    """
    template = get_plotly_template()
    # Apply template layout properties via update_layout to allow proper merging
    # with subsequent chart-specific layout updates.
    # Don't rely on 'key in fig.layout' because Plotly graph objects report True
    # for properties that exist as defaults but weren't explicitly set by chart code.
    fig.update_layout(template["layout"])
    if height is not None:
        fig.update_layout(height=height)
    if uirevision is not None:
        fig.update_layout(uirevision=uirevision)
    return fig


# =============================================================================
# Score Chart
# =============================================================================


def render_score_chart(
    recent_scores: list[int],
    recent_mean_scores: list[float],
    episode: int,
    window: int = 50,
) -> go.Figure:
    """Render score per episode with rolling average."""
    fig = go.Figure()

    if recent_scores:
        n = len(recent_scores)
        episodes = list(range(episode - n + 1, episode + 1))

        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=recent_scores,
                mode="lines",
                name="Score",
                line={"color": "#4488ff", "width": 1},
                opacity=0.7,
                hovertemplate="Episode %{x}<br>Score: %{y}<extra></extra>",
            )
        )

        if recent_mean_scores:
            fig.add_trace(
                go.Scatter(
                    x=episodes,
                    y=recent_mean_scores,
                    mode="lines",
                    name=f"Avg({window})",
                    line={"color": "#ffd700", "width": 2},
                    opacity=0.9,
                    hovertemplate="Episode %{x}<br>Rolling Avg: %{y:.1f}<extra></extra>",
                )
            )

        # Add annotation for window info
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            text=f"Window: {window} episodes · Showing {n} samples",
            showarrow=False,
            font={"size": 9, "color": "rgba(128,128,128,0.8)"},
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No episodes completed yet",
            showarrow=False,
            font={"size": 12, "color": "rgba(128,128,128,0.8)"},
        )

    fig.update_layout({
        "title": {"text": "Episode Scores", "font": {"size": 11}},
        "xaxis_title": "Episode",
        "yaxis_title": "Score",
    })

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=CHART_HEIGHT_MAIN, uirevision="snake-scores")
    # Chart-specific margins
    fig.update_layout({"margin": {"l": 50, "r": 20, "t": 35, "b": 40}})
    return fig


# =============================================================================
# Loss Chart
# =============================================================================


def render_loss_chart(
    recent_loss: list[float],
    optimizer_steps: int,
    log_scale: bool = True,
) -> go.Figure:
    """Render optimizer loss per step. Handles zero/nonfinite values honestly."""
    fig = go.Figure()

    if recent_loss:
        values: np.ndarray = np.asarray(recent_loss, dtype=float)
        if values.size:
            last_step = optimizer_steps
            steps = np.arange(last_step - len(values) + 1, last_step + 1)
            finite = np.isfinite(values)

            # Separate positive, zero, and negative/nonfinite
            positive = finite & (values > 0)
            zero = finite & (values == 0)
            nonfinite = ~finite
            negative = finite & (values < 0)

            # Plot positive values on log scale
            if np.any(positive):
                plotted = np.full_like(values, np.nan)
                plotted[positive] = values[positive]
                fig.add_trace(
                    go.Scatter(
                        x=steps[positive],
                        y=plotted[positive],
                        mode="lines",
                        name="Optimizer Loss",
                        line={"color": "#ff6666", "width": 1},
                        opacity=0.7,
                        hovertemplate="Step %{x}<br>Loss: %{y:.6f}<extra></extra>",
                    )
                )

            # Mark zero losses with a distinct marker on linear scale inset
            if np.any(zero):
                if log_scale:
                    # Add as separate trace with linear y-axis? Not directly supported.
                    # Instead, annotate and show on linear if user toggles.
                    pass

            # Annotation for data quality
            annotations = []
            if np.any(zero):
                annotations.append(f"Zero losses: {int(np.sum(zero))}")
            if np.any(negative):
                annotations.append(f"Negative losses: {int(np.sum(negative))} (clipped)")
            if np.any(nonfinite):
                annotations.append(f"Non-finite: {int(np.sum(nonfinite))}")

            if annotations:
                fig.add_annotation(
                    x=0.01,
                    y=0.01,
                    xref="paper",
                    yref="paper",
                    text="<br>".join(annotations),
                    showarrow=False,
                    font={"size": 9, "color": "#ffaa00"},
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="#ffaa00",
                    borderwidth=1,
                    borderpad=4,
                )

            if not np.any(positive) and not np.any(zero):
                fig.add_annotation(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="No positive/zero finite losses to display",
                    showarrow=False,
                    font={"color": "rgba(128,128,128,0.8)", "size": 11},
                )
        else:
            fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="No optimizer updates yet",
                showarrow=False,
                font={"color": "rgba(128,128,128,0.8)", "size": 12},
            )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No optimizer updates yet",
            showarrow=False,
            font={"color": "rgba(128,128,128,0.8)", "size": 12},
        )

    yaxis_config = {"title": "Loss"}
    if log_scale:
        yaxis_config["type"] = "log"
        yaxis_config["title"] = "Loss (log)"

    fig.update_layout({
        "title": {"text": "Loss per Optimizer Update", "font": {"size": 11}},
        "xaxis_title": "Optimizer Step",
        "yaxis": yaxis_config,
    })

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=CHART_HEIGHT_SMALL, uirevision="snake-loss")
    fig.update_layout({"margin": {"l": 50, "r": 20, "t": 35, "b": 40}})
    return fig


# =============================================================================
# Epsilon Chart
# =============================================================================


def render_epsilon_chart(
    recent_epsilon: list[float],
    episode: int,
) -> go.Figure:
    """Render exploration rate over episodes."""
    fig = go.Figure()

    if recent_epsilon:
        n = len(recent_epsilon)
        episodes = list(range(episode - n + 1, episode + 1))

        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=recent_epsilon,
                mode="lines",
                name="ε",
                line={"color": "#44cc44", "width": 1.5},
                fill="tozeroy",
                fillcolor="rgba(68,204,68,0.1)",
                hovertemplate="Episode %{x}<br>ε: %{y:.4f}<extra></extra>",
            )
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No epsilon history",
            showarrow=False,
            font={"color": "rgba(128,128,128,0.8)", "size": 12},
        )

    fig.update_layout({
        "title": {"text": "Exploration Rate (ε)", "font": {"size": 11}},
        "xaxis_title": "Episode",
        "yaxis": {"range": [0, 1.05], "title": "ε"},
    })

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=CHART_HEIGHT_SMALL, uirevision="snake-epsilon")
    fig.update_layout({"margin": {"l": 50, "r": 20, "t": 35, "b": 40}})
    return fig


# =============================================================================
# Throughput Chart (optional)
# =============================================================================


def render_throughput_chart(
    transitions_per_sec: list[float],
    optimizer_updates_per_sec: list[float],
    steps: list[int],
) -> go.Figure:
    """Render training throughput over time."""
    fig = go.Figure()

    if transitions_per_sec:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=transitions_per_sec,
                mode="lines",
                name="Transitions/s",
                line={"color": "#4488ff", "width": 1.5},
                opacity=0.8,
            )
        )

    if optimizer_updates_per_sec:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=optimizer_updates_per_sec,
                mode="lines",
                name="Opt Updates/s",
                line={"color": "#ffd700", "width": 1.5},
                opacity=0.8,
            )
        )

    if not transitions_per_sec and not optimizer_updates_per_sec:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No throughput data",
            showarrow=False,
            font={"color": "rgba(128,128,128,0.8)", "size": 12},
        )

    fig.update_layout({
        "title": {"text": "Training Throughput", "font": {"size": 11}},
        "xaxis_title": "Measurement Step",
        "yaxis_title": "Rate (per sec)",
    })

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=CHART_HEIGHT_SMALL, uirevision="snake-throughput")
    fig.update_layout({"margin": {"l": 50, "r": 20, "t": 35, "b": 40}})
    return fig


# =============================================================================
# State Vector Chart
# =============================================================================


def render_state_vector(observation: np.ndarray | None) -> go.Figure | None:
    """Render the 22-dimensional state vector as horizontal bars grouped by category."""
    if observation is None:
        return None

    labels = [
        name.replace("_", " ").title()
        for name in OBSERVATION_FEATURES
    ]

    if len(observation) != len(labels):
        raise ValueError("Observation and feature-label counts differ")

    # Color by group: Danger=red, Direction=blue, Food=gold, 
    # Food displacement=orange, Wall=purple, Length/timeout=green, Space=teal
    colors = []
    for i, val in enumerate(observation):
        if i < 3:  # Danger
            colors.append("#ff4444" if val > 0.5 else "#44cc44")
        elif i < 7:  # Direction
            colors.append("#4488ff" if val > 0.5 else "rgba(100,100,100,0.3)")
        elif i < 11:  # Food direction
            colors.append("#ffd700" if val > 0.5 else "rgba(100,100,100,0.3)")
        elif i < 13:  # Food displacement (dx, dy)
            colors.append("#ff8800")
        elif i < 17:  # Wall distances
            colors.append("#aa44ff")
        elif i < 19:  # Length, timeout
            colors.append("#44cc44")
        else:  # Reachable space
            colors.append("#00cccc")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=observation,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in observation],
            textposition="inside",
            textfont={"color": "white", "size": 10, "family": "Arial Black"},
            showlegend=False,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=650, uirevision="snake-state-vector")
    # Then chart-specific layout
    fig.update_layout(
        title={"text": "State Vector (22D)", "font": {"size": 11}},
        xaxis={
            "range": [-1.1, 1.1],
            "showgrid": False,
            "showticklabels": True,
            "zeroline": True,
        },
        yaxis={
            "showgrid": False,
            "autorange": "reversed",
            "tickfont": {"size": 11},
            "automargin": True,
        },
        margin={"l": 150, "r": 20, "t": 45, "b": 25},
    )
    return fig


# =============================================================================
# Q-Values Chart
# =============================================================================


def render_q_values(
    q_values: np.ndarray | None,
    action_executed: int | None,
    action_preferred: int | None,
    selection_reason: str | None,
) -> go.Figure | None:
    """Render Q-values for three actions with executed/preferred markers.

    Args:
        q_values: Array of 3 Q-values [Straight, Right, Left]
        action_executed: The action actually taken (0,1,2) or None
        action_preferred: The greedy action (argmax Q) or None
        selection_reason: "exploring" | "exploiting" | "manual" | None
    """
    if q_values is None:
        return None

    fig = go.Figure()

    # Determine bar colors
    colors = []
    for i in range(3):
        if i == action_executed and i == action_preferred:
            colors.append("#44cc44")  # Executed AND preferred (exploiting or lucky explore)
        elif i == action_executed:
            colors.append("#ff8800")  # Executed but not preferred (exploring)
        elif i == action_preferred:
            colors.append("#4488ff")  # Preferred but not executed (would have been greedy)
        else:
            colors.append("#666666")

    fig.add_trace(
        go.Bar(
            x=ACTION_NAMES,
            y=q_values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in q_values],
            textposition="outside",
            textfont={"color": "white", "size": 11},
            showlegend=False,
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )

    # Add markers for executed/preferred
    annotations = []
    if action_executed is not None:
        annotations.append(
            {
                "x": ACTION_NAMES[action_executed],
                "y": q_values[action_executed],
                "text": "▲ EXECUTED",
                "showarrow": True,
                "arrowhead": 2,
                "arrowcolor": "#ff8800",
                "arrowsize": 1,
                "ax": 0,
                "ay": -35,
                "font": {"color": "#ff8800", "size": 9, "family": "monospace"},
                "bgcolor": "rgba(0,0,0,0.7)",
                "bordercolor": "#ff8800",
                "borderwidth": 1,
                "borderpad": 2,
            }
        )
    if action_preferred is not None and action_preferred != action_executed:
        annotations.append(
            {
                "x": ACTION_NAMES[action_preferred],
                "y": q_values[action_preferred],
                "text": "◆ PREFERRED",
                "showarrow": True,
                "arrowhead": 2,
                "arrowcolor": "#4488ff",
                "arrowsize": 1,
                "ax": 0,
                "ay": 35,
                "font": {"color": "#4488ff", "size": 9, "family": "monospace"},
                "bgcolor": "rgba(0,0,0,0.7)",
                "bordercolor": "#4488ff",
                "borderwidth": 1,
                "borderpad": 2,
            }
        )

    # Q-value gap annotation
    if len(q_values) >= 2:
        sorted_q = sorted(q_values)
        gap = abs(sorted_q[-1] - sorted_q[-2])
        gap_text = f"Q-value gap: {gap:.3f}"
        if selection_reason == "exploring":
            gap_text += " (exploring)"
        elif selection_reason == "exploiting":
            gap_text += " (exploiting)"
        annotations.append(
            {
                "x": 0.5,
                "y": 1.08,
                "xref": "paper",
                "yref": "paper",
                "text": gap_text,
                "showarrow": False,
                "font": {"size": 10, "color": "rgba(255,255,255,0.8)"},
                "xanchor": "center",
                "yanchor": "bottom",
            }
        )

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=CHART_HEIGHT_QVALUES, uirevision="snake-qvalues")
    # Then chart-specific layout
    fig.update_layout({
        "title": {"text": "Q-Values per Action", "font": {"size": 11}},
        "yaxis_title": "Q-Value",
        "annotations": annotations,
        "margin": {"l": 50, "r": 20, "t": 55, "b": 45},
    })

    # Ensure y-axis has padding for annotations
    y_min = min(q_values) if len(q_values) > 0 else -1
    y_max = max(q_values) if len(q_values) > 0 else 1
    y_range = y_max - y_min
    pad = max(0.5, y_range * 0.3)
    fig.update_yaxes(range=[y_min - pad, y_max + pad])
    return fig


# =============================================================================
# Network Activations Chart
# =============================================================================


def render_network_activations(
    snapshot,
    controller,
    model_revision: int | None = None,
) -> go.Figure | None:
    """Render network activations for the current observation.

    Labels the chart as 'recomputed with current weights' unless activations
    were captured at decision time with matching model_revision.
    """
    import torch

    if not hasattr(snapshot, "observation") or snapshot.observation is None:
        return None

    if controller.agent is None:
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

    action_idx = snapshot.action if snapshot.action is not None else -1

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

        for _j, (idx, y, act) in enumerate(zip(indices, y_positions, display_acts, strict=True)):
            intensity = min(1.0, abs(float(act)) * 2)
            if layer_idx == 0:
                color = f"rgba(255, {int(255 * (1 - intensity))}, {int(255 * (1 - intensity))}, 1)"
            elif layer_idx == 3:
                color = (
                    "#ffd700" if idx == action_idx else f"rgba(100,100,255,{0.3 + 0.7 * intensity})"
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

    # Title with disclaimer moved to caption - keep title clean
    title_text = "Network Activations"
    if model_revision is None:
        title_text += " (recomputed with current weights)"

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=CHART_HEIGHT_NETWORK, uirevision="snake-network")
    # Then chart-specific layout
    fig.update_layout(
        title={"text": title_text, "font": {"size": 11}},
        xaxis={"range": [-0.4, 3.4], "showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"range": [-1.3, 1.3], "showgrid": False, "showticklabels": False, "zeroline": False},
        margin={"l": 30, "r": 30, "t": 45, "b": 30},
    )
    return fig


# =============================================================================
# Termination Reason Stacked Bar
# =============================================================================


def render_termination_summary(
    termination_counts: dict[str, int] | None = None,
    window: int = 50,
    recent_terminations: list[str] | None = None,
) -> go.Figure | None:
    """Render horizontal stacked bar of episode termination reasons.

    Args:
        termination_counts: Lifetime totals (used if recent_terminations not provided)
        window: Maximum episodes in the window (denominator is actual completed in window)
        recent_terminations: List of recent termination reasons (most recent last), maxlen=window.
                           If provided, this takes precedence over termination_counts.
    """
    # Determine what to display
    if recent_terminations is not None:
        # Count from recent window
        counts = {"wall": 0, "self": 0, "timeout": 0, "victory": 0}
        for reason in recent_terminations:
            if reason in counts:
                counts[reason] += 1
        completed = len(recent_terminations)
        if completed == 0:
            # No completed episodes - return empty state figure
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="No episodes completed yet",
                showarrow=False, font={"size": 11, "color": "rgba(128,128,128,0.8)"}
            )
            fig.update_layout(
                title={"text": "Episode Outcomes (none completed)", "font": {"size": 11}},
                barmode="stack",
                xaxis={"range": [0, 1], "showgrid": False, "showticklabels": False, "zeroline": False},
                yaxis={"showgrid": False, "showticklabels": False},
                height=80,
                margin={"l": 10, "r": 10, "t": 30, "b": 10},
            )
            fig = apply_template(fig, height=80, uirevision="snake-termination")
            return fig
        denominator = completed  # Use actual completed episodes in window
        source_label = f"last {completed}"
    elif termination_counts and sum(termination_counts.values()) > 0:
        # Fallback to lifetime counts
        counts = termination_counts
        completed = sum(counts.values())
        denominator = max(completed, window) if completed >= window else completed
        source_label = f"lifetime ({completed})"
    else:
        # No data at all - return empty state figure
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No episodes completed yet",
            showarrow=False, font={"size": 11, "color": "rgba(128,128,128,0.8)"}
        )
        fig.update_layout(
            title={"text": "Episode Outcomes (none completed)", "font": {"size": 11}},
            barmode="stack",
            xaxis={"range": [0, 1], "showgrid": False, "showticklabels": False, "zeroline": False},
            yaxis={"showgrid": False, "showticklabels": False},
            height=80,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        fig = apply_template(fig, height=80, uirevision="snake-termination")
        return fig

    # Color mapping
    colors = {
        "wall": "#ff4444",
        "self": "#ff8800",
        "timeout": "#ffd700",
        "victory": "#44cc44",
    }
    labels_map = {
        "wall": "Wall",
        "self": "Self",
        "timeout": "Timeout",
        "victory": "Victory",
    }

    fig = go.Figure()

    total = sum(counts.values())
    x_start: float = 0.0

    for reason in ["wall", "self", "timeout", "victory"]:
        count = counts.get(reason, 0)
        if count == 0:
            continue
        width = count / denominator  # Relative to actual completed in window
        fig.add_trace(
            go.Bar(
                x=[width],
                y=["Outcomes"],
                orientation="h",
                name=labels_map[reason],
                marker_color=colors[reason],
                text=[f"{count}"],
                textposition="inside",
                textfont={"color": "white", "size": 10, "family": "Arial Black"},
                showlegend=True,
                hovertemplate=f"{labels_map[reason]}: {count} ({count / max(total, 1) * 100:.1f}% of {total})<extra></extra>",
            )
        )
        x_start += width

    # Add remaining (incomplete to fill window) as transparent
    remaining = window - completed
    if remaining > 0:
        # Show as proportion of completed episodes (not window) to avoid invalid proportions
        # Use max(completed, 1) to avoid division by zero
        prop_denominator = max(completed, 1)
        fig.add_trace(
            go.Bar(
                x=[remaining / prop_denominator],
                y=["Outcomes"],
                orientation="h",
                name="Pending",
                marker_color="rgba(100,100,100,0.3)",
                text=[f"{remaining}"],
                textposition="inside",
                textfont={"color": "rgba(255,255,255,0.5)", "size": 10},
                showlegend=True,
                hovertemplate=f"Pending (to reach window of {window}): {remaining}<extra></extra>",
            )
        )

    # Add annotation for available episodes info
    if completed < window:
        annotation_text = f"{completed} of {window} episodes available"
    else:
        annotation_text = f"Showing last {window} episodes"
    
    fig.update_layout(
        title={
            "text": f"Episode Outcomes ({source_label})",
            "font": {"size": 11},
        },
        barmode="stack",
        xaxis={
            "range": [0, 1],
            "showgrid": False,
            "showticklabels": False,
            "zeroline": False,
            "title": "Proportion of Completed Episodes",
        },
        yaxis={"showgrid": False, "showticklabels": False},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 9},
        },
        height=80,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        annotations=[
            {
                "x": 0.5,
                "y": -0.25,
                "xref": "paper",
                "yref": "paper",
                "text": annotation_text,
                "showarrow": False,
                "font": {"size": 9, "color": "rgba(128,128,128,0.8)"},
                "xanchor": "center",
                "yanchor": "top",
            }
        ],
    )

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=80, uirevision="snake-termination")
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig


# =============================================================================
# Evaluation vs Training Score Comparison (optional)
# =============================================================================


def render_eval_comparison(
    train_scores: list[int],
    eval_scores: list[int],
) -> go.Figure | None:
    """Render comparison of training vs evaluation scores."""
    fig = go.Figure()

    if train_scores:
        episodes = list(range(len(train_scores)))
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=train_scores,
                mode="lines",
                name="Training (exploratory)",
                line={"color": "#4488ff", "width": 1},
                opacity=0.6,
            )
        )

    if eval_scores:
        # Plot at the end of training x-axis
        x_offset = len(train_scores)
        eval_x = list(range(x_offset, x_offset + len(eval_scores)))
        fig.add_trace(
            go.Scatter(
                x=eval_x,
                y=eval_scores,
                mode="markers+lines",
                name="Evaluation (greedy)",
                line={"color": "#44cc44", "width": 2},
                marker={"size": 8, "symbol": "diamond"},
            )
        )

    if not train_scores and not eval_scores:
        return None

    fig.update_layout({
        "title": {"text": "Training vs Evaluation Scores", "font": {"size": 11}},
        "xaxis_title": "Episode / Eval Run",
        "yaxis_title": "Score",
    })

    # Apply template FIRST (base layout)
    fig = apply_template(fig, height=CHART_HEIGHT_MAIN, uirevision="snake-eval-comparison")
    fig.update_layout({"margin": {"l": 50, "r": 20, "t": 35, "b": 40}})
    return fig


# =============================================================================
# Bellman Update Explanation
# =============================================================================


def render_bellman_update(details: dict | None) -> go.Figure | None:
    """Render the Bellman update equation for a sampled replay transition.

    Shows: y = r + γ(1-terminated)max_a Q_target(s',a)
    and: TD error = y - Q_online(s,a)
    """
    if not details:
        fig = go.Figure()
        fig.add_annotation({
            "x": 0.5, "y": 0.5, "xref": "paper", "yref": "paper",
            "text": "No Bellman update available\n(Replay not active or no recent training)",
            "showarrow": False, "font": {"size": 12, "color": "rgba(128,128,128,0.8)"}
        })
        fig = apply_template(fig, height=300, uirevision="snake-bellman")
        fig.update_layout({"margin": {"l": 50, "r": 20, "t": 35, "b": 40}})
        return fig

    # Extract values
    state = details.get("state", np.zeros(11))
    action = details.get("action", 0)
    reward = details.get("reward", 0.0)
    next_state = details.get("next_state", np.zeros(11))
    terminated = details.get("terminated", False)
    current_q = details.get("current_q", np.zeros(3))
    current_q_taken = details.get("current_q_taken", 0.0)
    target_q = details.get("target_q", 0.0)
    td_error = details.get("td_error", 0.0)
    target_q_max = details.get("target_q_max", 0.0)
    greedy_action = details.get("greedy_action", 0)
    gamma = details.get("gamma", 0.99)

    fig = go.Figure()

    # State vector (compact)
    fig.add_trace(go.Bar(
        x=state,
        y=[f"s[{i}]" for i in range(len(state))],
        orientation="h",
        marker_color="rgba(100,100,255,0.3)",
        name="State s",
        showlegend=False,
        hovertemplate="State[%{y}]: %{x:.2f}<extra></extra>",
    ))

    # Equation display as annotations
    terminated_str = "1" if terminated else "0"
    equation_lines = [
        f"Bellman Target: y = r + γ(1-d) max_a' Q'(s',a')",
        f"                = {reward:.3f} + {gamma:.3f}(1-{terminated_str}) × {target_q_max:.3f}",
        f"                = {target_q:.3f}",
        f"",
        f"Online Q(s,a): Q(s,a) = {current_q_taken:.3f}",
        f"",
        f"TD Error: δ = y - Q(s,a) = {target_q:.3f} - {current_q_taken:.3f} = {td_error:.3f}",
    ]

    y_pos = 1.0
    for line in equation_lines:
        fig.add_annotation(
            x=0.02, y=y_pos, xref="paper", yref="paper",
            text=line, showarrow=False,
            font={"size": 11, "family": "monospace", "color": "white"},
            xanchor="left", yanchor="top",
            bgcolor="rgba(0,0,0,0.5)", bordercolor="#4488ff", borderwidth=1, borderpad=4,
        )
        y_pos -= 0.12

    # Action info
    action_names = ["Straight", "Right", "Left"]
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text=f"Action taken: {action_names[action]} ({action})<br>Target-network greedy action at s': {action_names[greedy_action]} ({greedy_action})<br>Reward: {reward:.2f}<br>Terminated: {terminated}",
        showarrow=False, font={"size": 10, "color": "white"},
        xanchor="right", yanchor="top",
        bgcolor="rgba(0,0,0,0.7)", bordercolor="#4488ff", borderwidth=1, borderpad=6,
    )

    # Next state
    fig.add_trace(go.Bar(
        x=next_state,
        y=[f"s'[{i}]" for i in range(len(next_state))],
        orientation="h",
        marker_color="rgba(255,150,100,0.3)",
        name="Next State s'",
        showlegend=False,
        hovertemplate="Next[%{y}]: %{x:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title={"text": "Bellman Update (Sampled Replay Transition)", "font": {"size": 11}},
        barmode="overlay",
        xaxis={"range": [-0.1, 1.1], "showgrid": True, "gridcolor": "rgba(128,128,128,0.15)"},
        yaxis={"showgrid": False, "autorange": "reversed"},
        height=350,
        margin={"l": 80, "r": 20, "t": 45, "b": 40},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    fig = apply_template(fig, height=350, uirevision="snake-bellman")
    return fig
