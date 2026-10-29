# UI package for Snake DQN Dashboard
# This makes ui a proper Python package to avoid mypy duplicate module errors.

from ui.charts import (
    apply_template,
    render_epsilon_chart,
    render_eval_comparison,
    render_loss_chart,
    render_network_activations,
    render_q_values,
    render_score_chart,
    render_state_vector,
    render_termination_summary,
    render_throughput_chart,
)
from ui.controls import (
    render_manual_controls,
    render_primary_toolbar,
    render_sidebar,
    render_sidebar_agent,
    render_sidebar_checkpoints,
    render_sidebar_destructive,
    render_sidebar_display,
    render_sidebar_environment,
    render_sidebar_help,
    render_status_badge_inline,
)
from ui.game import (
    render_decision_strip,
    render_decision_strip_with_disclosure,
    render_event_feed,
    render_game_board,
    render_latest_event,
    render_metrics_row,
    render_outcome_summary,
)
from ui.inspection import (
    render_inspect_learning,
    render_inspect_learning_simple,
    render_training_diagnostics,
)
from ui.styles import (
    ACTION_NAMES,
    ACTION_NAMES_SHORT,
    BOARD_MARGIN,
    BOARD_SIZE,
    CHART_HEIGHT_MAIN,
    CHART_HEIGHT_NETWORK,
    CHART_HEIGHT_QVALUES,
    CHART_HEIGHT_SMALL,
    CHART_HEIGHT_STATE,
    STATE_LABELS_FLAT,
    get_plotly_template,
    inject_global_styles,
)

__all__ = [
    # charts
    "apply_template",
    "render_eval_comparison",
    "render_epsilon_chart",
    "render_loss_chart",
    "render_network_activations",
    "render_q_values",
    "render_score_chart",
    "render_state_vector",
    "render_termination_summary",
    "render_throughput_chart",
    # controls
    "render_manual_controls",
    "render_primary_toolbar",
    "render_sidebar",
    "render_sidebar_agent",
    "render_sidebar_checkpoints",
    "render_sidebar_destructive",
    "render_sidebar_display",
    "render_sidebar_environment",
    "render_sidebar_help",
    "render_status_badge_inline",
    # game
    "render_decision_strip",
    "render_decision_strip_with_disclosure",
    "render_event_feed",
    "render_game_board",
    "render_latest_event",
    "render_metrics_row",
    "render_outcome_summary",
    # inspection
    "render_inspect_learning",
    "render_inspect_learning_simple",
    "render_training_diagnostics",
    # styles
    "ACTION_NAMES",
    "ACTION_NAMES_SHORT",
    "BOARD_MARGIN",
    "BOARD_SIZE",
    "CHART_HEIGHT_MAIN",
    "CHART_HEIGHT_NETWORK",
    "CHART_HEIGHT_QVALUES",
    "CHART_HEIGHT_SMALL",
    "CHART_HEIGHT_STATE",
    "STATE_LABELS_FLAT",
    "get_plotly_template",
    "inject_global_styles",
]
