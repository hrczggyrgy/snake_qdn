"""
Theme-aware styling tokens and CSS for Snake DQN Dashboard.

Provides a small, consistent set of spacing, typography, radius, and color tokens.
Scopes all custom CSS to app-owned elements (prefixed with .sqdn-).
Supports both light and dark Streamlit themes.
"""

# =============================================================================
# Design Tokens
# =============================================================================

# Spacing scale (rem)
SPACE_1 = "0.25rem"   # 4px
SPACE_2 = "0.5rem"    # 8px
SPACE_3 = "0.75rem"   # 12px
SPACE_4 = "1rem"      # 16px
SPACE_5 = "1.5rem"    # 24px
SPACE_6 = "2rem"      # 32px

# Typography
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
FONT_SANS = "ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
TEXT_XS = "0.75rem"   # 12px
TEXT_SM = "0.875rem"  # 14px
TEXT_BASE = "1rem"    # 16px
TEXT_LG = "1.125rem"  # 18px
TEXT_XL = "1.5rem"    # 24px
TEXT_2XL = "2rem"     # 32px

# Border radius
RADIUS_SM = "4px"
RADIUS_MD = "6px"
RADIUS_LG = "8px"
RADIUS_FULL = "9999px"

# Color tokens (CSS custom properties for theme awareness)
# These map to Streamlit's theme colors
COLORS = """
:root {
    /* Spacing */
    --sqdn-space-1: 0.25rem;
    --sqdn-space-2: 0.5rem;
    --sqdn-space-3: 0.75rem;
    --sqdn-space-4: 1rem;
    --sqdn-space-5: 1.5rem;
    --sqdn-space-6: 2rem;

    /* Typography */
    --sqdn-font-mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --sqdn-font-sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    --sqdn-text-xs: 0.75rem;
    --sqdn-text-sm: 0.875rem;
    --sqdn-text-base: 1rem;
    --sqdn-text-lg: 1.125rem;
    --sqdn-text-xl: 1.5rem;
    --sqdn-text-2xl: 2rem;

    /* Radius */
    --sqdn-radius-sm: 4px;
    --sqdn-radius-md: 6px;
    --sqdn-radius-lg: 8px;
    --sqdn-radius-full: 9999px;

    /* Semantic colors - map to Streamlit theme */
    --sqdn-bg-primary: var(--background-color, #0e1117);
    --sqdn-bg-secondary: var(--secondary-background-color, #262730);
    --sqdn-bg-tertiary: var(--secondary-background-color, #262730);
    --sqdn-text-primary: var(--text-color, #fafafa);
    --sqdn-text-secondary: var(--text-color, #fafafa);
    --sqdn-text-muted: rgba(var(--text-color-rgb, 250, 250, 250), 0.6);
    --sqdn-border: rgba(var(--text-color-rgb, 250, 250, 250), 0.15);
    --sqdn-border-strong: rgba(var(--text-color-rgb, 250, 250, 250), 0.3);

    /* Accent colors (consistent across themes) */
    --sqdn-accent-blue: #4488ff;
    --sqdn-accent-green: #44cc44;
    --sqdn-accent-yellow: #ffd700;
    --sqdn-accent-red: #ff4444;
    --sqdn-accent-orange: #ff8800;
    --sqdn-accent-purple: #aa44ff;

    /* State colors */
    --sqdn-state-train: #4488ff;
    --sqdn-state-evaluate: #44cc44;
    --sqdn-state-manual: #ffd700;
    --sqdn-state-paused: #888888;
    --sqdn-state-running: #44cc44;

    /* Component backgrounds */
    --sqdn-card-bg: var(--sqdn-bg-secondary);
    --sqdn-card-border: var(--sqdn-border);
    --sqdn-panel-bg: rgba(30, 30, 46, 0.9);
    --sqdn-panel-border: #444;
}

/* Dark theme overrides (Streamlit default) */
@media (prefers-color-scheme: dark) {
    :root {
        --sqdn-bg-primary: #0e1117;
        --sqdn-bg-secondary: #262730;
        --sqdn-text-primary: #fafafa;
        --sqdn-text-muted: rgba(250, 250, 250, 0.6);
        --sqdn-border: rgba(250, 250, 250, 0.15);
    }
}

/* Light theme overrides */
@media (prefers-color-scheme: light) {
    :root {
        --sqdn-bg-primary: #ffffff;
        --sqdn-bg-secondary: #f0f2f6;
        --sqdn-text-primary: #262730;
        --sqdn-text-muted: rgba(38, 39, 48, 0.6);
        --sqdn-border: rgba(38, 39, 48, 0.15);
        --sqdn-panel-bg: rgba(240, 242, 246, 0.95);
        --sqdn-panel-border: #ccc;
    }
}

/* Streamlit theme class overrides */
[data-theme="dark"] {
    --sqdn-bg-primary: #0e1117;
    --sqdn-bg-secondary: #262730;
    --sqdn-text-primary: #fafafa;
    --sqdn-text-muted: rgba(250, 250, 250, 0.6);
    --sqdn-border: rgba(250, 250, 250, 0.15);
    --sqdn-panel-bg: rgba(30, 30, 46, 0.9);
    --sqdn-panel-border: #444;
}

[data-theme="light"] {
    --sqdn-bg-primary: #ffffff;
    --sqdn-bg-secondary: #f0f2f6;
    --sqdn-text-primary: #262730;
    --sqdn-text-muted: rgba(38, 39, 48, 0.6);
    --sqdn-border: rgba(38, 39, 48, 0.15);
    --sqdn-panel-bg: rgba(240, 242, 246, 0.95);
    --sqdn-panel-border: #ccc;
}
"""

# =============================================================================
# Component CSS Classes
# =============================================================================

COMPONENT_CSS = """
/* ============================================================================
   Layout Containers
   ============================================================================ */
.sqdn-page {
    padding: var(--sqdn-space-4);
    max-width: 100%;
}

.sqdn-toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sqdn-space-3);
    align-items: center;
    padding: var(--sqdn-space-3) var(--sqdn-space-4);
    background: var(--sqdn-card-bg);
    border: 1px solid var(--sqdn-card-border);
    border-radius: var(--sqdn-radius-lg);
    margin-bottom: var(--sqdn-space-4);
}

.sqdn-metrics-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sqdn-space-3);
    margin-bottom: var(--sqdn-space-4);
}

.sqdn-main-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--sqdn-space-4);
    margin-bottom: var(--sqdn-space-4);
}

@media (max-width: 1024px) {
    .sqdn-main-row {
        grid-template-columns: 1fr;
    }
}

.sqdn-section-full {
    width: 100%;
    margin-bottom: var(--sqdn-space-4);
}

.sqdn-diagnostics-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--sqdn-space-3);
    margin-bottom: var(--sqdn-space-4);
}

/* ============================================================================
   Cards & Panels
   ============================================================================ */
.sqdn-card {
    background: var(--sqdn-card-bg);
    border: 1px solid var(--sqdn-card-border);
    border-radius: var(--sqdn-radius-lg);
    padding: var(--sqdn-space-4);
}

.sqdn-panel {
    background: var(--sqdn-panel-bg);
    border: 1px solid var(--sqdn-panel-border);
    border-radius: var(--sqdn-radius-md);
    padding: var(--sqdn-space-3);
}

.sqdn-panel-compact {
    background: var(--sqdn-panel-bg);
    border: 1px solid var(--sqdn-panel-border);
    border-radius: var(--sqdn-radius-md);
    padding: var(--sqdn-space-2) var(--sqdn-space-3);
}

/* ============================================================================
   Game Board Container
   ============================================================================ */
.sqdn-game-container {
    width: 380px;
    height: 380px;
    margin: 0 auto;
    border: 2px solid var(--sqdn-panel-border);
    border-radius: var(--sqdn-radius-lg);
    overflow: hidden;
    background: var(--sqdn-bg-primary);
    flex-shrink: 0;
}

/* ============================================================================
   Decision Strip
   ============================================================================ */
.sqdn-decision-strip {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sqdn-space-2) var(--sqdn-space-3);
    align-items: center;
    padding: var(--sqdn-space-2) var(--sqdn-space-3);
    background: var(--sqdn-panel-bg);
    border: 1px solid var(--sqdn-panel-border);
    border-radius: var(--sqdn-radius-md);
    font-family: var(--sqdn-font-mono);
    font-size: var(--sqdn-text-sm);
    min-width: 0;
    overflow-wrap: anywhere;
    width: 100%;
    box-sizing: border-box;
}

.sqdn-decision-strip .sqdn-label {
    font-weight: 600;
    white-space: nowrap;
}

.sqdn-decision-strip .sqdn-detail {
    color: var(--sqdn-text-muted);
    white-space: nowrap;
}

.sqdn-decision-strip .sqdn-decision-q-values {
    color: var(--sqdn-text-muted);
    font-size: var(--sqdn-text-xs);
    min-width: 0;
    overflow-wrap: anywhere;
}

/* Reason colors */
.sqdn-reason-exploring { color: var(--sqdn-accent-blue); }
.sqdn-reason-exploiting { color: var(--sqdn-accent-green); }
.sqdn-reason-manual { color: var(--sqdn-accent-yellow); }
.sqdn-reason-unknown { color: var(--sqdn-text-muted); }

/* ============================================================================
   Event Feed
   ============================================================================ */
.sqdn-event-feed {
    display: flex;
    flex-direction: column;
    gap: var(--sqdn-space-1);
    max-height: 120px;
    overflow-y: auto;
}

.sqdn-event {
    display: flex;
    align-items: center;
    gap: var(--sqdn-space-2);
    padding: var(--sqdn-space-1) var(--sqdn-space-2);
    background: var(--sqdn-panel-bg);
    border: 1px solid var(--sqdn-panel-border);
    border-radius: var(--sqdn-radius-sm);
    font-size: var(--sqdn-text-sm);
    border-left-width: 3px;
}

.sqdn-event-info { border-left-color: var(--sqdn-accent-blue); }
.sqdn-event-success { border-left-color: var(--sqdn-accent-green); }
.sqdn-event-warning { border-left-color: var(--sqdn-accent-yellow); }
.sqdn-event-error { border-left-color: var(--sqdn-accent-red); }

.sqdn-event-icon { flex-shrink: 0; }
.sqdn-event-message { flex: 1; min-width: 0; }

/* ============================================================================
   Outcome Summary (Horizontal Stacked Bar)
   ============================================================================ */
.sqdn-outcome-summary {
    background: var(--sqdn-panel-bg);
    border: 1px solid var(--sqdn-panel-border);
    border-radius: var(--sqdn-radius-md);
    padding: var(--sqdn-space-3);
}

.sqdn-outcome-header {
    font-weight: 600;
    margin-bottom: var(--sqdn-space-2);
    font-size: var(--sqdn-text-sm);
}

.sqdn-outcome-bar {
    display: flex;
    height: 28px;
    border-radius: var(--sqdn-radius-sm);
    overflow: hidden;
    background: var(--sqdn-bg-tertiary);
    margin-bottom: var(--sqdn-space-2);
}

.sqdn-outcome-segment {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 var(--sqdn-space-2);
    font-size: var(--sqdn-text-xs);
    font-weight: 600;
    color: white;
    white-space: nowrap;
}

.sqdn-outcome-legend {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sqdn-space-2) var(--sqdn-space-3);
    font-size: var(--sqdn-text-xs);
    color: var(--sqdn-text-muted);
}

.sqdn-outcome-legend-item {
    display: flex;
    align-items: center;
    gap: var(--sqdn-space-1);
}

.sqdn-outcome-legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: var(--sqdn-radius-sm);
}

/* ============================================================================
   Metric Cards
   ============================================================================ */
.sqdn-metric {
    flex: 1;
    min-width: 120px;
    background: var(--sqdn-card-bg);
    border: 1px solid var(--sqdn-card-border);
    border-radius: var(--sqdn-radius-md);
    padding: var(--sqdn-space-3);
    text-align: center;
}

.sqdn-metric-label {
    font-size: var(--sqdn-text-xs);
    color: var(--sqdn-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: var(--sqdn-space-1);
}

.sqdn-metric-value {
    font-size: var(--sqdn-text-xl);
    font-weight: 700;
    font-family: var(--sqdn-font-mono);
    color: var(--sqdn-text-primary);
}

/* ============================================================================
   Inspection Components
   ============================================================================ */
.sqdn-inspection-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--sqdn-space-3);
    padding-bottom: var(--sqdn-space-2);
    border-bottom: 1px solid var(--sqdn-border);
}

.sqdn-inspection-title {
    font-size: var(--sqdn-text-lg);
    font-weight: 600;
}

.sqdn-inspection-subtitle {
    font-size: var(--sqdn-text-sm);
    color: var(--sqdn-text-muted);
}

.sqdn-state-vector {
    height: 220px;
}

.sqdn-q-values-chart {
    height: 180px;
}

.sqdn-network-activations {
    height: 200px;
}

/* ============================================================================
   Chart Containers
   ============================================================================ */
.sqdn-chart-container {
    background: var(--sqdn-card-bg);
    border: 1px solid var(--sqdn-card-border);
    border-radius: var(--sqdn-radius-lg);
    padding: var(--sqdn-space-3);
    min-height: 220px;
}

.sqdn-chart-container-sm {
    background: var(--sqdn-card-bg);
    border: 1px solid var(--sqdn-card-border);
    border-radius: var(--sqdn-radius-lg);
    padding: var(--sqdn-space-3);
    min-height: 180px;
}

/* ============================================================================
   Buttons (Scoped to avoid global overrides)
   ============================================================================ */
.sqdn-toolbar .stButton > button,
.sqdn-sidebar .stButton > button {
    border-radius: var(--sqdn-radius-md);
    font-weight: 600;
    padding: var(--sqdn-space-1) var(--sqdn-space-3);
    transition: all 0.15s ease;
}

.sqdn-toolbar .stButton > button[kind="primary"],
.sqdn-sidebar .stButton > button[kind="primary"] {
    background: var(--sqdn-accent-blue);
    border-color: var(--sqdn-accent-blue);
}

.sqdn-toolbar .stButton > button:hover:not(:disabled),
.sqdn-sidebar .stButton > button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* ============================================================================
   Plotly Chart Background
   ============================================================================ */
.sqdn-chart-container .stPlotlyChart,
.sqdn-chart-container-sm .stPlotlyChart {
    background: transparent !important;
}

.sqdn-chart-container .js-plotly-plot,
.sqdn-chart-container-sm .js-plotly-plot {
    background: transparent !important;
}

/* ============================================================================
   Empty State
   ============================================================================ */
.sqdn-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--sqdn-space-6);
    color: var(--sqdn-text-muted);
    text-align: center;
}

.sqdn-empty-state-icon {
    font-size: var(--sqdn-text-2xl);
    margin-bottom: var(--sqdn-space-2);
    opacity: 0.5;
}

/* ============================================================================
   Status Badge
   ============================================================================ */
.sqdn-status-badge {
    display: inline-flex;
    align-items: center;
    gap: var(--sqdn-space-1);
    padding: var(--sqdn-space-1) var(--sqdn-space-2);
    border-radius: var(--sqdn-radius-full);
    font-size: var(--sqdn-text-sm);
    font-weight: 500;
}

.sqdn-status-train { background: rgba(68, 136, 255, 0.2); color: var(--sqdn-accent-blue); }
.sqdn-status-evaluate { background: rgba(68, 204, 68, 0.2); color: var(--sqdn-accent-green); }
.sqdn-status-manual { background: rgba(255, 215, 0, 0.2); color: var(--sqdn-accent-yellow); }
.sqdn-status-paused { background: rgba(136, 136, 136, 0.2); color: var(--sqdn-text-muted); }
.sqdn-status-running { background: rgba(68, 204, 68, 0.2); color: var(--sqdn-accent-green); }

/* ============================================================================
   Disclosure Notice
   ============================================================================ */
.sqdn-disclosure {
    font-size: var(--sqdn-text-xs);
    color: var(--sqdn-text-muted);
    font-style: italic;
    padding: var(--sqdn-space-2);
    background: var(--sqdn-bg-tertiary);
    border-radius: var(--sqdn-radius-sm);
    border: 1px dashed var(--sqdn-border);
}
"""

# =============================================================================
# Public API
# =============================================================================

def inject_global_styles():
    """Inject global CSS tokens and component styles."""
    import streamlit as st
    st.markdown(f"<style>{COLORS}\n{COMPONENT_CSS}</style>", unsafe_allow_html=True)


def get_plotly_template():
    """Return a Plotly template dict consistent with theme tokens."""
    return {
        "layout": {
            "font": {"family": "ui-sans-serif, system-ui, sans-serif", "color": "var(--sqdn-text-primary)"},
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {
                "showgrid": True,
                "gridcolor": "rgba(128,128,128,0.15)",
                "zeroline": False,
                "showline": False,
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": "rgba(128,128,128,0.15)",
                "zeroline": False,
                "showline": False,
            },
            "margin": {"l": 40, "r": 10, "t": 30, "b": 30},
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "font": {"size": 9},
                "bgcolor": "rgba(0,0,0,0)",
            },
        }
    }


# Board dimensions (single source of truth)
BOARD_SIZE = 380
BOARD_MARGIN = {"l": 5, "r": 5, "t": 5, "b": 5}

# Chart heights - increased for readable inspection charts
CHART_HEIGHT_MAIN = 300
CHART_HEIGHT_SMALL = 220
CHART_HEIGHT_STATE = 360
CHART_HEIGHT_QVALUES = 320
CHART_HEIGHT_NETWORK = 360

# State vector labels (grouped)
STATE_LABELS = [
    ("Danger", ["Straight", "Right", "Left"]),
    ("Direction", ["Left", "Right", "Up", "Down"]),
    ("Food", ["Left", "Right", "Up", "Down"]),
]

STATE_LABELS_FLAT = [
    "Danger Straight",
    "Danger Right",
    "Danger Left",
    "Dir Left",
    "Dir Right",
    "Dir Up",
    "Dir Down",
    "Food Left",
    "Food Right",
    "Food Up",
    "Food Down",
]

# Action names
ACTION_NAMES = ["Straight", "Right", "Left"]
ACTION_NAMES_SHORT = ["S", "R", "L"]

# Episode termination reasons
TERMINATION_REASONS = {
    "wall": "Wall Collision",
    "self": "Self Collision",
    "timeout": "Timeout",
    "victory": "Board Completed",
}
