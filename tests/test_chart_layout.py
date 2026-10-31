"""
Layout unit tests for apply_template helper.

Tests that the template helper correctly applies layout properties
regardless of Plotly's default property membership behavior.
"""

import plotly.graph_objects as go
import pytest

from ui.charts import apply_template, get_plotly_template


def _layout_to_dict(layout_obj) -> dict:
    """Convert Plotly layout object to plain dict for comparison."""
    return {key: getattr(layout_obj, key) for key in layout_obj}


def _margin_to_dict(margin_obj) -> dict:
    """Convert Plotly Margin object to plain dict with only l,r,t,b."""
    return {k: getattr(margin_obj, k) for k in ("l", "r", "t", "b")}


def _font_to_dict(font_obj) -> dict:
    """Convert Plotly Font object to plain dict with only family,color,size (non-None)."""
    return {
        k: getattr(font_obj, k)
        for k in ("family", "color", "size")
        if hasattr(font_obj, k) and getattr(font_obj, k) is not None
    }


def _legend_to_dict(legend_obj, template_keys) -> dict:
    """Convert Plotly Legend object to plain dict with only template keys."""
    result = {}
    for key in template_keys:
        if hasattr(legend_obj, key):
            value = getattr(legend_obj, key)
            if key == "font" and hasattr(value, "__dict__"):
                # Nested Font object
                result[key] = _font_to_dict(value)
            else:
                result[key] = value
    return result


def _axis_to_dict(axis_obj, template_keys) -> dict:
    """Convert Plotly Axis object to plain dict with only template keys."""
    return {key: getattr(axis_obj, key) for key in template_keys if hasattr(axis_obj, key)}


def test_apply_template_applies_all_template_properties():
    """Template properties should be applied even if Plotly reports them as present."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))

    # Apply template
    result = apply_template(fig, height=300, uirevision="test")

    template = get_plotly_template()
    for key, expected_value in template["layout"].items():
        # The property should be explicitly set to the template value
        actual_value = getattr(result.layout, key)
        if key == "margin":
            assert _margin_to_dict(actual_value) == expected_value, (
                f"Property {key} not correctly applied"
            )
        elif key == "font":
            assert _font_to_dict(actual_value) == expected_value, (
                f"Property {key} not correctly applied"
            )
        elif key in ("xaxis", "yaxis"):
            assert _axis_to_dict(actual_value, expected_value.keys()) == expected_value, (
                f"Property {key} not correctly applied"
            )
        elif key == "legend":
            assert _legend_to_dict(actual_value, expected_value.keys()) == expected_value, (
                f"Property {key} not correctly applied"
            )
        else:
            assert actual_value == expected_value, f"Property {key} not correctly applied"


def test_apply_template_overrides_existing_margin():
    """Explicit margin on figure should be overridden by template (then chart-specific can override)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
    # Set an explicit margin before applying template
    fig.update_layout(margin={"l": 100, "r": 100, "t": 100, "b": 100})

    result = apply_template(fig, height=300)

    # Template margin should be applied
    template = get_plotly_template()
    assert _margin_to_dict(result.layout.margin) == template["layout"]["margin"]


def test_apply_template_sets_height_and_uirevision():
    """Height and uirevision should be set after template."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))

    result = apply_template(fig, height=360, uirevision="test-revision")

    assert result.layout.height == 360
    assert result.layout.uirevision == "test-revision"


def test_apply_template_does_not_use_membership_check():
    """The helper should NOT use 'key in fig.layout' to detect existing properties.

    This test documents the bug that was fixed: Plotly's 'key in fig.layout' returns True
    for properties that exist as defaults but weren't explicitly set by user code.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))

    # These properties exist in Plotly's default layout but weren't explicitly set
    assert "margin" in fig.layout
    assert "font" in fig.layout
    assert "paper_bgcolor" in fig.layout
    assert "plot_bgcolor" in fig.layout

    # After apply_template, they should have template values
    result = apply_template(fig)
    template = get_plotly_template()

    assert _margin_to_dict(result.layout.margin) == template["layout"]["margin"]
    assert _font_to_dict(result.layout.font) == template["layout"]["font"]
    assert result.layout.paper_bgcolor == template["layout"]["paper_bgcolor"]
    assert result.layout.plot_bgcolor == template["layout"]["plot_bgcolor"]


def test_construction_order_template_then_chart_specific():
    """Correct construction order: base template -> chart-specific -> final dimensions."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))

    # Step 1: Apply base template
    fig = apply_template(fig)
    template = get_plotly_template()
    assert _margin_to_dict(fig.layout.margin) == template["layout"]["margin"]

    # Step 2: Chart-specific layout (e.g., state vector needs larger left margin)
    fig.update_layout(margin={"l": 120, "r": 20, "t": 45, "b": 25})
    assert fig.layout.margin.l == 120

    # Step 3: Final dimensions
    fig.update_layout(height=360, uirevision="test")
    assert fig.layout.height == 360


def test_state_vector_margins_sufficient_for_labels():
    """State vector chart should have sufficient left margin for 22 labels."""
    import numpy as np

    from ui.charts import render_state_vector
    from observation_schema import OBSERVATION_SIZE

    obs = np.ones(OBSERVATION_SIZE, dtype=np.float32)
    fig = render_state_vector(obs)

    assert fig is not None
    assert fig.layout.height == 650
    assert fig.layout.margin.l >= 140  # Sufficient for labels
    assert fig.layout.yaxis.autorange == "reversed"
    assert fig.layout.yaxis.tickfont.size >= 11


def test_q_values_margins_sufficient_for_annotations():
    """Q-values chart should have sufficient margins for executed/preferred annotations."""
    import numpy as np

    from ui.charts import render_q_values

    q_vals = np.array([1.0, 0.5, -0.2], dtype=np.float32)
    fig = render_q_values(
        q_vals, action_executed=0, action_preferred=1, selection_reason="exploring"
    )

    assert fig is not None
    assert fig.layout.height == 320  # CHART_HEIGHT_QVALUES
    assert fig.layout.margin.t >= 50  # Top margin for gap annotation
    assert fig.layout.margin.b >= 40  # Bottom margin for executed annotation


def test_network_activations_margins_sufficient():
    """Network activations chart should have sufficient margins for layer labels."""

    from training import TrainingController
    from ui.charts import render_network_activations

    # Create a minimal controller and snapshot
    controller = TrainingController()
    controller.set_mode(controller.mode.__class__.TRAIN)
    controller.restart_board()

    snapshot = controller.get_snapshot()
    fig = render_network_activations(snapshot, controller, model_revision=1)

    assert fig is not None
    assert fig.layout.height == 360  # CHART_HEIGHT_NETWORK
    assert fig.layout.margin.t >= 40  # Top margin for layer labels
    assert fig.layout.margin.b >= 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
