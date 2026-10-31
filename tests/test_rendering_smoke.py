"""Backend smoke tests; these do not replace browser animation tests."""

import ast
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from snake_qdn import SnakeEnv


def assert_clean(app):
    assert not app.exception, [error.message for error in app.exception]


def main():
    # Verify new UI module structure exists
    tree = ast.parse(Path("app.py").read_text())
    func_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "main" in func_names, "main() function not found"

    # Verify old fragmented chart functions are gone (top-level fragment definitions)
    source = Path("app.py").read_text()
    assert "live_charts_area" not in source, "Old live_charts_area should be removed"
    # Current architecture has 3 fragments: execution (game board), metrics, inspect_learning
    assert source.count("@st.fragment") == 3, (
        f"Should have exactly 3 fragments (execution, metrics, inspect), found {source.count('@st.fragment')}"
    )
    assert "_loss_slot" not in source
    assert "_epsilon_slot" not in source
    assert "chart_versions" not in source

    # Test AppTest with new structure
    app = AppTest.from_file("app.py").run(timeout=60)
    assert_clean(app)

    # Test that controller is in session state
    assert "controller" in app.session_state, "controller should be in session_state"

    # Test basic interaction - click step button
    # (Need to set up controller properly for AppTest)
    # The controller is already created by the app, so we just interact with it
    # The new toolbar has key="toolbar_step"
    step_buttons = app.get("button")
    step_button = next((b for b in step_buttons if b.key == "toolbar_step"), None)
    if step_button:
        step_button.click().run(timeout=60)
        assert_clean(app)

    # Test mode change
    mode_radios = app.get("radio")
    mode_radio = next((r for r in mode_radios if r.key == "toolbar_mode"), None)
    if mode_radio:
        mode_radio.set_value("Evaluate").run(timeout=60)
        assert_clean(app)
        mode_radio.set_value("Train").run(timeout=60)
        assert_clean(app)

    # Test game board rendering (using Plotly, not canvas)
    # Test a second run (unchanged data rerun)
    app.run(timeout=60)
    assert_clean(app)

    # Test step once button
    step_buttons = app.get("button")
    step_button = next((b for b in step_buttons if b.key == "toolbar_step"), None)
    if step_button:
        step_button.click().run(timeout=60)
        assert_clean(app)

    # Test environment rendering works
    env = SnakeEnv(grid_size=10, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=42)
        state = env.get_state_for_display()
        assert state["grid_size"] == 10
        json.dumps(state, default=lambda value: value.tolist())
    finally:
        env.close()


if __name__ == "__main__":
    main()
