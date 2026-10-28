"""Backend smoke tests; these do not replace browser animation tests."""
import ast
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
from snake_env import SnakeGameAI
from agent import Agent


def assert_clean(app):
    assert not app.exception, [error.message for error in app.exception]


def main():
    tree = ast.parse(Path("app.py").read_text())
    charts = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "live_charts_area"
    )
    source = ast.unparse(charts)
    assert "_loss_slot" not in source
    assert "_epsilon_slot" not in source
    assert "chart_versions" not in source
    assert not any(isinstance(node, ast.Return) for node in ast.walk(charts))

    app = AppTest.from_file("app.py").run(timeout=60)
    assert_clean(app)
    env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=42)
        agent = Agent(hidden_size=64, batch_size=32, device="cpu")
        app.session_state["env"] = env
        app.session_state["agent"] = agent
        app.session_state["current_observation"] = observation
        app.session_state["game_state"] = env.get_state_for_display()
        app.session_state["game_over"] = False
        app.session_state["training"] = False
        app.session_state["paused"] = True
        app.run(timeout=60)
        assert_clean(app)
        app.run(timeout=60)
        assert_clean(app)
        app.button(key="step_btn").click().run(timeout=60)
        assert_clean(app)
        from game_canvas import render_game_canvas
        assert callable(render_game_canvas)
        state = env.get_state_for_display()
        assert state["grid_size"] == 10
        json.dumps(state, default=lambda value: value.tolist())
    finally:
        env.close()


if __name__ == "__main__":
    main()
