"""Coordinate-only Snake renderer using Streamlit Components v2."""

import streamlit as st

_JS = r"""
export default function ({parentElement, data}) {
    const canvas = parentElement.querySelector("canvas");
    if (!canvas || !data) return;
    const signature = JSON.stringify(data);
    if (canvas.dataset.signature === signature) return;
    const ctx = canvas.getContext("2d");
    const n = data.grid_size;
    if (!ctx || !Number.isInteger(n) || n < 1) return;
    const size = canvas.width;
    const cell = size / n;
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#1e1e2e";
    ctx.fillRect(0, 0, size, size);
    if (data.show_grid) {
        ctx.strokeStyle = "rgba(160,160,160,0.15)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const p = i * cell;
            ctx.moveTo(p, 0); ctx.lineTo(p, size);
            ctx.moveTo(0, p); ctx.lineTo(size, p);
        }
        ctx.stroke();
    }
    data.snake.forEach(([x, y], i) => {
        ctx.fillStyle = i === 0 ? "#32cd32" : "#228b22";
        ctx.fillRect(x * cell + 1, y * cell + 1, cell - 2, cell - 2);
    });
    if (data.snake.length) {
        const [x, y] = data.snake[0];
        const eyes = [
            [[0.75, 0.3], [0.75, 0.7]],
            [[0.3, 0.75], [0.7, 0.75]],
            [[0.25, 0.3], [0.25, 0.7]],
            [[0.3, 0.25], [0.7, 0.25]],
        ][data.direction];
        ctx.fillStyle = "white";
        for (const [dx, dy] of eyes) {
            ctx.beginPath();
            ctx.arc((x + dx) * cell, (y + dy) * cell, cell * 0.08, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
    if (data.food !== null) {
        ctx.fillStyle = "#ff4444";
        ctx.beginPath();
        ctx.arc((data.food[0] + 0.5) * cell,
                (data.food[1] + 0.5) * cell, cell * 0.38, 0, 2 * Math.PI);
        ctx.fill();
    }
    if (data.game_over) {
        ctx.fillStyle = "rgba(0,0,0,0.65)";
        ctx.fillRect(0, size * 0.4, size, size * 0.2);
        ctx.fillStyle = "#ff6666";
        ctx.font = "bold 42px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("GAME OVER", size / 2, size * 0.525);
    }
    canvas.dataset.signature = signature;
}
"""

_canvas = st.components.v2.component(
    "snake_board_canvas",
    html='<canvas width="600" height="600" role="img" aria-label="Live Snake board"></canvas>',
    css="canvas {display:block; width:100%; aspect-ratio:1; border:1px solid #444; border-radius:6px; box-sizing:border-box;}",
    js=_JS,
)


def render_game_canvas(state, *, show_grid=False, key="live_game_canvas"):
    """Send a copied, JSON-safe board; never transmit image bytes."""
    food = state.get("food")
    data = {
        "grid_size": int(state["grid_size"]),
        "snake": [[int(x), int(y)] for x, y in state["snake"]],
        "food": None if food is None else [int(food[0]), int(food[1])],
        "direction": int(state.get("direction", 0)) % 4,
        "game_over": bool(state.get("game_over", False)),
        "show_grid": bool(show_grid),
    }
    if data["grid_size"] < 1:
        raise ValueError("grid_size must be positive")
    return _canvas(data=data, key=key, width="stretch", height="content")
