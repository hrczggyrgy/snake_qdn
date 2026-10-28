# Rendering patch

Apply render-game-and-charts.patch to main at commit 63abcfc or compatible code.
This supersedes fix-fragment-layout.patch; do not apply both patches.

    git apply --check render-game-and-charts.patch
    git apply render-game-and-charts.patch
    python -m pip install -r requirements.txt
    python -m compileall -q app.py game_canvas.py
    streamlit run app.py

The game uses a Components v2 canvas with a fixed key and coordinate-only data.
The PNG path remains only as a fallback when no board state exists.
Charts own their fragment layout and all three are rendered on every tick.
Existing history windows and Plotly uirevision settings are retained.

This does not move training into a worker, change training schedules, or promise
real-time frame rates. The caller may still construct an RGB frame before the
canvas call. Benchmark and browser-test on the deployed Streamlit version.
Check pause/resume, unchanged chart ticks, history reset, new agent, food=None,
game over, different grid sizes, and browser resizing before deployment.
