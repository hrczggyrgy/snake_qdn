"""
Streamlit Application for Snake DQN Training & Visualization.

A comprehensive dashboard with three main views:
1. Live Game Arena - Real-time game rendering and controls
2. Neural Network Internals - State vector, architecture, Q-values
3. Learning Analytics - Scores, loss curves, exploration tracking
"""

import streamlit as st
import numpy as np
import torch
import time
import json
import random
import base64
from pathlib import Path
from collections import deque
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# PIL for inline PNG encoding
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# Import our modules
from snake_env import SnakeGameAI
from model import Linear_QNet, QTrainer
from agent import Agent, ReplayMemory, train_agent, evaluate_agent


# ============================================================================
# Streamlit Version Compatibility
# ============================================================================
try:
    _ST_VERSION = tuple(map(int, st.__version__.split('.')[:2]))
except:
    _ST_VERSION = (1, 28)

def btn_full(label, **kwargs):
    """Button with full-width compatibility across Streamlit versions."""
    # Streamlit 1.37+ supports width='stretch', older versions use_container_width also works
    return st.button(label, width='stretch', **kwargs)


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Snake DQN - Live Training Dashboard",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
    div[data-testid="stPlotlyChart"] > div {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Game Frame Display (inline PNG for reliable rendering)
# ============================================================================
def display_game_frame(frame):
    """Display game frame using inline PNG to avoid Streamlit media delivery issues."""
    if frame is None:
        st.error("The environment returned no game frame.")
        return

    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        st.error(
            f"Invalid game frame: shape={frame.shape}, dtype={frame.dtype}"
        )
        return

    if not _PIL_AVAILABLE:
        st.error("PIL not available for inline PNG encoding")
        return

    buffer = BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    height, width = frame.shape[:2]

    st.markdown(
        f"""
        <div style="width:100%;aspect-ratio:{width}/{height};
                    background:#1e1e28;overflow:hidden;border:1px solid #333;border-radius:4px">
          <img src="data:image/png;base64,{encoded}"
               alt="Live Snake board"
               width="{width}" height="{height}"
               style="display:block;width:100%;height:100%;
                      object-fit:contain;image-rendering:pixelated">
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Session State Initialization
# ============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Environment
        'env': None,
        'env_config': {
            'grid_size': 20,
            'step_penalty': -0.01,
            'food_reward': 10.0,
            'death_penalty': -10.0
        },
        
        # Agent
        'agent': None,
        'agent_config': {
            'hidden_size': 256,
            'lr': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 100000,
            'batch_size': 1000,
        },
        
        # Game State
        'game_state': None,
        'current_observation': None,
        'current_q_values': None,
        'current_action': None,
        'current_action_probs': None,
        'game_over': False,
        'paused': True,
        
        # Training
        'training': False,
        'training_episode': 0,
        'training_step': 0,
        'training_stats': {
            'scores': [],
            'mean_scores': [],
            'epsilon': [],
            'loss': [],
            'high_score': 0
        },
        
        # UI Controls
        'speed': 50,  # ms per frame
        'speed_preset': 'Watch',  # Watch / Normal / Fast / Max
        'activity': 'Train',  # Train / Evaluate / Manual
        'show_network': False,
        
        # Visualization Data
        'score_history': deque(maxlen=500),
        'mean_score_history': deque(maxlen=500),
        'loss_history': deque(maxlen=500),
        'epsilon_history': deque(maxlen=500),
        
        # Network activations for visualization
        'activations': None,
        
        # Decision snapshot (pre-action)
        'decision_snapshot': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ============================================================================
# Helper Functions
# ============================================================================
def create_env():
    """Create or recreate the environment."""
    config = st.session_state.env_config
    env = SnakeGameAI(
        grid_size=config['grid_size'],
        step_penalty=config['step_penalty'],
        food_reward=config['food_reward'],
        death_penalty=config['death_penalty'],
        render_mode='rgb_array'
    )
    return env


def create_agent():
    """Create or recreate the agent."""
    config = st.session_state.agent_config
    agent = Agent(
        input_size=11,
        hidden_size=config['hidden_size'],
        output_size=3,
        lr=config['lr'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        device='auto'
    )
    return agent


def reset_game():
    """Reset the game to initial state."""
    if st.session_state.env is None:
        st.session_state.env = create_env()
    
    obs, info = st.session_state.env.reset()
    st.session_state.current_observation = obs
    st.session_state.game_state = st.session_state.env.get_state_for_display()
    st.session_state.game_over = False
    st.session_state.current_q_values = None
    st.session_state.current_action = None
    st.session_state.current_action_probs = None
    st.session_state.activations = None
    st.session_state.decision_snapshot = None  # Clear previous decision
    
    # Get initial Q-values if agent exists
    if st.session_state.agent is not None:
        st.session_state.current_q_values = st.session_state.agent.get_q_values(obs)
        # Compute probabilities from Q-values (no extra forward pass)
        st.session_state.current_action_probs = st.session_state.agent.trainer.probabilities_from_q_values(
            st.session_state.current_q_values
        )
        # Move input to model's device for activation extraction (only when network shown)
        if st.session_state.get('show_network', False):
            device = st.session_state.agent.trainer.device
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            model = st.session_state.agent.model
            was_training = model.training
            model.eval()
            try:
                with torch.inference_mode():
                    acts = model.get_activations(obs_tensor)
                st.session_state.activations = acts
            finally:
                model.train(was_training)
        else:
            st.session_state.activations = None


def step_game(action=None):
    """Execute one game step."""
    if st.session_state.game_over or st.session_state.env is None:
        return
    
    # === GET ACTION FIRST (for automatic play) ===
    if action is None and st.session_state.agent is not None:
        action, selection_reason = st.session_state.agent.get_action_with_reason(
            st.session_state.current_observation, 
            training=st.session_state.training
        )
    elif action is not None:
        # Manual action provided
        selection_reason = "manual"
    else:
        action = 0  # Default: straight
        selection_reason = "exploiting"
    
    # === CAPTURE PRE-ACTION DECISION SNAPSHOT ===
    # This is what the agent saw and decided BEFORE stepping
    if st.session_state.agent is not None and st.session_state.current_observation is not None:
        pre_q_values = st.session_state.agent.get_q_values(st.session_state.current_observation)
        # Compute probabilities from Q-values (no extra forward pass)
        pre_action_probs = st.session_state.agent.trainer.probabilities_from_q_values(pre_q_values)
        
        # Store decision snapshot for visualization
        st.session_state.decision_snapshot = {
            'observation': st.session_state.current_observation.copy(),
            'q_values': pre_q_values.copy(),
            'action_probs': pre_action_probs.copy(),
            'action': action,
            'epsilon': st.session_state.agent.epsilon,
            'reason': selection_reason,
        }
    
    # Step environment
    next_obs, reward, terminated, truncated, info = st.session_state.env.step(action)
    done = terminated or truncated
    
    # Store transition if training
    if st.session_state.training and st.session_state.agent is not None:
        # Use terminated for bootstrap mask, truncated is stored separately
        st.session_state.agent.remember(
            st.session_state.current_observation, action, reward, next_obs, terminated, truncated
        )
        # Train on short memory - use terminated for bootstrap
        loss = st.session_state.agent.train_short_memory(
            st.session_state.current_observation, action, reward, next_obs, terminated
        )
        # Train on replay memory
        replay_loss = st.session_state.agent.train_long_memory()
        
        # Update training stats
        if replay_loss:
            st.session_state.training_stats['loss'].append(replay_loss)
            st.session_state.loss_history.append(replay_loss)
        
        # Sync agent's step counter
        st.session_state.agent.total_steps += 1
    
    # Update state
    st.session_state.current_observation = next_obs
    st.session_state.game_state = st.session_state.env.get_state_for_display()
    st.session_state.game_over = done
    st.session_state.current_action = action
    
    # Update Q-values and activations for NEXT state visualization
    if st.session_state.agent is not None:
        st.session_state.current_q_values = st.session_state.agent.get_q_values(next_obs)
        # Compute probabilities from Q-values (no extra forward pass)
        st.session_state.current_action_probs = st.session_state.agent.trainer.probabilities_from_q_values(
            st.session_state.current_q_values
        )
        # Move input to model's device for activation extraction (only when network shown)
        if st.session_state.get('show_network', False):
            device = st.session_state.agent.trainer.device
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
            # Use eval mode and inference_mode for visualization
            model = st.session_state.agent.model
            was_training = model.training
            model.eval()
            try:
                with torch.inference_mode():
                    acts = model.get_activations(next_obs_tensor)
                st.session_state.activations = acts
            finally:
                model.train(was_training)
        else:
            st.session_state.activations = None
    
    # If episode ended, update stats
    if done:
        score = info['score']
        
        # Only update training stats and decay epsilon if actually training
        if st.session_state.training:
            # Update agent's stats (authoritative source)
            if st.session_state.agent:
                st.session_state.agent.update_stats(score)
                st.session_state.agent.decay_epsilon()  # Decay epsilon after episode
                # Dashboard mirrors agent's stats
                st.session_state.training_stats['scores'] = st.session_state.agent.scores_history.copy()
                st.session_state.training_stats['mean_scores'] = st.session_state.agent.mean_scores_history.copy()
                st.session_state.training_stats['epsilon'] = st.session_state.agent.epsilon_history.copy()
                st.session_state.training_stats['high_score'] = st.session_state.agent.high_score
                st.session_state.training_episode = st.session_state.agent.n_games
                
                # Sync history deques
                st.session_state.score_history.clear()
                st.session_state.score_history.extend(st.session_state.agent.scores_history)
                st.session_state.mean_score_history.clear()
                st.session_state.mean_score_history.extend(st.session_state.agent.mean_scores_history)
                st.session_state.epsilon_history.clear()
                st.session_state.epsilon_history.extend(st.session_state.agent.epsilon_history)
            
            # Note: Auto-reset handled by live_game_area fragment
            # No sleep here to avoid blocking UI


# ============================================================================
# Visualization Functions (Compact Versions)
# ============================================================================
def render_game_grid():
    """Render the game as a compact Plotly figure."""
    if st.session_state.game_state is None:
        fig = go.Figure()
        fig.update_layout(
            width=380, height=380,
            margin=dict(l=5, r=5, t=5, b=5),
            plot_bgcolor='#1e1e2e', paper_bgcolor='#1e1e2e',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig
    
    state = st.session_state.game_state
    grid_size = state['grid_size']
    snake = state['snake']
    food = state['food']
    direction = state['direction']
    game_over = state['game_over']
    
    fig = go.Figure()
    
    # Grid background
    if st.session_state.show_grid:
        for i in range(grid_size + 1):
            fig.add_shape(
                type="line", x0=0, y0=i, x1=grid_size, y1=i,
                line=dict(color="rgba(100,100,100,0.2)", width=0.5)
            )
            fig.add_shape(
                type="line", x0=i, y0=0, x1=i, y1=grid_size,
                line=dict(color="rgba(100,100,100,0.2)", width=0.5)
            )
    
    # Snake body
    for i, (x, y) in enumerate(snake):
        color = '#32CD32' if i == 0 else '#228B22'
        opacity = 1.0 if i == 0 else max(0.4, 1.0 - i * 0.03)
        
        fig.add_shape(
            type="rect", x0=x, y0=y, x1=x+1, y1=y+1,
            fillcolor=color, opacity=opacity,
            line=dict(color='#1a5c1a', width=1), layer="above"
        )
        
        # Eyes on head
        if i == 0:
            eye_size = 0.15
            if direction == 0:  # Right
                fig.add_shape(type="circle", x0=x+0.65, y0=y+0.2, x1=x+0.85, y1=y+0.4,
                            fillcolor="white", line=dict(width=0))
                fig.add_shape(type="circle", x0=x+0.65, y0=y+0.6, x1=x+0.85, y1=y+0.8,
                            fillcolor="white", line=dict(width=0))
            elif direction == 2:  # Left
                fig.add_shape(type="circle", x0=x+0.15, y0=y+0.2, x1=x+0.35, y1=y+0.4,
                            fillcolor="white", line=dict(width=0))
                fig.add_shape(type="circle", x0=x+0.15, y0=y+0.6, x1=x+0.35, y1=y+0.8,
                            fillcolor="white", line=dict(width=0))
            elif direction == 3:  # Up
                fig.add_shape(type="circle", x0=x+0.2, y0=y+0.15, x1=x+0.4, y1=y+0.35,
                            fillcolor="white", line=dict(width=0))
                fig.add_shape(type="circle", x0=x+0.6, y0=y+0.15, x1=x+0.8, y1=y+0.35,
                            fillcolor="white", line=dict(width=0))
            else:  # Down
                fig.add_shape(type="circle", x0=x+0.2, y0=y+0.65, x1=x+0.4, y1=y+0.85,
                            fillcolor="white", line=dict(width=0))
                fig.add_shape(type="circle", x0=x+0.6, y0=y+0.65, x1=x+0.8, y1=y+0.85,
                            fillcolor="white", line=dict(width=0))
    
    # Food (guard for victory state where food=None)
    if food is not None:
        fx, fy = food
        fig.add_shape(
            type="circle", x0=fx+0.1, y0=fy+0.1, x1=fx+0.9, y1=fy+0.9,
            fillcolor='#FF4444', line=dict(color='#CC0000', width=2), layer="above"
        )
        fig.add_shape(
            type="circle", x0=fx+0.2, y0=fy+0.2, x1=fx+0.4, y1=fy+0.4,
            fillcolor='#FFAAAA', line=dict(width=0), layer="above"
        )
    
    # Game over overlay
    if game_over:
        fig.add_annotation(
            x=grid_size/2, y=grid_size/2,
            text="GAME OVER", showarrow=False,
            font=dict(size=28, color='red', family='Arial Black'),
            bgcolor='rgba(0,0,0,0.8)', bordercolor='red', borderwidth=2, borderpad=8
        )
    
    fig.update_layout(
        xaxis=dict(range=[0, grid_size], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, grid_size], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        width=380, height=380,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor='#1e1e2e', paper_bgcolor='#1e1e2e',
        font=dict(color='white'), showlegend=False
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def render_state_vector():
    """Render the 11-dimensional state vector as compact horizontal bars."""
    if st.session_state.current_observation is None:
        return None
    
    obs = st.session_state.current_observation
    labels = [
        "Danger→", "Danger↘", "Danger↙",
        "Dir←", "Dir→", "Dir↑", "Dir↓",
        "Food←", "Food→", "Food↑", "Food↓"
    ]
    
    colors = []
    for i, val in enumerate(obs):
        if i < 3:  # Danger
            colors.append('#FF4444' if val > 0.5 else '#44FF44')
        elif i < 7:  # Direction
            colors.append('#4488FF' if val > 0.5 else '#333333')
        else:  # Food
            colors.append('#FFD700' if val > 0.5 else '#444444')
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=obs, y=labels, orientation='h',
        marker_color=colors,
        text=[f'{v:.0f}' for v in obs],
        textposition='inside', textfont=dict(color='white', size=10, family='Arial Black'),
        showlegend=False, hovertemplate='%{y}: %{x:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        height=220, margin=dict(l=50, r=5, t=5, b=5),
        xaxis=dict(range=[0, 1.1], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=9)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9)
    )
    return fig


def render_q_values():
    """Render Q-values and action probabilities as compact bar charts."""
    if st.session_state.current_q_values is None:
        return None, None
    
    q_vals = st.session_state.current_q_values
    action_names = ['Straight', 'Right', 'Left']
    action_idx = st.session_state.current_action
    
    # Q-values
    q_colors = ['#44FF44' if i == action_idx else '#666666' for i in range(3)]
    fig_q = go.Figure()
    fig_q.add_trace(go.Bar(
        x=action_names, y=q_vals, marker_color=q_colors,
        text=[f'{v:.2f}' for v in q_vals], textposition='outside',
        textfont=dict(color='white', size=11), showlegend=False
    ))
    fig_q.update_layout(
        title=dict(text="Q-Values", font=dict(size=12)),
        height=180, margin=dict(l=20, r=10, t=30, b=10),
        yaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9)
    )
    
    # Probabilities
    if st.session_state.current_action_probs is not None:
        probs = st.session_state.current_action_probs
        p_colors = ['#FFD700' if i == action_idx else '#555555' for i in range(3)]
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            x=action_names, y=probs, marker_color=p_colors,
            text=[f'{p:.0%}' for p in probs], textposition='outside',
            textfont=dict(color='white', size=11), showlegend=False
        ))
        fig_p.update_layout(
            title=dict(text="Probabilities", font=dict(size=12)),
            height=180, margin=dict(l=20, r=10, t=30, b=10),
            yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=9)
        )
    else:
        fig_p = None
    
    return fig_q, fig_p


def render_network_architecture():
    """Render compact neural network architecture."""
    if st.session_state.activations is None or st.session_state.agent is None:
        return None
    
    acts = st.session_state.activations
    model = st.session_state.agent.model
    
    # Derive layer sizes from actual model
    layer_sizes = [
        model.fc1.in_features,
        model.fc1.out_features,
        model.fc2.out_features,
        model.fc3.out_features,
    ]
    layer_names = [f'Input ({layer_sizes[0]})', 
                   f'Hidden 1 ({layer_sizes[1]})', 
                   f'Hidden 2 ({layer_sizes[2]})', 
                   f'Output ({layer_sizes[3]})']
    
    fig = go.Figure()
    layer_x = [0, 1, 2, 3]
    
    for layer_idx, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        x = layer_x[layer_idx]
        
        if layer_idx == 0:
            act_vals = st.session_state.current_observation
        else:
            key = f'fc{layer_idx}'
            if key in acts:
                act_vals = np.abs(acts[key][0])
                if act_vals.max() > 0:
                    act_vals = act_vals / act_vals.max()
            else:
                act_vals = np.zeros(size)
        
        display_nodes = min(size, 16)
        if size <= 16:
            indices = np.arange(size)
            display_acts = act_vals
        else:
            indices = np.concatenate([np.arange(min(6, size)), np.arange(max(6, size-6), size)])
            display_acts = act_vals[indices]
        
        y_positions = np.linspace(-1, 1, len(indices))
        
        for j, (idx, y, act) in enumerate(zip(indices, y_positions, display_acts)):
            intensity = min(1.0, act * 2)
            if layer_idx == 0:
                color = f'rgba(255, {int(255*(1-intensity))}, {int(255*(1-intensity))}, 1)'
            elif layer_idx == 3:
                color = '#FFD700' if idx == st.session_state.current_action else f'rgba(100,100,255,{0.3+0.7*intensity})'
            else:
                color = f'rgba(100, {int(100+155*intensity)}, 255, {0.3+0.7*intensity})'
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers',
                marker=dict(size=10, color=color, line=dict(width=0.5, color='white')),
                showlegend=False, hovertemplate=f'{name}<br>Neuron {idx}<br>Act: {act:.2f}<extra></extra>'
            ))
        
        fig.add_annotation(x=x, y=1.15, text=name, showarrow=False,
                          font=dict(color='white', size=9), xanchor='center')
    
    # Layer connections
    for i in range(3):
        fig.add_shape(type="line", x0=layer_x[i]+0.12, y0=-1.05, x1=layer_x[i+1]-0.12, y1=-1.05,
                     line=dict(color='rgba(100,100,100,0.2)', width=1.5), layer="below")
    
    fig.update_layout(
        title=dict(text="Network Activations", font=dict(size=12)),
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=[-0.4, 3.4], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1.2, 1.25], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9)
    )
    return fig


def render_training_charts():
    """Render compact training analytics charts - independent charts."""
    stats = st.session_state.training_stats
    agent = st.session_state.agent
    
    # ===== SCORE CHART =====
    if stats['scores']:
        # Limit displayed history to last 500 episodes
        display_scores = stats['scores'][-500:]
        display_means = stats['mean_scores'][-500:]
        episodes = list(range(len(stats['scores']) - len(display_scores) + 1, len(stats['scores']) + 1))
        
        fig_score = go.Figure()
        fig_score.add_trace(go.Scatter(
            x=episodes, y=display_scores, mode='lines', name='Score',
            line=dict(color='#4488FF', width=1), opacity=0.7
        ))
        if display_means:
            fig_score.add_trace(go.Scatter(
                x=episodes, y=display_means, mode='lines', name='Avg(50)',
                line=dict(color='#FFD700', width=2), opacity=0.9
            ))
    else:
        fig_score = go.Figure()
        fig_score.add_annotation(x=0.5, y=0.5, text="No episodes completed",
                              showarrow=False, font=dict(color='gray', size=12),
                              xref="paper", yref="paper")
    
    fig_score.update_layout(
        title=dict(text="Scores per Episode", font=dict(size=11)),
        height=180, margin=dict(l=30, r=10, t=25, b=25),
        xaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        uirevision="snake-scores",  # Preserve Plotly state (zoom/pan)
    )
    
    # ===== LOSS CHART - uses trainer's authoritative loss_history =====
    fig_loss = go.Figure()
    if agent and agent.trainer.loss_history:
        values = np.asarray(agent.trainer.loss_history, dtype=float)
        # Limit displayed history to last 1000 optimizer steps
        window = 1000
        values = values[-window:]
        if values.size:
            last_step = agent.trainer.training_step
            steps = np.arange(last_step - len(values) + 1, last_step + 1)
            finite = np.isfinite(values)
            
            # Log axes cannot display zero. Plot positive finite values.
            plotted = np.where(finite & (values > 0), values, np.nan)
            
            if np.any(np.isfinite(plotted)):
                fig_loss.add_trace(go.Scatter(
                    x=steps, y=plotted, mode='lines', name='Optimizer loss',
                    line=dict(color='#FF6666', width=1), opacity=0.7
                ))
            else:
                fig_loss.add_annotation(
                    text="No positive finite losses for log scale",
                    x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                    font=dict(color='gray', size=11)
                )
    else:
        fig_loss.add_annotation(
            text="No optimizer updates yet",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            font=dict(color='gray', size=12)
        )
    
    fig_loss.update_layout(
        title=dict(text="Loss per Optimizer Update (log)", font=dict(size=11)),
        height=180, margin=dict(l=40, r=10, t=25, b=30),
        xaxis=dict(title="Optimizer Step", showgrid=True, gridcolor='rgba(100,100,100,0.15)'),
        yaxis=dict(type='log', title="Loss", showgrid=True, gridcolor='rgba(100,100,100,0.15)'),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        uirevision="snake-loss",  # Preserve Plotly state (zoom/pan)
    )
    
    # ===== EPSILON CHART =====
    if stats['epsilon']:
        display_eps = stats['epsilon'][-500:]
        episodes = list(range(len(stats['epsilon']) - len(display_eps) + 1, len(stats['epsilon']) + 1))
        fig_eps = go.Figure()
        fig_eps.add_trace(go.Scatter(
            x=episodes, y=display_eps, mode='lines', name='ε',
            line=dict(color='#44FF88', width=1.5), fill='tozeroy', fillcolor='rgba(68,255,136,0.1)'
        ))
    else:
        fig_eps = go.Figure()
        fig_eps.add_annotation(x=0.5, y=0.5, text="No epsilon history",
                              showarrow=False, font=dict(color='gray', size=12),
                              xref="paper", yref="paper")
    
    fig_eps.update_layout(
        title=dict(text="Exploration Rate (ε)", font=dict(size=11)),
        height=180, margin=dict(l=30, r=10, t=25, b=25),
        xaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9),
        uirevision="snake-epsilon",  # Preserve Plotly state (zoom/pan)
    )
    
    return fig_score, fig_loss, fig_eps


# ============================================================================
# Main Application
# ============================================================================
def main():
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🐍 Snake DQN")
        st.caption("Live Training Dashboard")
        st.divider()
        
        # Environment Settings
        with st.expander("🌍 Environment", expanded=True):
            grid_size = st.slider("Grid Size", 10, 30, st.session_state.env_config['grid_size'])
            step_penalty = st.number_input("Step Penalty", -1.0, 0.0, st.session_state.env_config['step_penalty'], 0.001, format="%.3f")
            food_reward = st.number_input("Food Reward", 0.0, 100.0, st.session_state.env_config['food_reward'], 1.0)
            death_penalty = st.number_input("Death Penalty", -100.0, 0.0, st.session_state.env_config['death_penalty'], 1.0)
            
            if st.button("Apply Environment Settings", width='stretch'):
                st.session_state.env_config.update({
                    'grid_size': grid_size, 'step_penalty': step_penalty,
                    'food_reward': food_reward, 'death_penalty': death_penalty
                })
                st.session_state.env = create_env()
                st.session_state.agent = create_agent()
                reset_game()
                st.success("Environment updated - agent and replay reset")
                st.rerun()
        
        # Agent Settings
        with st.expander("🧠 Agent Hyperparameters", expanded=True):
            hidden_size = st.selectbox("Hidden Size", [128, 256, 512], index=[128, 256, 512].index(st.session_state.agent_config['hidden_size']))
            lr = st.number_input("Learning Rate", 1e-5, 1e-1, st.session_state.agent_config['lr'], 1e-4, format="%.5f")
            gamma = st.number_input("Gamma", 0.8, 0.999, st.session_state.agent_config['gamma'], 0.001, format="%.3f")
            epsilon_start = st.number_input("Epsilon Start", 0.0, 1.0, st.session_state.agent_config['epsilon_start'], 0.01)
            epsilon_end = st.number_input("Epsilon End", 0.0, 0.5, st.session_state.agent_config['epsilon_end'], 0.001, format="%.3f")
            epsilon_decay = st.number_input("Epsilon Decay", 0.9, 0.9999, st.session_state.agent_config['epsilon_decay'], 0.0001, format="%.4f")
            memory_size = st.selectbox("Memory Size", [10000, 50000, 100000, 200000], index=[10000, 50000, 100000, 200000].index(st.session_state.agent_config['memory_size']))
            batch_size = st.selectbox("Batch Size", [256, 512, 1000, 2048], index=[256, 512, 1000, 2048].index(st.session_state.agent_config['batch_size']))
            
            if st.button("Create New Agent", width='stretch'):
                st.session_state.agent_config.update({
                    'hidden_size': hidden_size, 'lr': lr, 'gamma': gamma,
                    'epsilon_start': epsilon_start, 'epsilon_end': epsilon_end,
                    'epsilon_decay': epsilon_decay, 'memory_size': memory_size,
                    'batch_size': batch_size
                })
                st.session_state.agent = create_agent()
                st.success("New agent created!")
                st.rerun()
        
        # Training Controls
        with st.expander("🏋️ Training", expanded=True):
            # Activity selector - determines who acts and whether learning happens
            activity_options = ["Train", "Evaluate", "Manual"]
            activity_labels = {
                "Train": "🧠 Train — Agent learns while playing",
                "Evaluate": "📊 Evaluate — Greedy policy, no learning",
                "Manual": "👤 Manual — You control the snake",
            }
            st.session_state.activity = st.radio(
                "Activity",
                activity_options,
                index=activity_options.index(st.session_state.get('activity', 'Train')),
                format_func=lambda x: activity_labels[x],
                horizontal=True,
            )
            
            # Speed presets for Train/Evaluate modes
            if st.session_state.activity != "Manual":
                speed_options = {
                    "Watch": 0.1,      # 100ms - slow enough to follow
                    "Normal": 0.05,    # 50ms - normal speed
                    "Fast": 0.01,      # 10ms - fast training
                    "Max": 0.0,        # 0ms - maximum speed
                }
                st.session_state.speed_preset = st.radio(
                    "Playback speed",
                    list(speed_options.keys()),
                    index=list(speed_options.keys()).index(st.session_state.get('speed_preset', 'Watch')),
                    horizontal=True,
                )
                st.session_state.speed = speed_options[st.session_state.speed_preset]
            
            # Main control button
            is_running = st.session_state.get('activity') == "Train" and st.session_state.training
            is_paused = st.session_state.paused
            
            if st.session_state.activity == "Train":
                if not st.session_state.training:
                    if st.button("▶️ Start training", width='stretch', type="primary"):
                        if st.session_state.agent is None: st.session_state.agent = create_agent()
                        if st.session_state.env is None: st.session_state.env = create_env()
                        st.session_state.training = True
                        st.session_state.paused = False
                        reset_game()
                        st.rerun()
                elif is_paused:
                    if st.button("▶️ Resume training", width='stretch', type="primary"):
                        st.session_state.paused = False
                        st.rerun()
                else:
                    if st.button("⏸️ Pause training", width='stretch'):
                        st.session_state.paused = True
                        st.rerun()
            
            elif st.session_state.activity == "Evaluate":
                if not st.session_state.training:
                    if st.button("▶️ Start evaluation", width='stretch'):
                        if st.session_state.agent is None: st.session_state.agent = create_agent()
                        if st.session_state.env is None: st.session_state.env = create_env()
                        st.session_state.training = False
                        st.session_state.paused = False
                        reset_game()
                        st.rerun()
                elif is_paused:
                    if st.button("▶️ Resume evaluation", width='stretch'):
                        st.session_state.paused = False
                        st.rerun()
                else:
                    if st.button("⏸️ Pause evaluation", width='stretch'):
                        st.session_state.paused = True
                        st.rerun()
            
            else:  # Manual
                if not st.session_state.paused:
                    if st.button("⏸️ Pause", width='stretch'):
                        st.session_state.paused = True
                        st.rerun()
            
            # Step once button (only when paused)
            if st.session_state.paused:
                if st.button("⏭️ Step once", width='stretch'):
                    step_game()
                    st.rerun()
            
            # Reset buttons - clearly labeled
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Restart board", width='stretch', help="Restart current game, keep learned weights and training history"):
                    reset_game()
                    st.rerun()
            with c2:
                if st.button("🗑️ Clear training history", width='stretch', help="Clear displayed charts and statistics, keep model weights"):
                    if st.session_state.agent: st.session_state.agent.reset_stats()
                    st.session_state.training_stats = {'scores': [], 'mean_scores': [], 'epsilon': [], 'loss': [], 'high_score': 0}
                    st.session_state.training_episode = 0
                    st.session_state.score_history.clear()
                    st.session_state.mean_score_history.clear()
                    st.session_state.loss_history.clear()
                    st.session_state.epsilon_history.clear()
                    st.rerun()
            
            # New agent / environment reset
            st.divider()
            if st.button("🆕 Start new training run", width='stretch', help="Create fresh agent and replay buffer, discard current model"):
                st.session_state.agent = create_agent()
                reset_game()
                st.success("New training run started")
                st.rerun()
            
            st.divider()
            save_path = st.text_input("Checkpoint Path", "checkpoints/snake_dqn.pth")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save checkpoint", width='stretch'):
                    if st.session_state.agent:
                        st.session_state.agent.save_checkpoint(save_path)
                        st.success("Saved!")
            with c2:
                if st.button("📂 Load checkpoint", width='stretch'):
                    if st.session_state.agent is None: st.session_state.agent = create_agent()
                    try:
                        info = st.session_state.agent.load_checkpoint(save_path)
                        st.session_state.training_episode = info.get('episode', 0)
                        st.session_state.agent.epsilon = info.get('epsilon', 1.0)
                        st.session_state.training_stats['high_score'] = info.get('high_score', 0)
                        st.success(f"Loaded ep {info.get('episode', 0)}, ε={info.get('epsilon', 1.0):.3f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Load failed: {e}")
        
        # Display Settings
        with st.expander("🎨 Display", expanded=False):
            st.session_state.show_network = st.checkbox("Show network internals", st.session_state.get('show_network', False))
            if st.button("🎮 New Game", width='stretch', help="Restart board (same as Restart board above)"):
                reset_game()
                st.rerun()
    
    # Initialize env and agent if not exist
    if st.session_state.env is None:
        st.session_state.env = create_env()
    if st.session_state.agent is None:
        st.session_state.agent = create_agent()
    if st.session_state.game_state is None:
        reset_game()
    
    # ============================================================================
    # Live Game Fragment (partial updates via st.fragment)
    # ============================================================================
    # Calculate refresh interval based on speed and activity
    _running = (
        not st.session_state.paused
        and st.session_state.activity != "Manual"
    )
    _refresh_seconds = (
        st.session_state.speed if _running else None
    )

    @st.fragment(run_every=_refresh_seconds)
    def live_game_area():
        """Render game board and auto-step. Runs independently of main page."""
        state = st.session_state
        
        # Auto-step logic (inside fragment)
        if not state.paused and state.activity != "Manual":
            if not state.game_over:
                step_game()
            elif state.training:
                reset_game()
            else:
                state.paused = True
        
        # Render game using RGB image (inline PNG for reliable rendering)
        if state.env is not None:
            frame = state.env.render()
            if frame is not None:
                display_game_frame(frame)
        
        # Status caption (updates on fragment refresh)
        if state.agent is not None and state.game_state is not None:
            st.caption(
                f"Ep {state.agent.n_games} · "
                f"Score {state.game_state['score']} · "
                f"Best {state.agent.high_score} · "
                f"ε {state.agent.epsilon:.3f}"
            )

    # Chart fragment - refreshes every 1 second when running (reduced from 0.5s)
    _chart_refresh = 1.0 if _running else None

    # Persistent chart placeholders (created once, updated by fragment)
    score_slot = st.empty()
    loss_col, epsilon_col = st.columns(2, gap="small")
    loss_slot = loss_col.empty()
    epsilon_slot = epsilon_col.empty()

    # Track chart data versions for change detection
    if "chart_versions" not in st.session_state:
        st.session_state.chart_versions = {"scores": 0, "loss": 0, "epsilon": 0}

    @st.fragment(run_every=1.0 if _running else None)
    def live_charts_area():
        """Render learning charts. Updates independently of game fragment."""
        stats = st.session_state.training_stats
        agent = st.session_state.agent

        # Compute version tuples for change detection
        versions = {
            "scores": (len(stats["scores"]), len(stats["mean_scores"])),
            "loss": agent.trainer.training_step if agent else 0,
            "epsilon": (len(stats["epsilon"]), agent.epsilon if agent else None),
        }

        # Check which charts need updating
        changed = {
            name for name, version in versions.items()
            if st.session_state.chart_versions.get(name) != version
        }

        if not changed:
            return

        # Build only the changed charts
        figures = dict(zip(
            ("scores", "loss", "epsilon"),
            render_training_charts(),
        ))

        slots = {
            "scores": score_slot,
            "loss": st.session_state.get("_loss_slot", loss_col.empty()),
            "epsilon": st.session_state.get("_epsilon_slot", epsilon_col.empty()),
        }

        # Store slots in session state for persistence
        st.session_state._loss_slot = loss_col.empty()
        st.session_state._epsilon_slot = epsilon_col.empty()

        for name in ("scores", "loss", "epsilon"):
            if name not in changed:
                continue

            fig = figures[name]
            fig.update_layout(
                height=220 if name == "scores" else 180,
                autosize=True,
                width=None,
                uirevision=f"snake-{name}",  # Preserve Plotly state (zoom/pan)
                transition_duration=0,
            )

            if name == "scores":
                slots[name].plotly_chart(
                    fig, width='stretch', key=f"live_{name}",
                    config={"displayModeBar": False}
                )
            elif name == "loss":
                st.session_state._loss_slot.plotly_chart(
                    fig, width='stretch', key=f"live_{name}",
                    config={"displayModeBar": False}
                )
            else:  # epsilon
                st.session_state._epsilon_slot.plotly_chart(
                    fig, width='stretch', key=f"live_{name}",
                    config={"displayModeBar": False}
                )

        # Update tracked versions
        st.session_state.chart_versions.update(versions)

    # ============================================================================
    # Main Layout: Game (40%) + Charts (60%)
    # ============================================================================
    game_col, charts_col = st.columns([2, 3], gap="small")
    
    # ----- LEFT COLUMN: LIVE GAME ARENA -----
    with game_col:
        # Live game fragment (auto-updates)
        live_game_area()
        
        # State indicator
        activity = st.session_state.activity
        is_training = st.session_state.training and activity == "Train"
        is_paused = st.session_state.paused
        
        if activity == "Train":
            if not is_training:
                status_text = "🟡 Train · Ready"
            elif is_paused:
                status_text = "🟡 Train · Paused"
            else:
                status_text = "🟢 Train · Running"
        elif activity == "Evaluate":
            if is_paused:
                status_text = "🔵 Evaluate · Paused"
            else:
                status_text = "🔵 Evaluate · Running"
        else:  # Manual
            if is_paused:
                status_text = "⚪ Manual · Paused"
            else:
                status_text = "⚪ Manual · Running"
        
        st.caption(status_text)
        
        # Controls row (outside fragment - user interactions trigger full rerun)
        c1, c2, c3 = st.columns(3)
        with c1:
            if activity == "Train":
                if not st.session_state.training:
                    if st.button("▶️ Start training", width='stretch', type="primary", key="play_pause"):
                        if st.session_state.agent is None: st.session_state.agent = create_agent()
                        if st.session_state.env is None: st.session_state.env = create_env()
                        st.session_state.training = True
                        st.session_state.paused = False
                        reset_game()
                        st.rerun()
                elif is_paused:
                    if st.button("▶️ Resume", width='stretch', type="primary", key="play_pause"):
                        st.session_state.paused = False
                        st.rerun()
                else:
                    if st.button("⏸️ Pause", width='stretch', key="play_pause"):
                        st.session_state.paused = True
                        st.rerun()
            elif activity == "Evaluate":
                if not st.session_state.training:
                    if st.button("▶️ Start evaluation", width='stretch', type="primary", key="play_pause"):
                        if st.session_state.agent is None: st.session_state.agent = create_agent()
                        if st.session_state.env is None: st.session_state.env = create_env()
                        st.session_state.training = False
                        st.session_state.paused = False
                        reset_game()
                        st.rerun()
                elif is_paused:
                    if st.button("▶️ Resume", width='stretch', type="primary", key="play_pause"):
                        st.session_state.paused = False
                        st.rerun()
                else:
                    if st.button("⏸️ Pause", width='stretch', key="play_pause"):
                        st.session_state.paused = True
                        st.rerun()
            else:  # Manual
                if not is_paused:
                    if st.button("⏸️ Pause", width='stretch', key="play_pause"):
                        st.session_state.paused = True
                        st.rerun()
        
        with c2:
            if st.session_state.paused:
                if st.button("⏭️ Step once", width='stretch', key="step_btn"):
                    step_game()
                    st.rerun()
            else:
                st.button("⏭️ Step once", width='stretch', disabled=True, key="step_btn")
        
        with c3:
            if st.button("🔄 Restart board", width='stretch', key="reset_btn", help="Restart current game, keep learned weights"):
                reset_game()
                st.rerun()
        
        # Manual controls when paused (Manual activity only)
        if st.session_state.activity == "Manual" and st.session_state.paused and not st.session_state.game_over:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                if st.button("⬆️ Straight", width='stretch', key="manual_straight"):
                    step_game(0); st.rerun()
            with mc2:
                if st.button("➡️ Turn right", width='stretch', key="manual_right"):
                    step_game(1); st.rerun()
            with mc3:
                if st.button("⬅️ Turn left", width='stretch', key="manual_left"):
                    step_game(2); st.rerun()
    
    # ----- RIGHT COLUMN: LEARNING CHARTS -----
    with charts_col:
        # Live charts fragment (auto-updates)
        live_charts_area()
    
    # Network Details Fragment (independent refresh when enabled)
    _net_refresh = 1.0 if (st.session_state.get('show_network', False) and _running) else None

    @st.fragment(run_every=_net_refresh)
    def live_network_area():
        """Render network details. Updates independently when enabled."""
        if not st.session_state.get('show_network', False):
            return
            
        with st.expander("▸ Neural-network details", expanded=True):
            # State Vector
            st.write("**State Vector (11D)**")
            fig_state = render_state_vector()
            if fig_state:
                st.plotly_chart(fig_state, width='stretch', key="state_chart", config={'displayModeBar': False})
            
            st.divider()
            
            # Two columns for network + decision
            nc1, nc2 = st.columns([1, 1])
            
            with nc1:
                st.write("**Network Activations**")
                fig_net = render_network_architecture()
                if fig_net:
                    st.plotly_chart(fig_net, width='stretch', key="net_chart", config={'displayModeBar': False})
            
            with nc2:
                st.write("**Decision Making**")
                
                snap = st.session_state.decision_snapshot
                is_training = st.session_state.training
                if snap is not None:
                    q_vals = snap['q_values']
                    action_idx = snap['action']
                    action_names = ['Straight', 'Right', 'Left']
                    epsilon = snap['epsilon'] if is_training else 0.0
                    reason = snap['reason']
                    
                    greedy_action = int(np.argmax(q_vals))
                    actual_probs = np.full(3, epsilon / 3.0)
                    actual_probs[greedy_action] += 1.0 - epsilon
                else:
                    q_vals = st.session_state.current_q_values
                    action_idx = st.session_state.current_action
                    action_names = ['Straight', 'Right', 'Left']
                    epsilon = st.session_state.agent.epsilon if (st.session_state.agent and is_training) else 0.0
                    reason = "unknown"
                    
                    if q_vals is not None:
                        greedy_action = int(np.argmax(q_vals))
                        actual_probs = np.full(3, epsilon / 3.0)
                        actual_probs[greedy_action] += 1.0 - epsilon
                    else:
                        actual_probs = np.full(3, 1.0 / 3.0)
                        greedy_action = 0
                
                if q_vals is not None:
                    q_colors = ['#44FF44' if i == action_idx else '#666666' for i in range(3)]
                    fig_q = go.Figure()
                    fig_q.add_trace(go.Bar(
                        x=action_names, y=q_vals, marker_color=q_colors,
                        text=[f'{v:.2f}' for v in q_vals], textposition='outside',
                        textfont=dict(color='white', size=11), showlegend=False
                    ))
                    fig_q.update_layout(
                        title=dict(text="Q-Values (at decision time)", font=dict(size=12)),
                        height=180, margin=dict(l=20, r=10, t=30, b=10),
                        yaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=9)
                    )
                    st.plotly_chart(fig_q, width='stretch', key="q_chart", config={'displayModeBar': False})
                    
                    p_colors = ['#FFD700' if i == action_idx else '#555555' for i in range(3)]
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Bar(
                        x=action_names, y=actual_probs, marker_color=p_colors,
                        text=[f'{p:.0%}' for p in actual_probs], textposition='outside',
                        textfont=dict(color='white', size=11), showlegend=False
                    ))
                    fig_p.update_layout(
                        title=dict(text=f"Policy (ε-greedy, ε={epsilon:.3f})", font=dict(size=12)),
                        height=180, margin=dict(l=20, r=10, t=30, b=10),
                        yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white', size=9)
                    )
                    st.plotly_chart(fig_p, width='stretch', key="prob_chart", config={'displayModeBar': False})
                    
                    action_label = action_names[action_idx] if action_idx is not None else "N/A"
                    if reason == "exploring":
                        st.info(f"🎲 **Exploring** (ε={epsilon:.3f}): {action_label} chosen randomly")
                    elif reason == "manual":
                        st.info(f"👤 **Manual**: {action_label} chosen by user")
                    else:
                        st.success(f"🎯 **Exploiting** (ε={epsilon:.3f}): {action_label} = argmax Q")
                else:
                    st.info("No decision data yet")

    # ============================================================================
    # Network Details (conditional on toggle - fragment renders itself)
    # ============================================================================
    if st.session_state.get('show_network', False):
        live_network_area()

if __name__ == "__main__":
    main()