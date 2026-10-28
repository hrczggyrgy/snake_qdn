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
from pathlib import Path
from collections import deque
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Import our modules
from snake_env import SnakeGameAI
from model import Linear_QNet, QTrainer
from agent import Agent, ReplayMemory, train_agent, evaluate_agent


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
        'step_mode': False,
        'show_grid': True,
        
        # Visualization Data
        'score_history': deque(maxlen=500),
        'mean_score_history': deque(maxlen=500),
        'loss_history': deque(maxlen=500),
        'epsilon_history': deque(maxlen=500),
        
        # Network activations for visualization
        'activations': None,
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
    
    # Get initial Q-values if agent exists
    if st.session_state.agent is not None:
        st.session_state.current_q_values = st.session_state.agent.get_q_values(obs)
        st.session_state.current_action_probs = st.session_state.agent.get_action_probs(obs)
        acts = st.session_state.agent.model.get_activations(torch.FloatTensor(obs).unsqueeze(0))
        st.session_state.activations = acts


def step_game(action=None):
    """Execute one game step."""
    if st.session_state.game_over or st.session_state.env is None:
        return
    
    # Get action from agent if not provided
    if action is None and st.session_state.agent is not None:
        action = st.session_state.agent.get_action(
            st.session_state.current_observation, 
            training=st.session_state.training
        )
    elif action is None:
        action = 0  # Default: straight
    
    # Step environment
    next_obs, reward, terminated, truncated, info = st.session_state.env.step(action)
    done = terminated or truncated
    
    # Store transition if training
    if st.session_state.training and st.session_state.agent is not None:
        st.session_state.agent.remember(
            st.session_state.current_observation, action, reward, next_obs, done
        )
        # Train on short memory
        loss = st.session_state.agent.train_short_memory(
            st.session_state.current_observation, action, reward, next_obs, done
        )
        # Train on replay memory
        replay_loss = st.session_state.agent.train_long_memory()
        
        # Update training stats
        if replay_loss:
            st.session_state.training_stats['loss'].append(replay_loss)
            st.session_state.loss_history.append(replay_loss)
    
    # Update state
    st.session_state.current_observation = next_obs
    st.session_state.game_state = st.session_state.env.get_state_for_display()
    st.session_state.game_over = done
    st.session_state.current_action = action
    
    # Update Q-values and activations for visualization
    if st.session_state.agent is not None:
        st.session_state.current_q_values = st.session_state.agent.get_q_values(next_obs)
        st.session_state.current_action_probs = st.session_state.agent.get_action_probs(next_obs)
        acts = st.session_state.agent.model.get_activations(torch.FloatTensor(next_obs).unsqueeze(0))
        st.session_state.activations = acts
    
    # If episode ended, update stats
    if done:
        score = info['score']
        st.session_state.training_stats['scores'].append(score)
        st.session_state.score_history.append(score)
        
        # Calculate rolling average
        window = min(50, len(st.session_state.training_stats['scores']))
        avg_score = np.mean(st.session_state.training_stats['scores'][-window:])
        st.session_state.training_stats['mean_scores'].append(avg_score)
        st.session_state.mean_score_history.append(avg_score)
        
        if score > st.session_state.training_stats['high_score']:
            st.session_state.training_stats['high_score'] = score
        
        if st.session_state.agent:
            st.session_state.agent.decay_epsilon()
            st.session_state.training_stats['epsilon'].append(st.session_state.agent.epsilon)
            st.session_state.epsilon_history.append(st.session_state.agent.epsilon)
        
        st.session_state.training_episode += 1
        
        # Auto-reset for continuous training
        if st.session_state.training:
            time.sleep(0.1)  # Brief pause
            reset_game()


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
    
    # Food
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
    if st.session_state.activations is None:
        return None
    
    acts = st.session_state.activations
    layer_sizes = [11, 256, 256, 3]
    layer_names = ['Input (11)', 'Hidden 1 (256)', 'Hidden 2 (256)', 'Output (3)']
    
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
    """Render compact training analytics charts."""
    stats = st.session_state.training_stats
    
    if not stats['scores']:
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, text="No training data yet",
                          showarrow=False, font=dict(color='gray', size=12),
                          xref="paper", yref="paper")
        fig.update_layout(height=180, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        return fig, fig, fig
    
    episodes = list(range(1, len(stats['scores']) + 1))
    
    # Score chart
    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(
        x=episodes, y=stats['scores'], mode='lines', name='Score',
        line=dict(color='#4488FF', width=1), opacity=0.7
    ))
    if stats['mean_scores']:
        fig_score.add_trace(go.Scatter(
            x=episodes, y=stats['mean_scores'], mode='lines', name='Avg(50)',
            line=dict(color='#FFD700', width=2), opacity=0.9
        ))
    fig_score.update_layout(
        title=dict(text="Scores", font=dict(size=11)),
        height=180, margin=dict(l=30, r=10, t=25, b=25),
        xaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
    )
    
    # Loss chart
    fig_loss = go.Figure()
    if stats['loss']:
        loss_x = list(range(1, len(stats['loss']) + 1))
        fig_loss.add_trace(go.Scatter(
            x=loss_x, y=stats['loss'], mode='lines', name='Loss',
            line=dict(color='#FF6666', width=1), opacity=0.6
        ))
        if len(stats['loss']) > 10:
            window = min(30, len(stats['loss']) // 10)
            smoothed = pd.Series(stats['loss']).rolling(window, min_periods=1).mean()
            fig_loss.add_trace(go.Scatter(
                x=loss_x, y=smoothed, mode='lines', name=f'Smooth({window})',
                line=dict(color='#FF0000', width=1.5), opacity=0.9
            ))
    fig_loss.update_layout(
        title=dict(text="Loss (log)", font=dict(size=11)),
        height=180, margin=dict(l=30, r=10, t=25, b=25),
        xaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        yaxis=dict(type='log', showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
    )
    
    # Epsilon chart
    fig_eps = go.Figure()
    if stats['epsilon']:
        fig_eps.add_trace(go.Scatter(
            x=episodes, y=stats['epsilon'], mode='lines', name='ε',
            line=dict(color='#44FF88', width=1.5), fill='tozeroy', fillcolor='rgba(68,255,136,0.1)'
        ))
    fig_eps.update_layout(
        title=dict(text="Epsilon", font=dict(size=11)),
        height=180, margin=dict(l=30, r=10, t=25, b=25),
        xaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        yaxis=dict(range=[0, 1.05], showgrid=True, gridcolor='rgba(100,100,100,0.15)', title=""),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=9)
    )
    
    return fig_score, fig_loss, fig_eps


# ============================================================================
# Main Application
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🐍 Snake DQN - Live Training Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
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
                reset_game()
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
            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶️ Start Training", width='stretch', disabled=st.session_state.training):
                    if st.session_state.agent is None: st.session_state.agent = create_agent()
                    if st.session_state.env is None: st.session_state.env = create_env()
                    st.session_state.training = True
                    st.session_state.paused = False
                    reset_game()
                    st.rerun()
            with c2:
                if st.button("⏹️ Stop Training", width='stretch', disabled=not st.session_state.training):
                    st.session_state.training = False
                    st.session_state.paused = True
                    st.rerun()
            
            if st.button("🔄 Reset Stats", width='stretch'):
                if st.session_state.agent: st.session_state.agent.reset_stats()
                st.session_state.training_stats = {'scores': [], 'mean_scores': [], 'epsilon': [], 'loss': [], 'high_score': 0}
                st.session_state.training_episode = 0
                st.session_state.score_history.clear()
                st.session_state.mean_score_history.clear()
                st.session_state.loss_history.clear()
                st.session_state.epsilon_history.clear()
                st.rerun()
            
            st.session_state.speed = st.slider("Speed (ms/frame)", 0, 200, st.session_state.speed, 10)
            st.session_state.step_mode = st.checkbox("Step Mode", st.session_state.step_mode)
            
            st.divider()
            save_path = st.text_input("Checkpoint Path", "checkpoints/snake_dqn.pth")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save", width='stretch'):
                    if st.session_state.agent:
                        st.session_state.agent.save_checkpoint(save_path)
                        st.success("Saved!")
            with c2:
                if st.button("📂 Load", width='stretch'):
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
            st.session_state.show_grid = st.checkbox("Show Grid", st.session_state.show_grid)
            if st.button("🎮 New Game", width='stretch'):
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
    # Top Metrics Row (Compact)
    # ============================================================================
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Episode", st.session_state.training_episode, 
                 delta=f"Step: {st.session_state.training_step}" if st.session_state.training else None)
    with m2:
        current_score = st.session_state.game_state['score'] if st.session_state.game_state else 0
        st.metric("Score", current_score, delta=f"High: {st.session_state.training_stats['high_score']}")
    with m3:
        eps = st.session_state.agent.epsilon if st.session_state.agent else 0
        st.metric("Epsilon", f"{eps:.3f}")
    with m4:
        avg50 = st.session_state.training_stats['mean_scores'][-1] if st.session_state.training_stats['mean_scores'] else 0
        st.metric("Avg (50)", f"{avg50:.1f}")
    
    # ============================================================================
    # TABBED LAYOUT - Better organization
    # ============================================================================
    tab1, tab2, tab3 = st.tabs(["🎮 Game Arena", "🧠 Network Internals", "📈 Analytics"])
    
    # ----- TAB 1: GAME ARENA -----
    with tab1:
        # Game controls row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("▶️ Play" if st.session_state.paused else "⏸️ Pause", width='stretch', key="play_pause"):
                st.session_state.paused = not st.session_state.paused
                st.rerun()
        with c2:
            if st.button("⏭️ Step", width='stretch', disabled=not st.session_state.step_mode, key="step_btn"):
                step_game()
                st.rerun()
        with c3:
            if st.button("🔄 Reset", width='stretch', key="reset_btn"):
                reset_game()
                st.rerun()
        with c4:
            st.write("")  # Spacer
        
        # Manual controls when paused
        if st.session_state.paused and not st.session_state.game_over:
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                if st.button("⬆️ Straight", width='stretch', key="manual_straight"):
                    step_game(0); st.rerun()
            with mc2:
                if st.button("➡️ Right", width='stretch', key="manual_right"):
                    step_game(1); st.rerun()
            with mc3:
                if st.button("⬅️ Left", width='stretch', key="manual_left"):
                    step_game(2); st.rerun()
        
        # Game render - fixed width for stability
        fig_game = render_game_grid()
        st.plotly_chart(fig_game, width=380, key="game_chart", config={'displayModeBar': False})
        
        # Game info
        if st.session_state.game_state:
            info = st.session_state.game_state
            head = info['snake'][0] if info['snake'] else (0, 0)
            dirs = ['→', '↓', '←', '↑']
            st.caption(f"Length: {len(info['snake'])} | Head: {head} | Food: {info['food']} | Dir: {dirs[info['direction']]}")
        
        # Auto-step logic
        if not st.session_state.paused and not st.session_state.game_over:
            step_game()
            time.sleep(st.session_state.speed / 1000.0)
            st.rerun()
        elif not st.session_state.paused and st.session_state.game_over:
            if st.session_state.training:
                reset_game()
                time.sleep(0.1)
                st.rerun()
            else:
                st.session_state.paused = True
    
    # ----- TAB 2: NETWORK INTERNALS -----
    with tab2:
        # State Vector
        st.write("**State Vector (11D)**")
        fig_state = render_state_vector()
        if fig_state:
            st.plotly_chart(fig_state, width=380, key="state_chart", config={'displayModeBar': False})
        
        st.divider()
        
        # Two columns for network + decision
        nc1, nc2 = st.columns([1, 1])
        
        with nc1:
            st.write("**Network Activations**")
            fig_net = render_network_architecture()
            if fig_net:
                st.plotly_chart(fig_net, width=380, key="net_chart", config={'displayModeBar': False})
        
        with nc2:
            st.write("**Decision Making**")
            fig_q, fig_p = render_q_values()
            if fig_q:
                st.plotly_chart(fig_q, width=380, key="q_chart", config={'displayModeBar': False})
            if fig_p:
                st.plotly_chart(fig_p, width=380, key="prob_chart", config={'displayModeBar': False})
            
            # Current action display
            if st.session_state.current_action is not None:
                action_names = ['⬆️ Straight', '➡️ Right', '⬅️ Left']
                action_emoji = action_names[st.session_state.current_action]
                if st.session_state.training and st.session_state.agent and random.random() < st.session_state.agent.epsilon:
                    st.info(f"🎲 Exploring: {action_emoji}")
                else:
                    st.success(f"🎯 Exploiting: {action_emoji}")
    
    # ----- TAB 3: ANALYTICS -----
    with tab3:
        fig_score, fig_loss, fig_eps = render_training_charts()
        
        st.plotly_chart(fig_score, width=380, key="score_chart", config={'displayModeBar': False})
        st.plotly_chart(fig_loss, width=380, key="loss_chart", config={'displayModeBar': False})
        st.plotly_chart(fig_eps, width=380, key="eps_chart", config={'displayModeBar': False})
        
        # Statistics
        st.divider()
        st.write("**Training Statistics**")
        stats = st.session_state.training_stats
        
        if stats['scores']:
            recent = stats['scores'][-10:] if len(stats['scores']) >= 10 else stats['scores']
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Recent Avg (10)", f"{np.mean(recent):.1f}")
                st.metric("Best Score", stats['high_score'])
            with s2:
                st.metric("Episodes", len(stats['scores']))
                if stats['loss']:
                    st.metric("Loss", f"{stats['loss'][-1]:.4f}")
            
            # Compact score distribution
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=stats['scores'], nbinsx=min(15, max(stats['scores'])+1),
                marker_color='#4488FF', opacity=0.7
            ))
            fig_dist.update_layout(
                title=dict(text="Score Distribution", font=dict(size=11)),
                height=160, margin=dict(l=20, r=10, t=25, b=20),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=9), showlegend=False,
                xaxis=dict(title=""), yaxis=dict(title="")
            )
            st.plotly_chart(fig_dist, width=380, key="dist_chart", config={'displayModeBar': False})
        else:
            st.info("Start training to see statistics")

if __name__ == "__main__":
    main()