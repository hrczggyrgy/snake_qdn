"""
Observation schema definition for Snake DQN.

Centralizes feature ordering, version, and size to avoid hard-coded
numbers scattered throughout the codebase.
"""

# Incremented when the observation structure changes in a way that
# breaks checkpoint compatibility (e.g., feature count/order changes).
OBSERVATION_SCHEMA_VERSION = 2

# Ordered feature names. Indices are stable across the codebase.
OBSERVATION_FEATURES = (
    # Original 11 inputs: preserve their existing order.
    "danger_straight",
    "danger_right",
    "danger_left",
    "dir_left",
    "dir_right",
    "dir_up",
    "dir_down",
    "food_left",
    "food_right",
    "food_up",
    "food_down",
    # Additional inputs (indices 11-21).
    "food_dx",
    "food_dy",
    "wall_left",
    "wall_right",
    "wall_up",
    "wall_down",
    "snake_length",
    "timeout_progress",
    "space_straight",
    "space_right",
    "space_left",
)

# Derived size - single source of truth.
OBSERVATION_SIZE = len(OBSERVATION_FEATURES)

# For backward compatibility with old checkpoints
OBSERVATION_SCHEMA_VERSION_LEGACY = 1
OBSERVATION_SIZE_LEGACY = 11
