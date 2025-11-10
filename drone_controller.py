import numpy as np
import pickle
import random
from collections import deque
from environment import DroneEnv
# Import the new target-seeking function
from q_learning import find_best_move_to_intrusion 

# Global storage for the loaded Q-table
Q_TABLE = None
LAST_STATE = None
LAST_CHOSEN_ACTION = None

def discretize_state(state_vec, env):
    x, y, battery = state_vec[0], state_vec[1], state_vec[2]
    battery_bin = int(battery // 10)
    auto_return_flag = 1 if (env.battery <= env.battery_critical and tuple(env.drone_pos) != (0, 0)) else 0
    return (int(x), int(y), battery_bin, auto_return_flag)

def load_q_table(path='q_table.pkl'):
    global Q_TABLE
    if Q_TABLE is None:
        try:
            with open(path, 'rb') as f:
                Q_TABLE = pickle.load(f)
        except FileNotFoundError:
            Q_TABLE = {}
    return Q_TABLE

def choose_action(env):
    global Q_TABLE, LAST_STATE, LAST_CHOSEN_ACTION
    if Q_TABLE is None:
        Q_TABLE = load_q_table()
        
    state_vec = env._get_obs()
    current_state = discretize_state(state_vec, env)
    
    # 1. Prioritize Auto-Return
    if env.auto_returning and env.return_path:
        action = 5
        LAST_CHOSEN_ACTION = action
        return action
    
    q_values = Q_TABLE.get(current_state, np.zeros(env.action_space.n))
    temp_q_values = np.copy(q_values)
    valid_exploratory_actions = [0, 1, 2, 3] 

    # 2. Obstacle/Stuck Loop Detection and Override
    last_message = env.info.get("message")
    was_blocked = last_message and "blocked by obstacle" in last_message
    
    if was_blocked or env.consecutive_stuck_steps >= 2:
        # Penalize the problematic action
        if LAST_CHOSEN_ACTION is not None:
             temp_q_values[LAST_CHOSEN_ACTION] = -99999.0 
        temp_q_values[4] = -99999.0 
        temp_q_values[5] = -99999.0 
        
        action = np.argmax(temp_q_values)
        
        move_q_values = [temp_q_values[a] for a in valid_exploratory_actions]
        if np.all(np.isclose(move_q_values, move_q_values[0])):
            action = random.choice(valid_exploratory_actions)
    
    # 3. Normal Exploitation
    else:
        # Base exit fix
        if env.drone_pos == [0, 0] and not env.is_done():
            temp_q_values[4] = -99999.0 
            temp_q_values[5] = -99999.0
            
            action = np.argmax(temp_q_values)
            
            if np.all(np.isclose(temp_q_values[valid_exploratory_actions], temp_q_values[valid_exploratory_actions][0])):
                action = random.choice(valid_exploratory_actions) 

        # Target-Seeking when Q-table is empty
        elif np.all(np.isclose(q_values, 0.0)):
            current_pos = tuple(env.drone_pos)
            target_action = find_best_move_to_intrusion(env, current_pos)
            
            if target_action is not None:
                action = target_action
            else:
                action = random.choice(valid_exploratory_actions)
        else:
            action = np.argmax(q_values)

    # 4. Update History
    LAST_STATE = current_state
    LAST_CHOSEN_ACTION = action
    return action