import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from environment import DroneEnv
from utils import find_shortest_path_bfs 

Q_TABLE_PATH = 'q_table.pkl'
GRID_SIZE = 5 
NUM_OBSTACLES = 4 
MAX_STEPS_TRAINING = 1000 

def discretize_state(state_vec, env):
    """ 
    Converts the continuous state vector into the discrete tuple key 
    (x, y, battery_bin, auto_return_flag). 
    """
    x, y, battery = state_vec[0], state_vec[1], state_vec[2]
    battery_bin = int(battery // 10) 
    auto_return_flag = 1 if (env.battery <= env.battery_critical and tuple(env.drone_pos) != (0, 0)) else 0
    return (int(x), int(y), battery_bin, auto_return_flag)

def find_best_move_to_intrusion(env, current_pos):
    """
    Finds the first move (action 0-3) that leads to the single intrusion cell 
    using BFS, prioritizing the path toward the target.
    """
    if not env.intrusion_locations:
        return None 
    
    intrusion_idx = next(iter(env.intrusion_locations))
    target_pos = divmod(intrusion_idx, env.grid_size)

    path = find_shortest_path_bfs(current_pos, target_pos, env.grid_size, env.obstacle_locations)
    
    if path:
        next_pos = path[0]
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        if dx == -1 and dy == 0:
            return 0 # Up
        elif dx == 1 and dy == 0:
            return 1 # Down
        elif dx == 0 and dy == -1:
            return 2 # Left
        elif dx == 0 and dy == 1:
            return 3 # Right
            
    return None

def train_q_learning(
    episodes=5000,
    alpha=0.02,             # FIX: Lowered learning rate for stability
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9998,   # Adjusted decay
    epsilon_min=0.01,       # Lower epsilon floor
    max_steps=MAX_STEPS_TRAINING,
    plot=True,
    callback=None
):
    env = DroneEnv(grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)
    q_table = {}
    rewards_per_episode = []
    avg_td_error_per_episode = []
    print("Starting Q-Learning Training (Target Seeking Mode)...")
    
    for ep in range(episodes):
        state_vec, _ = env.reset()
        state = discretize_state(state_vec, env)
        total_reward = 0
        episode_td_errors = []

        for step in range(max_steps):
            if state not in q_table:
                q_table[state] = np.zeros(env.action_space.n)

            # ACTION SELECTION LOGIC
            if np.random.rand() < epsilon:
                # EXPLORATION: Prioritizes moving towards the intrusion target
                current_pos = tuple(env.drone_pos)
                exploratory_action = find_best_move_to_intrusion(env, current_pos)
                
                if exploratory_action is not None:
                    action = exploratory_action
                else:
                    action = random.choice([0, 1, 2, 3]) 
                
            else:
                # EXPLOITATION: Standard Q-value selection
                q_values = q_table.get(state, np.zeros(env.action_space.n))
                
                temp_q_values = np.copy(q_values)
                temp_q_values[4] -= 100 
                if state[3] == 0: 
                    temp_q_values[5] -= 100 
                
                if np.all(np.isclose(temp_q_values, temp_q_values[0])):
                    action = random.choice([0, 1, 2, 3]) 
                else:
                    action = np.argmax(temp_q_values)
            
            # CRITICAL: Override to prevent being stuck at (0,0) when not done
            if tuple(env.drone_pos) == (0, 0) and not env.is_done():
                if action in [4, 5]: 
                    action = random.choice([0, 1, 3]) 

            next_state_vec, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_state(next_state_vec, env)

            if next_state not in q_table:
                q_table[next_state] = np.zeros(env.action_space.n)

            best_next_q = np.max(q_table[next_state])
            current_q = q_table[state][action]
            td_target = reward + gamma * best_next_q
            td_error = td_target - current_q
            
            q_table[state][action] = current_q + alpha * td_error
            
            episode_td_errors.append(np.abs(td_error))
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        if episode_td_errors:
            avg_td_error_per_episode.append(np.mean(episode_td_errors))
        else:
            avg_td_error_per_episode.append(0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        rewards_per_episode.append(total_reward)

        if callback:
            callback(ep+1, total_reward, avg_td_error_per_episode[-1])

        if (ep + 1) % 100 == 0 or ep == 1:
            print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.2f} | Avg TD Error: {avg_td_error_per_episode[-1]:.3f} | Epsilon: {epsilon:.3f} | Visited States: {len(q_table)}")

    env.close()

    with open(Q_TABLE_PATH, 'wb') as f:
        pickle.dump(q_table, f)
    print(f"\nTraining complete. Q-table saved to {Q_TABLE_PATH}")

    # Plotting TD Error (Agent Loss Graph)
    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode)
        plt.title("Q-Learning Training Progress (Total Reward)")
        plt.subplot(1, 2, 2)
        plt.plot(avg_td_error_per_episode)
        plt.title("Agent 'Loss' Graph (TD Error Convergence)")
        plt.tight_layout()
        plt.show()

    return q_table

def evaluate_policy(q_table, episodes=100, max_steps=1000, plot=True):
    env = DroneEnv(grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)
    rewards = []
    intrusion_rates = []

    print("\n--- Starting Policy Evaluation (Target Seeking Mode) ---")

    for ep in range(episodes):
        state_vec, _ = env.reset()
        state = discretize_state(state_vec, env)
        total_reward = 0
        intrusions_detected = 0

        for step in range(max_steps):
            from drone_controller import choose_action 
            action = choose_action(env) 

            next_state_vec, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_state_vec, env)

            if info.get("intrusion_detected"):
                intrusions_detected += 1
                
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)
        
        intrusion_rate = intrusions_detected / env.max_intrusions
        intrusion_rates.append(intrusion_rate)
        
        print(f"Eval Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | Intrusion Rate: {intrusion_rate:.2f}")

    print("\n--- FINAL EVALUATION RESULTS ---")
    print(f"Average Total Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Intrusion Detection Rate: {np.mean(intrusion_rates):.2f}")
    
    if plot:
        plt.figure(figsize=(12, 4))
        plt.hist(rewards, bins=15)
        plt.title("Distribution of Total Rewards")
        plt.xlabel("Total Reward")
        plt.ylabel("Frequency")
        plt.show()

if __name__ == "__main__":
    q_table = train_q_learning()
    if q_table:
        evaluate_policy(q_table, episodes=100, max_steps=MAX_STEPS_TRAINING)