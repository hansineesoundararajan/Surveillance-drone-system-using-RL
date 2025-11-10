import matplotlib.pyplot as plt
import time
from environment import DroneEnv
from drone_controller import choose_action 

# NEW GLOBAL CONSTANTS (Must match q_learning.py and environment.py)
GRID_SIZE = 5 
NUM_OBSTACLES = 4 
MAX_STEPS_EVAL = 500 

def main():
    # Load the environment settings using the correct, updated parameters
    env = DroneEnv(grid_size=GRID_SIZE, num_obstacles=NUM_OBSTACLES)
    fig = plt.figure(figsize=(8, 6))
    plt.ion() # interactive mode
    
    total_reward = 0
    step_count = 0
    max_steps = MAX_STEPS_EVAL
    movement_delay = 0.2 # seconds per step for visualization speed

    # Warm-up render
    env.render(total_reward=total_reward)
    plt.pause(1.0)
    print("\n--- Starting Live Evaluation of Trained RL Policy ---")

    while step_count < max_steps:
        # Pass 'env' to the controller for state discretization and dynamic stuck-fix logic
        action = choose_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1

        # Render the environment with updated reward
        env.render(total_reward=total_reward)
        plt.pause(0.001)

        # Print step info
        if info["message"]:
            print(f"Step: {step_count} | Action: {action} | Reward: {reward:.2f} | Msg: {info['message']}")
        else:
            print(f"Step: {step_count} | Action: {action} | Reward: {reward:.2f}")

        # Slow down movement for visualization
        time.sleep(movement_delay)

        if terminated:
            print(f"\n🎉 Exploration complete. Drone returned to base in {step_count} steps.")
            break
        
        if truncated:
            print("\n⚠️ Episode truncated (Max steps reached).")
            break

    plt.ioff()
    plt.show()
    env.close()

if __name__ == "__main__":
    main()