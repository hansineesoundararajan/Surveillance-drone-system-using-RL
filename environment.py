import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from visualization import draw_grid
from utils import find_shortest_path_bfs

class DroneEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=5, num_obstacles=4, max_battery=100, battery_critical=30):
        super().__init__()
        self.grid_size = grid_size
        self.max_battery = max_battery
        self.battery_critical = battery_critical
        self.num_obstacles = num_obstacles
        self.action_space = spaces.Discrete(6)

        obs_low = np.zeros(grid_size * grid_size + 3, dtype=np.float32)
        obs_high = np.ones(grid_size * grid_size + 3, dtype=np.float32)
        obs_high[0:2] = grid_size - 1
        obs_high[2] = max_battery
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        
        self.reset()
        self.new_cell = False
        self.last_state_pos = tuple(self.drone_pos)
        self.consecutive_stuck_steps = 0
        self.info = {"message": None}
        self.step_count = 0

    def state_index(self, x, y):
        return x * self.grid_size + y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = [0, 0]
        self.visited = set()
        self.visited.add(tuple(self.drone_pos))
        self.battery = self.max_battery
        self.auto_returning = False
        self.return_path = deque()
        self.new_cell = False
        self.last_state_pos = tuple(self.drone_pos)
        self.consecutive_stuck_steps = 0
        self.info = {"message": None}
        self.step_count = 0
        
        self.charger_locations = {
            self.state_index(0, 0),
            self.state_index(self.grid_size - 1, self.grid_size - 1)
        }
        self.intrusion_locations = set()
        self.total_intrusions_spawned = 0

        # Single Target Mode
        self.min_intrusions = 1
        self.max_intrusions = 1

        self._place_obstacles()
        # Add static intrusion at (min(3, grid_size-1), min(3, grid_size-1))
        static_x = min(3, self.grid_size - 1)
        static_y = min(3, self.grid_size - 1)
        static_intrusion_idx = self.state_index(static_x, static_y)
        if static_intrusion_idx not in self.obstacle_locations and static_intrusion_idx not in self.charger_locations and static_intrusion_idx != 0:
            self.intrusion_locations.add(static_intrusion_idx)
            self.total_intrusions_spawned = 1
        else:
            # Fallback if position invalid
            self.maybe_spawn_intrusion(probability=1.0, force=True)
        
        return self._get_obs(), {}

    def _place_obstacles(self):
        self.obstacle_locations = set()
        # Add static obstacle at (min(2, grid_size-1), min(2, grid_size-1))
        static_x = min(2, self.grid_size - 1)
        static_y = min(2, self.grid_size - 1)
        static_obstacle_idx = self.state_index(static_x, static_y)
        if static_obstacle_idx not in self.charger_locations and static_obstacle_idx != 0:
            self.obstacle_locations.add(static_obstacle_idx)

        # Add remaining random obstacles if needed
        max_allowed_obstacles = int(0.15 * self.grid_size * self.grid_size)
        num_obstacles = min(self.num_obstacles - 1, max_allowed_obstacles - 1)  # Adjust for static

        all_cells = set(range(self.grid_size * self.grid_size))
        candidate_cells = list(all_cells - self.charger_locations - {0} - self.obstacle_locations)
        random.shuffle(candidate_cells)

        for idx in candidate_cells:
            if len(self.obstacle_locations) >= num_obstacles + 1:  # +1 for static
                break
            self.obstacle_locations.add(idx)
            if not self.all_cells_reachable():
                self.obstacle_locations.remove(idx)

    def all_cells_reachable(self):
        from collections import deque
        start = (0, 0)
        visited = set()
        queue = deque([start])
        obstacles = {divmod(idx, self.grid_size) for idx in self.obstacle_locations}
        
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited or (x, y) in obstacles:
                continue
            visited.add((x, y))
            
            for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    queue.append((nx, ny))
                    
        total_free = self.grid_size * self.grid_size - len(obstacles)
        return len(visited) == total_free

    def step(self, action):
        self.step_count += 1
        msg = self._move(action)
        reward, intrusion_msg = self._compute_reward(action, msg) 
        terminated = self.is_done()
        truncated = False
        
        final_msg = ""
        if msg:
            final_msg += msg
        if intrusion_msg:
            final_msg += intrusion_msg
             
        intrusion_detected = "Intrusion detected!" in final_msg
        
        self.info = {"message": final_msg if final_msg else None, "intrusion_detected": intrusion_detected}
        
        return self._get_obs(), reward, terminated, truncated, self.info

    def _move(self, action):
        x, y = self.drone_pos
        current_pos = tuple(self.drone_pos)
        new_x, new_y = x, y
        self.new_cell = False

        if current_pos == (0, 0) and self.battery == self.max_battery:
            self.auto_returning = False
            self.return_path.clear()

        if self.battery <= self.battery_critical and not self.auto_returning and current_pos != (0, 0):
            target = (0, 0)
            self.return_path = deque(find_shortest_path_bfs(current_pos, target, self.grid_size, self.obstacle_locations))
            self.auto_returning = True

        if self.auto_returning and self.return_path:
            next_pos = self.return_path.popleft()
            new_x, new_y = next_pos
            action = 5 
        
        elif not self.auto_returning:
            if action == 0 and x > 0: new_x -= 1
            elif action == 1 and x < self.grid_size - 1: new_x += 1
            elif action == 2 and y > 0: new_y -= 1
            elif action == 3 and y < self.grid_size - 1: new_y += 1

        if self.state_index(new_x, new_y) in self.obstacle_locations:
            return " Movement blocked by obstacle."

        if (new_x, new_y) != current_pos:
            self.new_cell = (new_x, new_y) not in self.visited
            self.drone_pos = [new_x, new_y]
            self.visited.add(tuple(self.drone_pos))

            if action != 4:
                 self.battery -= 1

        if self.state_index(*self.drone_pos) in self.charger_locations and self.battery < self.max_battery:
            self.battery = self.max_battery
            if self.drone_pos == [0, 0]:
                return " Recharged at base."
            return " Recharged at charging station!"

        if self.battery <= 0:
            return " Battery depleted! Drone cannot move."
            
        return None

    def _compute_reward(self, action, msg):
        x, y = self.drone_pos
        idx = self.state_index(x, y)
        current_pos_tuple = tuple(self.drone_pos)
        reward = 0
        intrusion_msg = None 

        # Adjusted Reward Constants for clean Target Seeking
        revisit_penalty = 0.0     # Neutral
        new_cell_reward = 0.0     # Neutral
        intrusion_detection_reward = 500.0 # Target Hit Reward
        obstacle_penalty = -5.0
        battery_deplete_penalty = -100.0
        hover_penalty = -0.5      # Increased penalty
        stuck_penalty = -20.0     # Increased severity
        recharge_base_reward = 5.0 

        # --- Mutually Exclusive Penalties (Use elif) ---
        if msg and "Battery depleted" in msg:
            reward = battery_deplete_penalty
        elif msg and "blocked" in msg:
            reward = obstacle_penalty
            self.consecutive_stuck_steps = 0 # Reset stuck counter on obstacle hit
        elif action == 4: # Hover
            reward = hover_penalty
        elif self.new_cell:
            reward = new_cell_reward
        else:
            reward = revisit_penalty
        # -----------------------------------------------

        # Intrusion detection (Applies on top)
        if idx in self.intrusion_locations:
            reward += intrusion_detection_reward
            self.intrusion_locations.remove(idx)
            intrusion_msg = " Intrusion detected!"
        
        # Reward for reaching base with target hit (optional goal reinforcement)
        if self.total_intrusions_spawned == 1 and len(self.intrusion_locations) == 0 and self.drone_pos == [0, 0]:
             reward += 100.0 


        # Penalize redundant auto-return action (Applies on top)
        if action == 5 and not self.auto_returning:
            reward -= 1.0

        # Apply stuck penalty logic
        is_move_action = action in [0, 1, 2, 3] 
        
        # Only check for stuck if it was a move action, and it wasn't blocked
        if is_move_action and current_pos_tuple == self.last_state_pos and not (msg and "blocked" in msg):
            self.consecutive_stuck_steps += 1
            if self.consecutive_stuck_steps >= 2:
                reward += stuck_penalty * self.consecutive_stuck_steps
        else:
            self.consecutive_stuck_steps = 0
            
        self.last_state_pos = current_pos_tuple
        
        return reward, intrusion_msg

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in self.visited:
            grid[x, y] = 4
        for idx in self.intrusion_locations:
            x, y = divmod(idx, self.grid_size)
            grid[x, y] = 2
        for idx in self.obstacle_locations:
            x, y = divmod(idx, self.grid_size)
            grid[x, y] = 1
        for idx in self.charger_locations:
            x, y = divmod(idx, self.grid_size)
            grid[x, y] = 3
            
        return np.concatenate(([self.drone_pos[0], self.drone_pos[1], self.battery], grid.flatten()))

    def is_done(self):
        # Episode terminates as soon as the single target is detected
        intrusion_detected = self.total_intrusions_spawned > 0 and len(self.intrusion_locations) == 0
        return intrusion_detected

    def maybe_spawn_intrusion(self, probability=0.02, force=False):
        if self.total_intrusions_spawned >= self.max_intrusions:
            return

        if force or random.random() < probability:
            all_cells = set(range(self.grid_size * self.grid_size))
            invalid = (
                self.intrusion_locations | 
                self.obstacle_locations | 
                self.charger_locations | 
                {self.state_index(*self.drone_pos)}
            )
            valid_cells = list(all_cells - invalid)
            
            if valid_cells:
                new_intrusion = random.choice(valid_cells)
                self.intrusion_locations.add(new_intrusion)
                self.total_intrusions_spawned += 1

    def render(self, mode='human', total_reward=0):
        draw_grid(self, total_reward=total_reward)

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()