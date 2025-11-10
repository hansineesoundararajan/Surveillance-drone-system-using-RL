import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg for embedding in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from environment import DroneEnv
from drone_controller import choose_action
from q_learning import train_q_learning
import threading
import time

class DroneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Self-Driving Drone Simulator")
        self.env = DroneEnv()
        self.total_reward = 0
        self.step_count = 0
        self.running = False
        self.canvas_size = 400
        self.cell_size = self.canvas_size // self.env.grid_size
        self.episodes_data = []
        self.rewards_data = []

        self.create_widgets()
        self.draw_grid()

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="🚁 Surveillance Drone ", font=('Arial', 16, 'bold'), fg='blue')
        title_label.pack(pady=10)

        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        # Left panel: Grid and legend
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10)

        # Canvas for grid
        self.canvas = tk.Canvas(left_frame, width=self.canvas_size, height=self.canvas_size, bg='lightblue', relief='sunken', bd=2)
        self.canvas.pack()

        # Legend
        legend_frame = tk.Frame(left_frame)
        legend_frame.pack(pady=5)
        tk.Label(legend_frame, text="Legend:", font=('Arial', 10, 'bold')).pack()
        tk.Label(legend_frame, text="🟦 Drone", fg='blue').pack(anchor='w')
        tk.Label(legend_frame, text="🔴 Intrusion", fg='red').pack(anchor='w')
        tk.Label(legend_frame, text="⬛ Obstacle", fg='black').pack(anchor='w')
        tk.Label(legend_frame, text="🟩 Charger", fg='green').pack(anchor='w')
        tk.Label(legend_frame, text="⬜ Visited", fg='gray').pack(anchor='w')

        # Right panel: Controls
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10)

        # Controls frame
        control_frame = tk.LabelFrame(right_frame, text="Controls", padx=10, pady=10)
        control_frame.pack(pady=10)

        self.train_button = ttk.Button(control_frame, text="🚀 Train", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5, pady=5)

        self.evaluate_button = ttk.Button(control_frame, text="▶️ Evaluate", command=self.start_evaluation)
        self.evaluate_button.grid(row=0, column=1, padx=5, pady=5)

        self.reset_button = ttk.Button(control_frame, text="🔄 Reset", command=self.reset_simulation)
        self.reset_button.grid(row=0, column=2, padx=5, pady=5)

        # Sliders
        self.episode_var = tk.IntVar(value=1000)
        ttk.Label(control_frame, text="Episodes:").grid(row=1, column=0, sticky='w')
        self.episode_slider = ttk.Scale(control_frame, from_=100, to=5000, variable=self.episode_var, orient=tk.HORIZONTAL)
        self.episode_slider.grid(row=1, column=1, columnspan=2, sticky='ew')

        self.grid_var = tk.IntVar(value=5)
        ttk.Label(control_frame, text="Grid Size:").grid(row=2, column=0, sticky='w')
        self.grid_slider = ttk.Scale(control_frame, from_=3, to=10, variable=self.grid_var, orient=tk.HORIZONTAL)
        self.grid_slider.grid(row=2, column=1, columnspan=2, sticky='ew')

        # Log
        log_frame = tk.LabelFrame(right_frame, text="Log", padx=10, pady=10)
        log_frame.pack(pady=10)
        self.log_text = tk.Text(log_frame, height=10, width=50, wrap=tk.WORD)
        self.log_text.pack()

        # Graph
        graph_frame = tk.LabelFrame(right_frame, text="Training Graph", padx=10, pady=10)
        graph_frame.pack(pady=10)
        self.fig = Figure(figsize=(5,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Rewards per Episode")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_graph.get_tk_widget().pack()

    def draw_grid(self):
        self.canvas.delete("all")
        grid_size = self.env.grid_size
        cell_size = self.canvas_size // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                color = 'white'
                if (i, j) in self.env.visited:
                    color = 'lightgray'
                for idx in self.env.intrusion_locations:
                    x, y = divmod(idx, grid_size)
                    if x == i and y == j:
                        color = 'red'
                for idx in self.env.obstacle_locations:
                    x, y = divmod(idx, grid_size)
                    if x == i and y == j:
                        color = 'black'
                for idx in self.env.charger_locations:
                    x, y = divmod(idx, grid_size)
                    if x == i and y == j:
                        color = 'green'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')
                if self.env.drone_pos == [i, j]:
                    # Draw drone as a circle with propellers
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    radius = cell_size / 4
                    self.canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, fill='blue', outline='black')
                    # Propellers
                    prop_len = radius * 1.5
                    self.canvas.create_line(center_x - prop_len, center_y, center_x + prop_len, center_y, fill='black', width=2)
                    self.canvas.create_line(center_x, center_y - prop_len, center_x, center_y + prop_len, fill='black', width=2)
        # Info
        self.canvas.create_text(10, 10, anchor='nw', text=f"Drone: {self.env.drone_pos} Reward: {self.total_reward:.1f} Battery: {self.env.battery:.1f}", font=('Arial', 10))

    def start_training(self):
        self.episodes = int(self.episode_var.get())
        self.log_text.insert(tk.END, f"Starting training with {self.episodes} episodes...\n")
        self.train_button.config(state='disabled')
        threading.Thread(target=self.train_worker, args=(self.episodes,)).start()

    def train_worker(self, episodes):
        def callback(ep, reward, td):
            self.root.after(0, lambda: self._update_graph(ep, reward, td))
        train_q_learning(episodes=episodes, plot=False, callback=callback)
        self.root.after(0, self.training_finished)

    def _update_graph(self, ep, reward, td):
        self.episodes_data.append(ep)
        self.rewards_data.append(reward)
        if len(self.episodes_data) % 10 == 0 or ep == self.episodes:
            self.ax.clear()
            self.ax.plot(self.episodes_data, self.rewards_data)
            self.ax.set_title("Rewards per Episode")
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")
            self.canvas_graph.draw()

    def training_finished(self):
        self.log_text.insert(tk.END, "Training completed.\n")
        self.train_button.config(state='normal')

    def start_evaluation(self):
        if not self.running:
            self.running = True
            self.evaluate_step()

    def evaluate_step(self):
        if self.env.is_done() or not self.running:
            self.running = False
            self.log_text.insert(tk.END, f"Evaluation complete in {self.step_count} steps.\n")
            return
        action = choose_action(self.env)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        self.draw_grid()
        if info['message']:
            self.log_text.insert(tk.END, f"Step {self.step_count}: {info['message']}\n")
        self.root.after(500, self.evaluate_step)  # Next step after 500ms

    def reset_simulation(self):
        self.running = False
        grid_size = int(self.grid_var.get())
        self.env = DroneEnv(grid_size=grid_size)
        self.total_reward = 0
        self.step_count = 0
        self.cell_size = self.canvas_size // grid_size
        self.episodes_data = []
        self.rewards_data = []
        self.ax.clear()
        self.canvas_graph.draw()
        self.draw_grid()
        self.log_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    gui = DroneGUI(root)
    root.mainloop()
