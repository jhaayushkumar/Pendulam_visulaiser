import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

class Visualizer:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.dt = df['t'].iloc[1] - df['t'].iloc[0]

    def plot_phase_space(self, filename="phase_space.png"):
        """Plots the phase space (theta vs omega) for both pendulums."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        points = np.array([self.df['theta1'], self.df['omega1']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(self.df['t'].min(), self.df['t'].max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(self.df['t'])
        lc.set_linewidth(1)
        ax1.add_collection(lc)
        ax1.autoscale()
        ax1.set_title("Pendulum 1 Phase Space")
        ax1.set_xlabel(r"$\theta_1$ (rad)")
        ax1.set_ylabel(r"$\omega_1$ (rad/s)")
        
        points = np.array([self.df['theta2'], self.df['omega2']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='inferno', norm=norm)
        lc.set_array(self.df['t'])
        lc.set_linewidth(1)
        ax2.add_collection(lc)
        ax2.autoscale()
        ax2.set_title("Pendulum 2 Phase Space")
        ax2.set_xlabel(r"$\theta_2$ (rad)")
        ax2.set_ylabel(r"$\omega_2$ (rad/s)")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Phase space plot saved to {filename}")
        plt.close()

    def animate(self, filename="simulation.mp4", fps=30, duration_seconds=15):
        """
        Creates an animation of the double pendulum.
        Refined for speed and aesthetics.
        """
        
        sim_dt = self.dt
        frame_step = int((1/fps) / sim_dt)
        if frame_step < 1: frame_step = 1
        
        max_frames = int(duration_seconds * fps)
        
        data = self.df.iloc[::frame_step].head(max_frames).reset_index(drop=True)
        
        l1, l2 = self.params[2], self.params[3]
        limit = l1 + l2 + 0.5
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("Double Pendulum Chaos")
        
        line, = ax.plot([], [], 'o-', lw=2, color='black')
        trace, = ax.plot([], [], '-', lw=1, alpha=0.5, color='red')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
        history_len = int(2 * fps)
        history_x, history_y = [], []
        
        def init():
            line.set_data([], [])
            trace.set_data([], [])
            time_text.set_text('')
            return line, trace, time_text
        
        def update(i):
            if i >= len(data): return line, trace, time_text
            
            x1, y1 = data.loc[i, 'x1'], data.loc[i, 'y1']
            x2, y2 = data.loc[i, 'x2'], data.loc[i, 'y2']
            
            line.set_data([0, x1, x2], [0, y1, y2])
            
            history_x.append(x2)
            history_y.append(y2)
            
            if len(history_x) > history_len:
                history_x.pop(0)
                history_y.pop(0)
                
            trace.set_data(history_x, history_y)
            time_text.set_text(time_template % data.loc[i, 't'])
            
            return line, trace, time_text
        
        print("Generating animation...")
        ani = animation.FuncAnimation(fig, update, frames=len(data),
                                      init_func=init, interval=1000/fps, blit=True)
        
        if filename.endswith('.gif'):
            ani.save(filename, writer='pillow', fps=fps)
        else:
            try:
                ani.save(filename, writer='ffmpeg', fps=fps)
            except Exception as e:
                print(f"FFMpeg failed: {e}. Falling back to GIF.")
                filename = filename.replace('.mp4', '.gif')
                ani.save(filename, writer='pillow', fps=fps)
                
        print(f"Animation saved to {filename}")
        plt.close()

    def animate_comparison(self, df1, df2, filename="butterfly_effect.mp4", fps=30, duration_seconds=15):
        sim_dt = self.dt
        frame_step = int((1/fps) / sim_dt)
        if frame_step < 1: frame_step = 1
        
        max_frames = int(duration_seconds * fps)
        
        d1 = df1.iloc[::frame_step].head(max_frames).reset_index(drop=True)
        d2 = df2.iloc[::frame_step].head(max_frames).reset_index(drop=True)
        
        l1, l2 = self.params[2], self.params[3]
        limit = l1 + l2 + 0.5
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("Butterfly Effect: Sensitivity to Initial Conditions")
        
        line1, = ax.plot([], [], 'o-', lw=2, color='black', label='Sim A')
        trace1, = ax.plot([], [], '-', lw=1, alpha=0.5, color='cyan')
        
        line2, = ax.plot([], [], 'o-', lw=2, color='red', alpha=0.6, label='Sim B (Perturbed)')
        trace2, = ax.plot([], [], '-', lw=1, alpha=0.5, color='magenta')
        
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        ax.legend(loc='upper right')
        
        history_len = int(2 * fps)
        hist1_x, hist1_y = [], []
        hist2_x, hist2_y = [], []
        
        def init():
            line1.set_data([], [])
            trace1.set_data([], [])
            line2.set_data([], [])
            trace2.set_data([], [])
            time_text.set_text('')
            return line1, trace1, line2, trace2, time_text
        
        def update(i):
            if i >= len(d1) or i >= len(d2): return line1, trace1, line2, trace2, time_text
            
            x1a, y1a = d1.loc[i, 'x1'], d1.loc[i, 'y1']
            x2a, y2a = d1.loc[i, 'x2'], d1.loc[i, 'y2']
            
            x1b, y1b = d2.loc[i, 'x1'], d2.loc[i, 'y1']
            x2b, y2b = d2.loc[i, 'x2'], d2.loc[i, 'y2']
            
            line1.set_data([0, x1a, x2a], [0, y1a, y2a])
            line2.set_data([0, x1b, x2b], [0, y1b, y2b])
            
            hist1_x.append(x2a)
            hist1_y.append(y2a)
            hist2_x.append(x2b)
            hist2_y.append(y2b)
            
            if len(hist1_x) > history_len:
                hist1_x.pop(0)
                hist1_y.pop(0)
                hist2_x.pop(0)
                hist2_y.pop(0)
                
            trace1.set_data(hist1_x, hist1_y)
            trace2.set_data(hist2_x, hist2_y)
            
            time_text.set_text(time_template % d1.loc[i, 't'])
            
            return line1, trace1, line2, trace2, time_text
        
        print("Generating comparison animation...")
        ani = animation.FuncAnimation(fig, update, frames=min(len(d1), len(d2)),
                                      init_func=init, interval=1000/fps, blit=True)
        
        if filename.endswith('.gif'):
            ani.save(filename, writer='pillow', fps=fps)
        else:
            try:
                ani.save(filename, writer='ffmpeg', fps=fps)
            except Exception as e:
                print(f"FFMpeg failed: {e}. Falling back to GIF.")
                filename = filename.replace('.mp4', '.gif')
                ani.save(filename, writer='pillow', fps=fps)
                
        print(f"Comparison saved to {filename}")
        plt.close()

if __name__ == "__main__":
    pass
