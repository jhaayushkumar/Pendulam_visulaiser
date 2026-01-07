import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.physics import DoublePendulumPhysics
from src.simulator import Simulator
from src.analysis import SimulationAnalyzer

def run_simulation():
    phys = DoublePendulumPhysics()
    m1, m2, l1, l2, g = 1.0, 1.0, 1.0, 1.0, 9.81
    sim = Simulator(phys, m1, m2, l1, l2, g)
    
    init_state = [np.pi/2, np.pi/2, 0, 0]
    t, states = sim.run(init_state, 30.0, 0.05)
    
    params = (m1, m2, l1, l2, g)
    analyzer = SimulationAnalyzer(t, states, params)
    analyzer.calculate_energy()
    return analyzer.df

def create_dashboard(df, filename="output/dashboard.png"):
    print("Generating Data Science Dashboard...")
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Double Pendulum: Chaos Analytics Dashboard', fontsize=20, color='white')
    
    gs = fig.add_gridspec(2, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['x2'], df['y2'], lw=0.5, alpha=0.8, color='cyan')
    ax1.set_title("Pendulum 2 Trajectory (Real Space)")
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.grid(True, alpha=0.2)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['theta2'], df['omega2'], lw=0.5, alpha=0.8, color='magenta')
    ax2.set_title("Phase Space: Attractor (Angle vs Velocity)")
    ax2.set_xlabel("Theta (rad)")
    ax2.set_ylabel("Omega (rad/s)")
    ax2.grid(True, alpha=0.2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    E_mean = df['E'].mean()
    E_dev = (df['E'] - E_mean) / E_mean * 100
    ax3.plot(df['t'], E_dev, color='lime', lw=1)
    ax3.set_title("System Energy Stability (% Drift)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Deviation from Mean (%)")
    ax3.grid(True, alpha=0.2)
    
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(df['t'], df['theta1'], label=r'$\theta_1$ (Inner)', color='yellow', alpha=0.7)
    ax4.plot(df['t'], df['theta2'], label=r'$\theta_2$ (Outer)', color='cyan', alpha=0.7)
    ax4.set_title("Time Series Analysis: Angular Evolution")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (rad)")
    ax4.legend()
    ax4.grid(True, alpha=0.2)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=150)
    print(f"Dashboard saved to {filename}")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    df = run_simulation()
    create_dashboard(df)
