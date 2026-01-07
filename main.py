import os
import numpy as np
import matplotlib.pyplot as plt
from src.physics import DoublePendulumPhysics
from src.simulator import Simulator
from src.analysis import SimulationAnalyzer
from src.visualizer import Visualizer

def main():
    print("=== Double Pendulum Chaos Visualizer ===")
    
    # 1. Physics
    print("\n[1/5] Initialization Physics Engine...")
    phys = DoublePendulumPhysics()
    
    # 2. Simulator setup
    # Parameters matching a known chaotic configuration
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.81
    
    sim = Simulator(phys, m1, m2, l1, l2, g)
    
    # Initial state: High energy state to ensure chaos
    # [theta1, theta2, omega1, omega2]
    init_state = [np.pi/2, np.pi/2, 0, 0]
    
    t_max = 20.0
    dt = 0.05
    
    # 3. Running Simulation
    print(f"\n[2/5] Running Simulation for {t_max}s...")
    t, states = sim.run(init_state, t_max, dt)
    
    # 4. Analysis
    print("\n[3/5] Analyzing Data...")
    params = (m1, m2, l1, l2, g)
    analyzer = SimulationAnalyzer(t, states, params)
    analyzer.calculate_energy()
    
    # Check energy conservation
    E = analyzer.df['E']
    E_drift = (E.max() - E.min()) / np.abs(E.mean()) * 100
    print(f"Energy Drift: {E_drift:.4f}%")
    
    # Save Data
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    analyzer.save_csv(os.path.join(output_dir, "simulation_data.csv"))
    
    # 5. Visualization
    print("\n[4/5] Generating Visualizations...")
    viz = Visualizer(analyzer.df, params)
    
    # Phase Space
    viz.plot_phase_space(os.path.join(output_dir, "phase_space.png"))
    
    # Animation
    print("\n[5/5] Creating Animation (this may take a minute)...")
    try:
        viz.animate(os.path.join(output_dir, "pendulum_chaos.gif"), fps=20, duration_seconds=t_max)
    except Exception as e:
        print(f"Animation failed: {e}")
    
    print(f"\n=== Done! Output saved to {output_dir}/ ===")

if __name__ == "__main__":
    main()
