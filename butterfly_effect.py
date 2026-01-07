import os
import numpy as np
from src.physics import DoublePendulumPhysics
from src.simulator import Simulator
from src.analysis import SimulationAnalyzer
from src.visualizer import Visualizer

def main():
    print("=== Butterfly Effect Visualizer 🦋 ===")
    print("Demonstrating Sensitivity to Initial Conditions")
    
    # 1. Physics Setup
    phys = DoublePendulumPhysics()
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.81
    sim = Simulator(phys, m1, m2, l1, l2, g)
    
    # 2. Simulation Comparison
    t_max = 20.0
    dt = 0.05
    
    # Simulation A (Base Case)
    init_state_a = [np.pi/2, np.pi/2, 0, 0]
    print(f"\n[1/3] Running Simulation A (Base)... {init_state_a}")
    t_a, states_a = sim.run(init_state_a, t_max, dt)
    
    # Simulation B (Perturbed by epsilon)
    epsilon = 1e-4 # Very small difference
    init_state_b = [np.pi/2 + epsilon, np.pi/2, 0, 0]
    print(f"\n[2/3] Running Simulation B (Perturbed by {epsilon} rad)... {init_state_b}")
    t_b, states_b = sim.run(init_state_b, t_max, dt)
    
    # 3. Analysis & Prep
    params = (m1, m2, l1, l2, g)
    analyzer_a = SimulationAnalyzer(t_a, states_a, params)
    analyzer_b = SimulationAnalyzer(t_b, states_b, params)
    
    # 4. Visualization
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    viz = Visualizer(analyzer_a.df, params)
    
    print("\n[3/3] Generating Comparison Animation...")
    try:
        viz.animate_comparison(
            analyzer_a.df, 
            analyzer_b.df, 
            filename=os.path.join(output_dir, "butterfly_effect.gif"),
            fps=20, 
            duration_seconds=t_max
        )
    except Exception as e:
        print(f"Animation failed: {e}")
        
    print(f"\n=== Done! 🦋 ===")
    print(f"Check {output_dir}/butterfly_effect.gif to see the divergence!")

if __name__ == "__main__":
    main()
