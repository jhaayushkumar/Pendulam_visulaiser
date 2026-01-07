import numpy as np
from scipy.integrate import odeint

class Simulator:
    def __init__(self, physics_engine, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        """
        Initialize the simulator.
        
        Args:
            physics_engine: Instance of DoublePendulumPhysics
            m1, m2: Masses using kg
            l1, l2: Lengths using meters
            g: Gravity using m/s^2
        """
        self.phys = physics_engine
        self.params = (m1, m2, l1, l2, g)
        
        print("Retrieving numerical functions from Physics Engine...")
        self.accel_funcs = self.phys.get_numerical_funcs()
        print("Numerical functions ready.")

    def run(self, initial_state, t_max, dt):
        """
        Run the simulation.
        
        Args:
            initial_state: [theta1, theta2, omega1, omega2] (rad, rad, rad/s, rad/s)
            t_max: Total time to simulate (s)
            dt: Time step (s)
            
        Returns:
            t: Time array
            states: Array of state vectors [theta1, theta2, omega1, omega2]
        """
        t = np.arange(0, t_max, dt)
        
        m1, m2, l1, l2, g = self.params
        accel1_f, accel2_f = self.accel_funcs
        
        def deriv(state, t):
            theta1, theta2, omega1, omega2 = state
            
            
            a1 = accel1_f(t, m1, m2, l1, l2, g, theta1, theta2, omega1, omega2)
            a2 = accel2_f(t, m1, m2, l1, l2, g, theta1, theta2, omega1, omega2)
            
            return [omega1, omega2, a1, a2]

        print(f"Starting simulation for {t_max} seconds...")
        states, info = odeint(deriv, initial_state, t, rtol=1e-9, atol=1e-9, full_output=True)
        print("Simulation complete.")
        
        if info['message'] != 'Integration successful.':
            print(f"Warning: ODE integration message: {info['message']}")
        
        return t, states

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    from src.physics import DoublePendulumPhysics
    
    phys = DoublePendulumPhysics()
    sim = Simulator(phys)
    t, states = sim.run([np.pi/2, np.pi/2, 0, 0], 5, 0.05)
    print("Final State:", states[-1])
