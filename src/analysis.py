import pandas as pd
import numpy as np

class SimulationAnalyzer:
    def __init__(self, t, states, params):
        """
        Args:
            t: Time array
            states: State array [theta1, theta2, omega1, omega2]
            params: (m1, m2, l1, l2, g)
        """
        self.t = t
        self.states = states
        self.params = params
        self.df = self._create_dataframe()

    def _create_dataframe(self):
        """Creates the initial DataFrame from simulation data."""
        df = pd.DataFrame(self.states, columns=['theta1', 'theta2', 'omega1', 'omega2'])
        df['t'] = self.t
        
        # Calculate Cartesian coordinates
        m1, m2, l1, l2, g = self.params
        
        df['x1'] = l1 * np.sin(df['theta1'])
        df['y1'] = -l1 * np.cos(df['theta1'])
        
        df['x2'] = df['x1'] + l2 * np.sin(df['theta2'])
        df['y2'] = df['y1'] - l2 * np.cos(df['theta2'])
        
        return df

    def calculate_energy(self):
        """
        Calculates Total Energy (T + V) for verification.
        Returns the DataFrame with an 'Energy' column.
        """
        m1, m2, l1, l2, g = self.params
        df = self.df
        
        # Velocity components (using chain rule manually or deriving)
        # v1^2 = (l1 * theta1_dot)^2
        # v2^2 ... formula is complex, let's use the explicit coordinate derivatives
        # dx1/dt = l1 * cos(theta1) * theta1_dot
        # dy1/dt = l1 * sin(theta1) * theta1_dot
        
        vx1 = l1 * np.cos(df['theta1']) * df['omega1']
        vy1 = l1 * np.sin(df['theta1']) * df['omega1']
        
        vx2 = vx1 + l2 * np.cos(df['theta2']) * df['omega2']
        vy2 = vy1 + l2 * np.sin(df['theta2']) * df['omega2']
        
        v1_sq = vx1**2 + vy1**2
        v2_sq = vx2**2 + vy2**2
        
        # Kinetic Energy
        T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        
        # Potential Energy
        V = m1 * g * df['y1'] + m2 * g * df['y2']
        
        df['T'] = T
        df['V'] = V
        df['E'] = T + V
        
        return df
    
    def save_csv(self, filename="simulation_data.csv"):
        self.df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Test stub
    t = np.linspace(0, 10, 100)
    states = np.zeros((100, 4))
    params = (1.0, 1.0, 1.0, 1.0, 9.81)
    analyzer = SimulationAnalyzer(t, states, params)
    df = analyzer.calculate_energy()
    print(df.head())
