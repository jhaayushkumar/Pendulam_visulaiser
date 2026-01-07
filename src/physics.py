import sympy as sp
import numpy as np

class DoublePendulumPhysics:
    def __init__(self):
        self.t = sp.symbols('t')
        self.m1, self.m2, self.l1, self.l2, self.g = sp.symbols('m1 m2 l1 l2 g')
        self.theta1, self.theta2 = sp.symbols('theta1 theta2', cls=sp.Function)
        
        self.theta1 = self.theta1(self.t)
        self.theta2 = self.theta2(self.t)
        
        self.theta1_d = sp.diff(self.theta1, self.t)
        self.theta2_d = sp.diff(self.theta2, self.t)
        self.theta1_dd = sp.diff(self.theta1_d, self.t)
        self.theta2_dd = sp.diff(self.theta2_d, self.t)
        
        self.sol = None
        self.funcs = None

    def derive_equations(self):
        """
        Derives the equations of motion using Lagrangian Mechanics.
        L = T - V
        """
        x1 = self.l1 * sp.sin(self.theta1)
        y1 = -self.l1 * sp.cos(self.theta1)
        
        x2 = x1 + self.l2 * sp.sin(self.theta2)
        y2 = y1 - self.l2 * sp.cos(self.theta2)
        
        v1_sq = sp.diff(x1, self.t)**2 + sp.diff(y1, self.t)**2
        v2_sq = sp.diff(x2, self.t)**2 + sp.diff(y2, self.t)**2
        
        T = 0.5 * self.m1 * v1_sq + 0.5 * self.m2 * v2_sq
        
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        
        self.L = T - V
        
        
        def get_LE_eq(q, q_d):
            return sp.diff(sp.diff(self.L, q_d), self.t) - sp.diff(self.L, q)
        
        eq1 = get_LE_eq(self.theta1, self.theta1_d)
        eq2 = get_LE_eq(self.theta2, self.theta2_d)
        
        print("Simplifying equations... (this might take a moment)")
        sol = sp.solve([eq1, eq2], (self.theta1_dd, self.theta2_dd))
        
        self.sol = sol
        return sol

    def get_numerical_funcs(self):
        """
        Returns lambdified functions for accelerations.
        Returns: (func_accel_1, func_accel_2)
        Signature of funcs: (t, m1, m2, l1, l2, g, theta1, theta2, w1, w2)
        """
        if self.sol is None:
            self.derive_equations()
            
        accel1_expr = self.sol[self.theta1_dd]
        accel2_expr = self.sol[self.theta2_dd]
        
        w1, w2 = sp.symbols('w1 w2')
        accel1_expr = accel1_expr.subs({self.theta1_d: w1, self.theta2_d: w2})
        accel2_expr = accel2_expr.subs({self.theta1_d: w1, self.theta2_d: w2})
        
        args = (self.t, self.m1, self.m2, self.l1, self.l2, self.g, 
                self.theta1, self.theta2, w1, w2)
        
        func1 = sp.lambdify(args, accel1_expr, modules='numpy')
        func2 = sp.lambdify(args, accel2_expr, modules='numpy')
        
        self.funcs = (func1, func2)
        return self.funcs
    
    def get_energy_func(self):
        """
        Returns a lambdified function for Total Energy (Hamiltonian = T + V) to check conservation.
        """
        x1 = self.l1 * sp.sin(self.theta1)
        y1 = -self.l1 * sp.cos(self.theta1)
        x2 = x1 + self.l2 * sp.sin(self.theta2)
        y2 = y1 - self.l2 * sp.cos(self.theta2)
        
        v1_sq = sp.diff(x1, self.t)**2 + sp.diff(y1, self.t)**2
        v2_sq = sp.diff(x2, self.t)**2 + sp.diff(y2, self.t)**2
        
        T = 0.5 * self.m1 * v1_sq + 0.5 * self.m2 * v2_sq
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        
        E = T + V
        
        w1, w2 = sp.symbols('w1 w2')
        E = E.subs({self.theta1_d: w1, self.theta2_d: w2})
        
        args = (self.t, self.m1, self.m2, self.l1, self.l2, self.g, 
                self.theta1, self.theta2, w1, w2)
        
        return sp.lambdify(args, E, modules='numpy')

if __name__ == "__main__":
    print("Deriving Physics...")
    phys = DoublePendulumPhysics()
    sol = phys.derive_equations()
    print("Derivation Complete.")
    print("Accel 1:", sol[phys.theta1_dd])
