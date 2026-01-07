# Double Pendulum Chaos Visualizer 🌀

A high-performance, Python-based tool to simulate, analyze, and visualize the chaotic motion of a double pendulum. This project demonstrates advanced usage of **SymPy** for symbolic physics derivation, **NumPy/SciPy** for numerical integration, and **Pandas/Matplotlib** for data analysis and visualization.

## 🚀 Features

*   **Symbolic Core**: The equations of motion are derived *from scratch* at runtime using Lagrangian Mechanics (`SymPy`).
*   **High-Speed Simulation**: Numerical integration is handled by `scipy.integrate.odeint` with optimized `numy` functions.
*   **Chaos Analysis**: Computes Energy conservation to validate simulation accuracy.
*   **Visualization**: Generates phase-space plots ($\theta$ vs $\omega$) and high-quality animations of the system.

## 📦 Project Structure

```text
chaos_visualizer/
├── main.py              # Entry point to run the simulation
├── requirements.txt     # Python dependencies
├── output/              # Generated plots and animations
└── src/
    ├── physics.py       # SymPy Lagrangian Physics Engine
    ├── simulator.py     # Numerical Integration Module
    ├── analysis.py      # Pandas Data Analysis Module
    └── visualizer.py    # Matplotlib Visualization Engine
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🏃 Usage

Run the main script to start the simulation pipeline:

```bash
python3 main.py
```

The script will:
1.  Derive the equations of motion (EOM).
2.  Simulate the system for 20 seconds.
3.  Check for energy drift (numerical stability).
4.  Save simulation data to `output/simulation_data.csv`.
5.  Generate a phase space plot `output/phase_space.png`.
6.  Create an animation `output/pendulum_chaos.gif`.

## 🧮 Mathematical Theory

The system consists of two masses $m_1$ and $m_2$ connected by massless rods of length $l_1$ and $l_2$.

**Lagrangian mechanics** is used to solve the system:
$$L = T - V$$

Where $T$ is Kinetic Energy and $V$ is Potential Energy. Euler-Lagrange equations are applied:

$$\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{\theta}_i}\right) - \frac{\partial L}{\partial \theta_i} = 0$$

This project automatically performs these calculus operations symbolically to generate the exact differential equations used for the simulation.

## 📊 Sample Output

*(Run the code to see the animation locally!)*

**Energy Stability**: Typical energy drift is < 0.01% for short-term simulations, validating the physics implementation.
