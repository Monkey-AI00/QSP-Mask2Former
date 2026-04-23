import numpy as np
import matplotlib.pyplot as plt

# Given values
r = 0.0001
c = np.linspace(0, 15, 8000)

# Example rtde_r.getActualTCPPose(), assuming initial positions as [0, 0] for simplicity
initial_y = 0
initial_z = 0

# Calculating x and y based on the equations
y = initial_y + c * r * np.cos(np.pi * 5 * c)
z = initial_z + c * r * np.sin(np.pi * 5 * c) * 1.5

# Plotting the trajectory
plt.figure(figsize=(8, 6))
plt.plot(y, z)
plt.title("Generated Path Based on the Given Equation")
plt.xlabel("Y")
plt.ylabel("Z")
plt.grid(True)
plt.show()
