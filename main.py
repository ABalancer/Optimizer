import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

# Constants
m_T = 80.0  # Total mass (kg)
g = 9.81  # Gravitational acceleration (m/s^2)
T = 5.0  # Time limit (s)
w = 0.105  # Width of the foot (m)
h = 0.25  # Height of the foot (m)
x_l, y_l = 0.168, 0.24  # Left object's centre coordinates (m)
x_r, y_r = 0.312, 0.24  # Right object's centre coordinates (m)


# Piecewise function P(x, y, t)
def P(x, y, t):
    left_condition = (2 * (y - y_l) / h) ** 2 + (2 * (x - x_l) / w) ** 2 <= 1
    right_condition = (2 * (y - y_r) / h) ** 2 + (2 * (x - x_r) / w) ** 2 <= 1

    if left_condition:
        return (4 * m_T * g / (T * w * h * np.pi)) * t
    elif right_condition:
        return (4 * m_T * g / (T * w * h * np.pi)) * (T - t)
    else:
        return 0


# Perform the double integral over x and y for a specific time t
def spatial_integral(time, x_j, y_i, c_wj, c_hi):
    # Define the integrand function
    def integrand(y, x):
        return P(x, y, time)

    # Bounds for x and y based on c_wj and c_hi
    x_lower = x_j - c_wj / 2
    x_upper = x_j + c_wj / 2
    y_lower = y_i - c_hi / 2
    y_upper = y_i + c_hi / 2

    # Perform the double integration
    result, _ = dblquad(integrand, x_lower, x_upper, lambda x: y_lower, lambda x: y_upper, epsabs=1e-8, epsrel=1e-8)

    return result


if __name__ == "__main__":

    # Example use for one time step
    t = 2.5  # Current time in seconds (s)
    c_w = 0.015  # Width of conductor track column (m)
    c_h = 0.015  # Height of conductor track row (m)
    p_w = 0.015  # Pitch width for column spacing (m)
    p_h = 0.015  # Pitch height for column spacing (m)
    n_c = 16  # Number of columns
    n_r = 16  # Number of rows
    results = np.zeros((n_r, n_c))
    centre_x, centre_y = c_w/2, c_h/2

    # Calculate the spatial integral
    for i in range(0, 16):
        for j in range(0, 16):
            results[i][j] = spatial_integral(t, centre_x, centre_y, c_w, c_h)
            centre_x += c_w + p_w
        centre_x = c_w / 2
        centre_y += c_h + p_h

    # Create the heatmap
    plt.figure(figsize=(8, 8))  # Create a figure with a set size

    # Use imshow to display the heatmap
    plt.imshow(results, cmap='hot', interpolation='nearest')

    # Add color bar to show the scale of values
    plt.colorbar()

    # Add labels and title
    plt.title("Heatmap of Integration Results")
    plt.xlabel("Columns, index j")
    plt.ylabel("Rows, index i")

    # Show the plot
    plt.show()
