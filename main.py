import numpy as np
from scipy.integrate import dblquad


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
def spatial_integral(t, x_j, y_i, c_wj, c_hi):
    # Define the integrand function
    def integrand(y, x):
        return P(x, y, t)

    # Bounds for x and y based on c_wj and c_hi
    x_lower = x_j - c_wj / 2
    x_upper = x_j + c_wj / 2
    y_lower = y_i - c_hi / 2
    y_upper = y_i + c_hi / 2

    # Perform the double integration
    result, _ = dblquad(integrand, x_lower, x_upper, lambda x: y_lower, lambda x: y_upper)

    return result


# Example use for one time step
t = 2.5  # Current time in seconds (s)
x_j = 0.16  # Midpoint distance of x in column j (m)
y_i = 0.24  # Midpoint distance of y in column i (m)
c_wj = 0.015  # Width of conductor track column (m)
c_hi = 0.015  # Height of conductor track row (m)

# Calculate the spatial integral
integral_value = spatial_integral(t, x_j, y_i, c_wj, c_hi)
print(f"Integral value at time t={t}: {integral_value}")
