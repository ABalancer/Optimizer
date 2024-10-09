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


def centre_of_pressure(number_of_rows, number_of_columns, conductor_widths, pitch_widths,
                       conductor_heights, pitch_heights, pressure):
    # Initialize sums
    numerator_x = 0
    numerator_y = 0
    denominator = 0

    # Loop over all regions i (height) and j (width)
    for i in range(number_of_rows):
        for j in range(number_of_columns):
            # Compute x_j and y_i based on the provided formulas
            x_j = 0.5 * conductor_widths[j] + sum(conductor_widths[k] + pitch_widths[k] for k in range(j))
            y_i = 0.5 * conductor_heights[i] + sum(conductor_heights[k] + pitch_heights[k] for k in range(i))

            # Add to numerators and denominator
            numerator_x += x_j * pressure[i][j]
            numerator_y += y_i * pressure[i][j]
            denominator += pressure[i][j]

    # Compute centre of pressure
    x_E = numerator_x / denominator if denominator != 0 else 0
    y_E = numerator_y / denominator if denominator != 0 else 0

    return x_E, y_E


def simulation_scenario(time_step, conductor_widths, conductor_heights, pitch_widths, pitch_heights):

    n_c = len(pitch_widths)  # Number of columns
    n_r = len(pitch_heights)  # Number of rows

    pitch_heights = np.zeros(n_r)
    pitch_widths = np.zeros(n_c)
    pressure_results = np.zeros((n_r, n_c))
    conductor_centre_x, conductor_centre_y = conductor_widths[0] / 2, conductor_heights[0] / 2
    time = 0

    for i in range(0, n_r):
        for j in range(0, n_c):
            # Calculate the spatial integral
            pressure_results[i][j] = spatial_integral(time_step, conductor_centre_x, conductor_centre_y,
                                                      conductor_widths[j], conductor_heights[i])
            conductor_centre_x += conductor_widths[j] + pitch_widths[j]
        conductor_centre_x = conductor_widths[0] / 2
        conductor_centre_y += conductor_heights[i] + pitch_heights[i]

    x, y = centre_of_pressure(n_r, n_c, conductor_widths, pitch_widths,
                              conductor_heights, pitch_heights, pressure_results)

    return x, y, pressure_results


if __name__ == "__main__":

    # Example use for one time step
    mat_width = 0.48
    mat_height = 0.48
    mat_resolution = 100
    time_step = 0.1  # Seconds

    pitch_widths = np.array(16*[0.015])
    pitch_heights = np.array(16*[0.015])
    conductor_widths = np.array(16*[0.015])
    conductor_heights = np.array(16*[0.015])
    x, y, pressure_results = simulation_scenario(1.25, conductor_widths, conductor_heights,
                                                 pitch_widths, pitch_heights)
    print(x, y)

    # Create the heatmap
    plt.figure(figsize=(8, 8))  # Create a figure with a set size

    # Use imshow to display the heatmap
    plt.imshow(pressure_results, cmap='hot', interpolation='nearest')

    # Add color bar to show the scale of values
    plt.colorbar()

    # Add labels and title
    plt.title("Heatmap of Integration Results")
    plt.xlabel("Columns, index j")
    plt.ylabel("Rows, index i")

    # Show the plot
    plt.show()
