import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


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


def simulation_scenario(time, conductor_widths, conductor_heights, pitch_widths, pitch_heights):

    n_c = len(pitch_widths)  # Number of columns
    n_r = len(pitch_heights)  # Number of rows

    pressure_results = np.zeros((n_r, n_c))
    conductor_centre_x, conductor_centre_y = conductor_widths[0] / 2, conductor_heights[0] / 2

    for i in range(0, n_r):
        for j in range(0, n_c):
            # Calculate the spatial integral
            pressure_results[i][j] = spatial_integral(time, conductor_centre_x, conductor_centre_y,
                                                      conductor_widths[j], conductor_heights[i])
            conductor_centre_x += conductor_widths[j] + pitch_widths[j]
        conductor_centre_x = conductor_widths[0] / 2
        conductor_centre_y += conductor_heights[i] + pitch_heights[i]

    x, y = centre_of_pressure(n_r, n_c, conductor_widths, pitch_widths,
                              conductor_heights, pitch_heights, pressure_results)

    return x, y, pressure_results


def bruteforcer():
    print("Todo")


def sum_square_section(matrix, midpoint, row_length, col_length):
    row_length *= 1000
    col_length *= 1000

    half_row_length = row_length / 2  # Half of the row length
    half_col_length = col_length / 2  # Half of the col length

    # Calculate the top-left corner of the square
    row_start = 1000 * midpoint[0] - half_row_length
    col_start = 1000 * midpoint[1] - half_col_length

    # Calculate the row and column end positions (be careful with boundaries)
    row_end = row_start + row_length
    col_end = col_start + col_length

    # Ensure the boundaries don't exceed the matrix dimensions
    row_start = int(max(0, row_start))
    col_start = int(max(0, col_start))
    row_end = int(min(matrix.shape[0], row_end))
    col_end = int(min(matrix.shape[1], col_end))

    # Extract the sub matrix and sum its values
    sub_matrix = matrix[row_start:row_end, col_start:col_end]
    return np.sum(sub_matrix)


def move_feet(left_foot_centre, right_foot_centre, left_foot_profile, right_foot_profile, mat_matrix_shape):
    mat_matrix = np.zeros((round(1000*mat_matrix_shape[0]), round(1000*mat_matrix_shape[1])))
    foot_height, foot_width = left_foot_profile.shape

    # Calculate top-left corner for small_matrix1
    top_left_of_left_foot = (left_foot_centre[0] - foot_height // 2, left_foot_centre[1] - foot_width // 2)

    mat_matrix[top_left_of_left_foot[0]:top_left_of_left_foot[0] + foot_height,
               top_left_of_left_foot[1]:top_left_of_left_foot[1] + foot_width] = left_foot_profile

    top_left_of_right_foot = (right_foot_centre[0] - foot_height // 2, right_foot_centre[1] - foot_width // 2)

    mat_matrix[top_left_of_right_foot[0]:top_left_of_right_foot[0] + foot_height,
               top_left_of_right_foot[1]:top_left_of_right_foot[1] + foot_width] = right_foot_profile

    return mat_matrix


def plot_heatmap(heatmap_matrix):
    # Plot the heatmap
    plt.figure(figsize=(6, 6))  # Set the figure size

    # Plot the heatmap using imshow
    plt.imshow(heatmap_matrix, cmap='viridis', origin='upper', interpolation='none')

    # Add a colorbar to show the intensity of values
    plt.colorbar()

    # Add labels and title
    plt.title('Heatmap of Large Matrix')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plot
    plt.show()


def high_res_centre_of_pressure(heatmap_matrix):
    # Dimensions of the image
    height, width = heatmap_matrix.shape

    # Create coordinate grids for x and y
    x_coords, y_coords = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)

    # Compute the total pressure (sum of all pixel pressures)
    total_pressure = np.sum(heatmap_matrix)

    # Compute the weighted sum for the x and y coordinates
    x = np.sum(x_coords * heatmap_matrix) / total_pressure
    y = np.sum(y_coords * heatmap_matrix) / total_pressure

    return x, y


def create_low_res_mat(sensor_heights, sensor_widths, pitch_heights, pitch_widths):
    low_res_pressure_map = np.zeros((sensor_heights.shape[0], sensor_widths.shape[0]))

    height_midpoint = sensor_heights[0] / 2
    for i in range(0, resolution[0]):
        width_midpoint = sensor_widths[0] / 2
        for j in range(0, resolution[1]):
            low_res_pressure_map[i][j] = sum_square_section(heatmap_matrix, (height_midpoint, width_midpoint),
                                                            sensor_widths[j], sensor_heights[i])
            width_midpoint += sensor_widths[j-1] / 2 + pitch_widths[j] + sensor_widths[j]/2
        height_midpoint += sensor_heights[i-1] / 2 + pitch_heights[i] + sensor_heights[i]/2

    return low_res_pressure_map


def rescale_mass(foot_profile, mass):
    scale_factor = mass * constants.g / np.sum(foot_profile)
    return foot_profile * scale_factor


if __name__ == "__main__":
    # Load the array back from the .npy file
    # Scale the pressure values to represent a realistic user weight.
    user_mass = 80
    gravity = 9.81
    left_foot_profile = np.genfromtxt("pressure_map.csv", delimiter=',', skip_header=0, filling_values=np.nan)
    left_foot_profile = rescale_mass(left_foot_profile, user_mass/2)
    right_foot_profile = np.flip(left_foot_profile, axis=1)

    # create heatmap with both feet
    mat_size = (0.48, 0.48)  # in metres
    left_foot_centre = (240, 168)
    right_foot_centre = (240, 312)
    heatmap_matrix = move_feet(left_foot_centre, right_foot_centre, left_foot_profile, right_foot_profile, mat_size)
    plot_heatmap(heatmap_matrix)

    # compute CoP
    x_cop, y_cop = high_res_centre_of_pressure(heatmap_matrix)
    print(x_cop, y_cop)

    resolution = (16, 16)
    sensor_heights = np.array(resolution[0]*[mat_size[0]/resolution[0]/2])
    sensor_widths = np.array(resolution[1]*[mat_size[1]/resolution[1]/2])
    pitch_heights = np.array(resolution[0]*[mat_size[0]/resolution[0]/2])
    pitch_widths = np.array(resolution[1]*[mat_size[1]/resolution[1]/2])

    sensor_results_16 = create_low_res_mat(sensor_heights, sensor_widths, pitch_heights, pitch_widths)
    plot_heatmap(sensor_results_16)
    print(np.sum(sensor_results_16)*4/gravity)

    '''
    # Simulation Settings
    time_step = 0.1  # Seconds

    time_steps = np.arange(0, 5 + time_step, time_step)
    cop_values = np.zeros((len(time_steps), 3))
    pitch_widths = np.array(128*[0.001875])
    pitch_heights = np.array(128*[0.001875])
    conductor_widths = np.array(128*[0.001875])
    conductor_heights = np.array(128*[0.001875])
    for t in time_steps:
        x, y, pressure_results = simulation_scenario(t, conductor_widths, conductor_heights,
                                                     pitch_widths, pitch_heights)
        cop_values[np.where(time_steps == t)[0]] = [t, x, y]
        print("time = %f, x = %f, y = %f" % (t, x, y))
        # update_heatmap(pressure_results)

    np.save("centre_of_pressure_results.npy", cop_values)
    '''
