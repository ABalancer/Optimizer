import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
from scipy.ndimage import zoom


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

    x, y = centre_of_pressure_estimate(n_r, n_c, conductor_widths, pitch_widths,
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
    mat_matrix = np.zeros((round(mat_matrix_shape[0]), round(mat_matrix_shape[1])))
    foot_height, foot_width = left_foot_profile.shape

    # Calculate top-left corner for small_matrix1
    top_left_of_left_foot = (left_foot_centre[0] - foot_height // 2, left_foot_centre[1] - foot_width // 2)

    mat_matrix[top_left_of_left_foot[0]:top_left_of_left_foot[0] + foot_height,
               top_left_of_left_foot[1]:top_left_of_left_foot[1] + foot_width] = left_foot_profile

    top_left_of_right_foot = (right_foot_centre[0] - foot_height // 2, right_foot_centre[1] - foot_width // 2)

    mat_matrix[top_left_of_right_foot[0]:top_left_of_right_foot[0] + foot_height,
               top_left_of_right_foot[1]:top_left_of_right_foot[1] + foot_width] = right_foot_profile

    return mat_matrix


def plot_heatmap(force_map):
    # Plot the heatmap
    plt.figure(figsize=(6, 6))  # Set the figure size

    # Plot the heatmap using imshow
    plt.imshow(force_map, cmap='viridis', origin='upper', interpolation='none')

    # Add a colorbar to show the intensity of values
    plt.colorbar()

    # Add labels and title
    plt.title('Heatmap of Large Matrix')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plot
    plt.show()


# Function to update the heatmap
def update_frame(frame, heatmap_line, heatmaps):
    number_of_frames = np.shape(heatmaps)[0]
    if frame > number_of_frames - 1:
        heatmap_line.set_data(heatmaps[2 * number_of_frames - frame - 1])
    else:
        heatmap_line.set_data(heatmaps[frame])
    return [heatmap_line]


def centre_of_pressure(force_map):
    # Dimensions of the image
    height, width = force_map.shape

    # Create coordinate grids for x and y
    x_coords, y_coords = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)

    # Compute the total force_map (sum of all pixel pressures)
    total_pressure = np.sum(force_map)

    # Compute the weighted sum for the x and y coordinates
    x = np.sum(x_coords * force_map) / total_pressure
    y = np.sum(y_coords * force_map) / total_pressure

    return x, y


def centre_of_pressure_estimate(conductor_heights, conductor_widths, pitch_heights, pitch_widths, force_map):
    # Initialize sums
    numerator_x = 0
    numerator_y = 0
    denominator = np.float64(0)

    # Loop over all regions i (height) and j (width)
    for i in range(conductor_heights.shape[0]):
        for j in range(conductor_widths.shape[0]):
            # Compute x_j and y_i based on the provided formulas
            x_j = 0.5 * conductor_widths[j] + sum(conductor_widths[k] + pitch_widths[k] for k in range(j))
            y_i = 0.5 * conductor_heights[i] + sum(conductor_heights[k] + pitch_heights[k] for k in range(i))

            # Add to numerators and denominator
            numerator_x += x_j * force_map[i][j]
            numerator_y += y_i * force_map[i][j]
            denominator += force_map[i][j]

    # Compute centre of force_map
    x_E = numerator_x / denominator if denominator != 0 else 0
    y_E = numerator_y / denominator if denominator != 0 else 0

    return x_E, y_E


def create_low_res_mat(conductor_heights, sensor_widths, pitch_heights, pitch_widths, high_res_heatmap_matrix):
    low_res_pressure_map = np.zeros((conductor_heights.shape[0], sensor_widths.shape[0]))

    height_midpoint = conductor_heights[0] / 2
    for i in range(0, resolution[0]):
        width_midpoint = sensor_widths[0] / 2
        for j in range(0, resolution[1]):
            low_res_pressure_map[i][j] = sum_square_section(high_res_heatmap_matrix,
                                                            (height_midpoint, width_midpoint),
                                                            sensor_widths[j], conductor_heights[i])
            width_midpoint += sensor_widths[j-1] / 2 + pitch_widths[j] + sensor_widths[j] / 2
        height_midpoint += conductor_heights[i - 1] / 2 + pitch_heights[i] + conductor_heights[i] / 2

    return low_res_pressure_map


def compute_sensing_ratios(sensor_heights, sensor_widths, pitch_heights, pitch_widths):
    sensing_ratios = np.zeros((sensor_heights.shape[0], sensor_widths.shape[0]))
    for i in range(0, sensor_heights.shape[0]):
        for j in range(0, sensor_widths.shape[0]):
            sensing_ratios[i][j] = ((sensor_heights[i] + pitch_heights[i]) * (sensor_widths[j] + pitch_widths[j])
                                    / (sensor_heights[i]*sensor_widths[j]))
    return sensing_ratios


def rescale_mass(foot_profile, mass):
    scale_factor = mass * constants.g / np.sum(foot_profile)
    return foot_profile * scale_factor


def convert_force_to_adc(R0, k, conductor_heights, conductor_widths, force_map):
    resolution = 4095
    adc_map = np.zeros(force_map.shape, dtype=np.int16)
    for i in range(conductor_heights.shape[0]):
        for j in range(conductor_widths.shape[0]):
            area = conductor_heights[i] * conductor_widths[j]
            base_resistance = R0 / area
            divider_resistance = base_resistance / 5
            sensor_resistance = R0 * area / (R0 * k * force_map[i][j] + pow(area, 2))
            offset = np.int16(np.floor(resolution * divider_resistance / (base_resistance + divider_resistance)))
            adc_map[i][j] = np.int16(np.floor(
                resolution * divider_resistance/(sensor_resistance + divider_resistance))) - offset

    return adc_map


def compute_error_for_instance(conductor_heights, conductor_widths, pitch_heights, pitch_widths, force_map, piezo=True):
    # compute real CoP
    x_cop, y_cop = centre_of_pressure(force_map)
    x_cop /= 1000
    y_cop /= 1000
    sensor_pressures = create_low_res_mat(conductor_heights, conductor_widths, pitch_heights, pitch_widths, force_map)
    # plot_heatmap(sensor_pressures)
    # compute estimated CoP
    if piezo:
        adc_map = convert_force_to_adc(R0, k, conductor_heights, conductor_widths, sensor_pressures)
    else:
        adc_map = sensor_pressures
    x_cop_e, y_cop_e = centre_of_pressure_estimate(conductor_heights, conductor_widths, pitch_heights, pitch_widths,
                                                   adc_map)

    x_e = 100 * abs((x_cop - x_cop_e) / x_cop)
    y_e = 100 * abs((y_cop - y_cop_e) / y_cop)
    print("Real x: %3.2f, y: %3.2f, Estimate x: %3.2f, y: %3.2f, Error x: %2.3f%%, y: %2.3f%%" %
          (1000 * x_cop, 1000 * y_cop, 1000 * x_cop_e, 1000 * y_cop_e, x_e, y_e))

    return x_e, y_e, adc_map


def run_weight_shift_scenario(conductor_heights, conductor_widths, pitch_heights, pitch_widths,
                              user_mass, left_foot_profile, right_foot_profile, piezo=False):
    average_x_e = 0
    average_y_e = 0

    time_step = 0.1  # Seconds
    time_steps = np.arange(0, total_time + time_step, time_step)
    number_of_time_stamps = len(time_steps)
    heatmaps = np.zeros((number_of_time_stamps, conductor_heights.shape[0], conductor_widths.shape[0]))

    for t in time_steps:
        left_foot_mass = user_mass / total_time * t
        right_foot_mass = user_mass - left_foot_mass
        temp_left_foot_profile = rescale_mass(left_foot_profile, left_foot_mass)
        temp_right_foot_profile = rescale_mass(right_foot_profile, right_foot_mass)
        high_res_heatmap_matrix = move_feet(left_foot_centre, right_foot_centre,
                                            temp_left_foot_profile, temp_right_foot_profile, high_res_resolution)
        x_e, y_e, adc_map = compute_error_for_instance(conductor_heights, conductor_widths, pitch_heights, pitch_widths,
                                                       high_res_heatmap_matrix, piezo)
        average_x_e += x_e
        average_y_e += y_e
        heatmaps[np.where(time_steps == t)] = adc_map
    average_x_e /= number_of_time_stamps
    average_y_e /= number_of_time_stamps

    # print("Average Errors x: %2.3f%%, y: %2.3f%%" % (average_x_e, average_y_e))
    return average_x_e, average_y_e, heatmaps


def create_animated_plot(heatmaps):
    # Create real-time plot
    # Set up the figure and axis
    fig, ax = plt.subplots()
    heatmap_line = ax.imshow(heatmaps[0], cmap='viridis', interpolation='none')
    cbar = plt.colorbar(heatmap_line)

    ani = animation.FuncAnimation(fig, update_frame, frames=2 * np.shape(heatmaps)[0], interval=100, blit=True,
                                  fargs=(heatmap_line, heatmaps))

    plt.show()


if __name__ == "__main__":
    # Load the array back from the .npy file
    # Scale the force_map values to represent a realistic user weight.
    total_time = 5
    user_mass = 80
    gravity = 9.81

    # create heatmap with both feet
    high_res_resolution = (512, 512)
    mat_size = (0.48, 0.48)  # in metres
    left_foot_centre = (0.24, 0.168)  # in metres
    right_foot_centre = (0.24, 0.312)  # in metres

    scale_factor = high_res_resolution[0] / mat_size[0] / 1000
    left_foot_centre = (round(left_foot_centre[0] * scale_factor * 1000),
                        round(left_foot_centre[1] * scale_factor * 1000))
    right_foot_centre = (round(right_foot_centre[0] * scale_factor * 1000),
                         round(right_foot_centre[1] * scale_factor * 1000))

    left_foot_profile = np.genfromtxt("pressure_map.csv", delimiter=',', skip_header=0, filling_values=np.nan)
    left_foot_profile = zoom(left_foot_profile, zoom=(scale_factor, scale_factor))
    left_foot_profile = rescale_mass(left_foot_profile, user_mass / 2)
    right_foot_profile = np.flip(left_foot_profile, axis=1)

    # Sensor parameters
    R0 = 0.2325  # resistance per metre squared
    k = 1.265535e-8

    # Simulation Settings
    resolution = (4, 4)

    sensor_heights = np.array(resolution[0] * [scale_factor * mat_size[0] / resolution[0] / 2])
    sensor_widths = np.array(resolution[1] * [scale_factor * mat_size[1] / resolution[1] / 2])
    pitch_heights = np.array(resolution[0] * [scale_factor * mat_size[0] / resolution[0] / 2])
    pitch_widths = np.array(resolution[1] * [scale_factor * mat_size[1] / resolution[1] / 2])

    print(sensor_heights, pitch_heights)
    # Create ranges for each width and height value, stepping by a quarter pitch width
    step_size = 4
    width_ranges = [np.arange(c_w, c_w + p_w, p_w / step_size) for c_w, p_w in zip(sensor_widths, pitch_widths)]
    height_ranges = [np.arange(c_h, c_h + p_h, p_h / step_size) for c_h, p_h in zip(sensor_heights, pitch_heights)]

    total_iterations = pow(step_size, np.shape(sensor_heights)[0]) * pow(step_size, np.shape(sensor_widths)[0])
    errors = np.zeros((total_iterations, 2), dtype=np.float64)
    indexes = np.zeros((total_iterations, 2), dtype=np.int32)
    iteration_number, i, j = 0, 0, 0
    for pitch_height_combinations in itertools.product(*height_ranges):
        for pitch_width_combinations in itertools.product(*width_ranges):
            indexes[iteration_number][0] = i
            indexes[iteration_number][1] = j
            print(indexes[iteration_number])
            # print("Widths:", pitch_width_combinations, ". Heights:", pitch_height_combinations)
            #x, y, heatmaps = run_weight_shift_scenario(sensor_heights, sensor_widths, pitch_heights, pitch_widths,
            #                                           user_mass, left_foot_profile, right_foot_profile)
            j += 1
            iteration_number += 1
        i += 1
        j = 0
    # create_animated_plot(heatmaps)

    '''
    np.save("centre_of_pressure_results.npy", cop_values)
    '''
