import numpy as np
from scipy import constants
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import itertools
from scipy.ndimage import zoom


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


def move_feet(_left_foot_centre, _right_foot_centre, _left_foot_profile, _right_foot_profile, _mat_matrix_shape):
    _mat_matrix = np.zeros((round(_mat_matrix_shape[0]), round(_mat_matrix_shape[1])))
    _foot_height, _foot_width = _left_foot_profile.shape

    # Calculate top-left corner for small_matrix1
    _top_left_of_left_foot = (round(_left_foot_centre[0]) - _foot_height // 2,
                              round(_left_foot_centre[1]) - _foot_width // 2)

    _mat_matrix[_top_left_of_left_foot[0]:_top_left_of_left_foot[0] + _foot_height,
                _top_left_of_left_foot[1]:_top_left_of_left_foot[1] + _foot_width] = _left_foot_profile

    _top_left_of_right_foot = (round(_right_foot_centre[0]) - _foot_height // 2,
                               round(_right_foot_centre[1]) - _foot_width // 2)

    _mat_matrix[_top_left_of_right_foot[0]:_top_left_of_right_foot[0] + _foot_height,
                _top_left_of_right_foot[1]:_top_left_of_right_foot[1] + _foot_width] = _right_foot_profile

    return _mat_matrix


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
    _x = np.sum(x_coords * force_map) / total_pressure
    _y = np.sum(y_coords * force_map) / total_pressure

    return _x, _y


def centre_of_pressure_estimate(_conductor_heights, _conductor_widths, _pitch_heights, _pitch_widths, _force_map):
    # Initialize sums
    _numerator_x = 0
    _numerator_y = 0
    _denominator = np.float64(0)

    # Loop over all regions i (height) and j (width)
    for _i in range(_conductor_heights.shape[0]):
        for _j in range(_conductor_widths.shape[0]):
            # Compute x_j and y_i based on the provided formulas
            _x_j = 0.5 * (_conductor_widths[_j]) + sum(_conductor_widths[_k] + _pitch_widths[_k] for _k in range(_j))
            _y_i = 0.5 * (_conductor_heights[_i]) + sum(_conductor_heights[_k] + _pitch_heights[_k] for _k in range(_i))

            # Add to numerators and denominator
            _numerator_x += _x_j * _force_map[_i][_j]
            _numerator_y += _y_i * _force_map[_i][_j]
            _denominator += _force_map[_i][_j]

    # Compute centre of force_map
    _x_E = _numerator_x / _denominator if _denominator != 0 else 0
    _y_E = _numerator_y / _denominator if _denominator != 0 else 0

    return _x_E, _y_E


def create_low_res_mat(_conductor_heights, _conductor_widths, _pitch_heights, _pitch_widths, high_res_heatmap_matrix):
    low_res_pressure_map = np.zeros((_conductor_heights.shape[0], _conductor_widths.shape[0]))

    height_midpoint = _conductor_heights[0] / 2
    for _i in range(0, resolution[0]):
        width_midpoint = _conductor_widths[0] / 2
        for _j in range(0, resolution[1]):
            low_res_pressure_map[_i][_j] = sum_square_section(high_res_heatmap_matrix,
                                                              (height_midpoint, width_midpoint),
                                                              _conductor_widths[_j], _conductor_heights[_i])
            width_midpoint += _conductor_widths[_j - 1] / 2 + _pitch_widths[_j] + _conductor_widths[_j] / 2
        height_midpoint += _conductor_heights[_i - 1] / 2 + _pitch_heights[_i] + _conductor_heights[_i] / 2

    return low_res_pressure_map


def compute_sensing_ratios(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths):
    sensing_ratios = np.zeros((_sensor_heights.shape[0], _sensor_widths.shape[0]))
    for _i in range(0, _sensor_heights.shape[0]):
        for _j in range(0, _sensor_widths.shape[0]):
            sensing_ratios[_i][_j] = ((_sensor_heights[_i] + _pitch_heights[_i]) *
                                      (_sensor_widths[_j] + _pitch_widths[_j])
                                      / (_sensor_heights[_i] * _sensor_widths[_j]))
    return sensing_ratios


def rescale_mass(foot_profile, mass):
    scale_factor = mass * constants.g / np.sum(foot_profile)
    return foot_profile * scale_factor


def convert_force_to_adc(_R0, _k, _conductor_heights, _conductor_widths, _force_map, _random_map=None):
    _resolution = 4095
    _adc_map = np.zeros(_force_map.shape, dtype=np.int16)
    for _i in range(_conductor_heights.shape[0]):
        for _j in range(_conductor_widths.shape[0]):
            area = _conductor_heights[_i] * _conductor_widths[_j]
            base_resistance = _R0 / area
            divider_resistance = base_resistance / 5
            if _random_map is None:
                sensor_resistance = _R0 * area / (_R0 * _k * _force_map[_i][_j] + pow(area, 2))
                adc_result = np.int16(np.round(
                    _resolution * divider_resistance/(sensor_resistance + divider_resistance)))
            else:
                force = (_force_map[_i][_j] / area + _random_map[_i][_j]) * area
                sensor_resistance = _R0 * area / (_R0 * _k * force + pow(area, 2))
                threshold_resistance = _R0 * area / (_R0 * _k * FORCE_RANDOM_OFFSET * area + pow(area, 2))
                adc_threshold = np.int16(np.round(_resolution * divider_resistance /
                                                  (threshold_resistance + divider_resistance)))
                adc_offset = np.int16(np.round(_resolution * divider_resistance /
                                               (base_resistance + divider_resistance)))
                adc_result = np.int16(np.round(_resolution * divider_resistance /
                                               (sensor_resistance + divider_resistance)))
                # removes random base offsets
                if adc_result <= adc_threshold:
                    adc_result = adc_offset
            restored_pressure = 1 / (_R0 * _k) * (_R0 / (divider_resistance * (_resolution / adc_result - 1)) - area)
            _adc_map[_i][_j] = restored_pressure * area

    return _adc_map


def compute_error_for_instance(_sensor_heights, _sensor_widths,
                               _pitch_heights, _pitch_widths, _force_map, _piezo=False, _random_map=None):
    # compute real CoP
    x_cop, y_cop = centre_of_pressure(_force_map)
    x_cop /= 1000
    y_cop /= 1000
    sensor_forces = create_low_res_mat(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths, _force_map)
    # compute estimated CoP
    if _piezo:
        adc_map = convert_force_to_adc(R0, k, _sensor_heights, _sensor_widths, sensor_forces, _random_map)
    else:
        adc_map = sensor_forces
    x_cop_e, y_cop_e = centre_of_pressure_estimate(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                                                   adc_map)
    _x_e = 100 * abs((x_cop - x_cop_e) / x_cop)
    _y_e = 100 * abs((y_cop - y_cop_e) / y_cop)
    _a_e = compute_absolute_error(_x_e, _y_e)
    return _x_e, _y_e, _a_e, adc_map, [x_cop, y_cop, x_cop_e, y_cop_e]


def run_side_weight_shift_scenario(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                                   _user_mass, _left_foot_profile, _right_foot_profile, _piezo=False, _random_map=None):
    _average_x_e = 0
    _average_y_e = 0
    _average_a_e = 0

    time_step = 0.1  # Seconds
    time_steps = np.arange(0, total_time + time_step, time_step)
    number_of_time_stamps = len(time_steps)
    heatmaps = np.zeros((number_of_time_stamps, _sensor_heights.shape[0], _sensor_widths.shape[0]))

    cop_data = np.zeros((number_of_time_stamps, 2))
    raw_data = np.zeros((number_of_time_stamps, 4))
    for t in time_steps:
        left_foot_mass = _user_mass / total_time * (total_time - t)
        right_foot_mass = _user_mass - left_foot_mass
        temp_left_foot_profile = rescale_mass(_left_foot_profile, left_foot_mass)
        temp_right_foot_profile = rescale_mass(_right_foot_profile, right_foot_mass)
        high_res_heatmap_matrix = move_feet(left_foot_centre, right_foot_centre,
                                            temp_left_foot_profile, temp_right_foot_profile, high_res_resolution)
        _x_e, _y_e, _a_e, adc_map, raw_results = compute_error_for_instance(_sensor_heights, _sensor_widths,
                                                                            _pitch_heights, _pitch_widths,
                                                                            high_res_heatmap_matrix, _piezo,
                                                                            _random_map)
        _average_x_e += _x_e
        _average_y_e += _y_e
        _average_a_e += _a_e

        heatmaps[np.where(time_steps == t)] = adc_map
        cop_data[np.where(time_steps == t)[0][0]][0] = _x_e
        cop_data[np.where(time_steps == t)[0][0]][1] = _y_e
        for _i in range(len(raw_results)):
            raw_data[np.where(time_steps == t)[0][0]][_i] = raw_results[_i]

    _average_x_e /= number_of_time_stamps
    _average_y_e /= number_of_time_stamps
    _average_a_e /= number_of_time_stamps

    return _average_x_e, _average_y_e, _average_a_e, raw_data, heatmaps


def run_foot_slide_scenario(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                            _user_mass, _left_foot_profile, _right_foot_profile, _piezo=False, _random_map=None):
    average_x_e = 0
    average_y_e = 0
    average_a_e = 0

    time_step = 0.1  # Seconds
    time_steps = np.arange(0, total_time + time_step, time_step)
    number_of_time_stamps = len(time_steps)
    heatmaps = np.zeros((number_of_time_stamps, _sensor_heights.shape[0], _sensor_widths.shape[0]))
    left_foot_mass = _user_mass / 2
    right_foot_mass = _user_mass / 2
    _left_foot_centre = (240, 150)  # in mm
    _right_foot_centre = (240, 330)  # in mm
    _left_foot_centre = (round(_left_foot_centre[0] * SCALE_FACTOR), round(_left_foot_centre[1] * SCALE_FACTOR))
    _right_foot_centre = (round(_right_foot_centre[0] * SCALE_FACTOR), round(_right_foot_centre[1] * SCALE_FACTOR))
    temp_left_foot_profile = rescale_mass(_left_foot_profile, left_foot_mass)
    temp_right_foot_profile = rescale_mass(_right_foot_profile, right_foot_mass)
    foot_height, foot_width = _left_foot_profile.shape
    left_foot_start = round(foot_width / 2)
    right_foot_end = (1000 * rescaled_mat_size[1]) - round(foot_width / 2)
    left_foot_gradient = 2 * (_left_foot_centre[1] - left_foot_start) / total_time
    right_foot_gradient = 2 * (right_foot_end - _right_foot_centre[1]) / total_time

    cop_data = np.zeros((number_of_time_stamps, 2))
    raw_data = np.zeros((number_of_time_stamps, 4))
    for t in time_steps:
        if t < total_time / 2:
            left_foot_position = (_left_foot_centre[0], left_foot_gradient * t + left_foot_start)
            right_foot_position = _right_foot_centre
        else:
            left_foot_position = _left_foot_centre
            right_foot_position = (_right_foot_centre[0], right_foot_gradient * t
                                   + 2 * _right_foot_centre[1] - right_foot_end)

        high_res_heatmap_matrix = move_feet(left_foot_position, right_foot_position,
                                            temp_left_foot_profile, temp_right_foot_profile, high_res_resolution)
        _x_e, _y_e, _a_e, adc_map, raw_results = compute_error_for_instance(_sensor_heights, _sensor_widths,
                                                                            _pitch_heights, _pitch_widths,
                                                                            high_res_heatmap_matrix, _piezo,
                                                                            _random_map)
        average_x_e += _x_e
        average_y_e += _y_e
        average_a_e += _a_e

        heatmaps[np.where(time_steps == t)] = adc_map
        cop_data[np.where(time_steps == t)[0][0]][0] = _x_e
        cop_data[np.where(time_steps == t)[0][0]][1] = _y_e
        for _i in range(len(raw_results)):
            raw_data[np.where(time_steps == t)[0][0]][_i] = raw_results[_i]
    average_x_e /= number_of_time_stamps
    average_y_e /= number_of_time_stamps
    average_a_e /= number_of_time_stamps

    # print("Average Errors x: %2.3f%%, y: %2.3f%%" % (average_x_e, average_y_e))
    return average_x_e, average_y_e, average_a_e, raw_data, heatmaps


def run_front_weight_shift_scenario(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                                    _user_mass, _left_foot_profile, _right_foot_profile,
                                    _piezo=False, _random_map=None):
    _average_x_e = 0
    _average_y_e = 0
    _average_a_e = 0

    time_step = 0.1  # Seconds
    time_steps = np.arange(0, total_time + time_step, time_step)
    number_of_time_stamps = len(time_steps)
    heatmaps = np.zeros((number_of_time_stamps, _sensor_heights.shape[0], _sensor_widths.shape[0]))

    cop_data = np.zeros((number_of_time_stamps, 2))
    raw_data = np.zeros((number_of_time_stamps, 4))
    for t in time_steps:
        if t <= total_time / 2:
            bottom_cut_off = 0
            top_cut_off = 4 * t / (3 * total_time) + 1 / 3
        else:
            bottom_cut_off = 4 / (3 * total_time) * t - 2 / 3
            top_cut_off = 1
        left_foot = redistribute_y_pressure(_left_foot_profile,
                                            (bottom_cut_off, top_cut_off), _user_mass / 2)
        right_foot = redistribute_y_pressure(_right_foot_profile,
                                             (bottom_cut_off, top_cut_off), _user_mass / 2)

        high_res_heatmap_matrix = move_feet(left_foot_centre, right_foot_centre,
                                            left_foot, right_foot, high_res_resolution)
        _x_e, _y_e, _a_e, adc_map, raw_results = compute_error_for_instance(_sensor_heights, _sensor_widths,
                                                                            _pitch_heights, _pitch_widths,
                                                                            high_res_heatmap_matrix, _piezo,
                                                                            _random_map)
        _average_x_e += _x_e
        _average_y_e += _y_e
        _average_a_e += _a_e

        heatmaps[np.where(time_steps == t)] = adc_map
        cop_data[np.where(time_steps == t)[0][0]][0] = _x_e
        cop_data[np.where(time_steps == t)[0][0]][1] = _y_e
        for _i in range(len(raw_results)):
            raw_data[np.where(time_steps == t)[0][0]][_i] = raw_results[_i]
    _average_x_e /= number_of_time_stamps
    _average_y_e /= number_of_time_stamps
    _average_a_e /= number_of_time_stamps
    return _average_x_e, _average_y_e, _average_a_e, raw_data, heatmaps


def run_layout_scenarios(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                         _user_mass, _left_foot_profile, _right_foot_profile, _piezo=False, _random_map=None):
    _x_error = 0
    _y_error = 0

    (x_error_1, y_error_1, a_error_1,
     side_weight_shift_raw_data, heatmaps) = run_side_weight_shift_scenario(_sensor_heights, _sensor_widths,
                                                                            _pitch_heights, _pitch_widths, _user_mass,
                                                                            _left_foot_profile, _right_foot_profile,
                                                                            _piezo, _random_map)

    (x_error_2, y_error_2, a_error_2,
     front_weight_shift_raw_data, heatmaps) = run_front_weight_shift_scenario(_sensor_heights, _sensor_widths,
                                                                              _pitch_heights, _pitch_widths, _user_mass,
                                                                              _left_foot_profile, _right_foot_profile,
                                                                              _piezo, _random_map)

    (x_error_3, y_error_3, a_error_3,
     foot_slide_raw_data, heatmaps) = run_foot_slide_scenario(_sensor_heights, _sensor_widths,
                                                              _pitch_heights, _pitch_widths, _user_mass,
                                                              _left_foot_profile, _right_foot_profile,
                                                              _piezo, _random_map)

    _x_error = (x_error_1 + x_error_2 + x_error_3) / 3
    _y_error = (y_error_1 + y_error_2 + y_error_3) / 3
    _a_error = (a_error_1 + a_error_2 + a_error_3) / 3
    return (_a_error, _x_error, _y_error,
            [(a_error_1, x_error_1, y_error_1), (a_error_2, x_error_2, y_error_2), (a_error_3, x_error_3, y_error_3)],
            side_weight_shift_raw_data, front_weight_shift_raw_data, foot_slide_raw_data)


def compute_absolute_error(_x, _y):
    a = np.sqrt(np.pow(_x, 2) + np.pow(_y, 2))
    return a


def create_big_map(_conductor_heights, _conductor_widths, _pitch_heights, _pitch_widths, matrix):
    _conductor_widths = (1000 * _conductor_widths).astype(int)
    _conductor_heights = (1000 * _conductor_heights).astype(int)
    _pitch_widths = (1000 * _pitch_widths).astype(int)
    _pitch_heights = (1000 * _pitch_heights).astype(int)
    if not np.all(_pitch_heights == _pitch_heights[0]):
        _pitch_heights = np.hstack((_pitch_heights, _pitch_heights[0]))
    if not np.all(_pitch_widths == _pitch_widths[0]):
        _pitch_widths = np.hstack((_pitch_widths, _pitch_widths[0]))
    big_map = np.zeros((round(_conductor_heights.sum() + _pitch_widths.sum()),
                       round(_conductor_heights.sum() + _pitch_widths.sum())))

    for _i in range(len(_conductor_heights)):
        pitch_height_position = sum(_pitch_heights[0:_i + 1]) + sum(_conductor_heights[0:_i])
        for _j in range(len(_conductor_widths)):
            pitch_width_position = sum(_pitch_widths[0:_j+1]) + sum(_conductor_widths[0:_j])
            for _y in range(_conductor_heights[_i]):
                for _x in range(_conductor_widths[_j]):
                    big_map[_y + pitch_height_position][_x + pitch_width_position] = matrix[_i][_j]

    return big_map


def create_animated_plot(heatmaps):
    # Create real-time plot
    # Set up the figure and axis
    fig, ax = plt.subplots()
    heatmap_line = ax.imshow(heatmaps[0], cmap='viridis', interpolation='none')
    cbar = plt.colorbar(heatmap_line)

    ani = animation.FuncAnimation(fig, update_frame, frames=2 * np.shape(heatmaps)[0], interval=100, blit=True,
                                  fargs=(heatmap_line, heatmaps))

    plt.show()


def create_area_map(matrix, threshold=0.0):
    area_matrix = np.zeros(matrix.shape, dtype=np.uint8)
    for _i in range(area_matrix.shape[0]):
        for _j in range(area_matrix.shape[1]):
            if matrix[_i][_j] > threshold:
                area_matrix[_i][_j] = 1
            else:
                area_matrix[_i][_j] = 0
    return area_matrix


def plot_layouts(matrix_height, matrix_width, scale_factor,
                 plot_title_1, conductor_heights_1, conductor_widths_1, pitch_heights_1, pitch_widths_1,
                 plot_title_2, conductor_heights_2, conductor_widths_2, pitch_heights_2, pitch_widths_2):

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    create_plot(matrix_height, matrix_width, scale_factor, ax1,
                plot_title_1, conductor_heights_1, conductor_widths_1, pitch_heights_1, pitch_widths_1)
    plt.xlabel("Width (m)")
    plt.ylabel("Height (m)")

    ax2 = fig.add_subplot(1, 2, 2)
    create_plot(matrix_height, matrix_width, scale_factor, ax2,
                plot_title_2, conductor_heights_2, conductor_widths_2, pitch_heights_2, pitch_widths_2)

    plt.xlabel("Width (m)")
    plt.ylabel("Height (m)")
    plt.grid(False)
    plt.show()


def create_plot(matrix_height, matrix_width, scale_factor, axis, plot_title,
                _sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths):

    # Draw the matrix boundary
    matrix_rect = patches.Rectangle((0, 0), matrix_width / scale_factor, matrix_height / scale_factor,
                                    linewidth=1, edgecolor='black', facecolor='none')
    axis.add_patch(matrix_rect)
    track_x = -_sensor_widths[0]
    track_y = -_sensor_heights[0]
    # Draw each track as a rectangle centered on x_positions and y_positions
    for c_w, p_w in zip(_sensor_widths, _pitch_widths):
        # Calculate the bottom-left corner of each track
        track_x += (p_w + c_w) / scale_factor
        track_rect = patches.Rectangle((track_x, 0), c_w, matrix_height / scale_factor,
                                       linewidth=1, edgecolor="None", alpha=0.5, facecolor="orange")
        axis.add_patch(track_rect)

    for c_h, p_h in zip(_sensor_heights, _pitch_heights):
        # Calculate the bottom-left corner of each track
        track_y += (p_h + c_h) / scale_factor
        track_rect = patches.Rectangle((0, track_y), matrix_width / scale_factor, c_h,
                                       linewidth=1, edgecolor="None", alpha=0.5, facecolor="orange")
        axis.add_patch(track_rect)

    # Set axis limits and labels
    axis.set_xlim(-0.01, matrix_width / scale_factor + 0.01)
    axis.set_ylim(-0.01, matrix_height / scale_factor + 0.01)
    axis.set_aspect('equal')
    axis.set_title(plot_title)


def redistribute_y_pressure(matrix, cut_offs, mass):
    adjusted_matrix = matrix.copy()
    upper_bound = round(cut_offs[1] * adjusted_matrix.shape[0])
    lower_bound = round(cut_offs[0] * adjusted_matrix.shape[0])
    if lower_bound > 0:
        for _i in range(0, lower_bound, 1):
            for _j in range(0, adjusted_matrix.shape[1]):
                adjusted_matrix[_i][_j] = 0
    if upper_bound < adjusted_matrix.shape[0]:
        for _i in range(upper_bound, adjusted_matrix.shape[0], 1):
            for _j in range(0, adjusted_matrix.shape[1]):
                adjusted_matrix[_i][_j] = 0
    return rescale_mass(adjusted_matrix, mass)


def print_errors(_absolute_error, _x_error, _y_error, _scenario_errors):
    print("Side Weight Shift:  A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%"
          % (_scenario_errors[0][0], _scenario_errors[0][1], _scenario_errors[0][2]))
    print("Front Weight Shift: A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%"
          % (_scenario_errors[1][0], _scenario_errors[1][1], _scenario_errors[1][2]))
    print("Foot slides:        A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%"
          % (_scenario_errors[2][0], _scenario_errors[2][1], _scenario_errors[2][2]))
    print("Average:            A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%" % (_absolute_error, _x_error, _y_error))


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})  # Default font size for all elements
    np.random.seed(38)  # 9 26 38
    # Load the array back from the .npy file
    # Scale the force_map values to represent a realistic user weight.
    total_time = 5
    USER_MASS = 70
    gravity = 9.81

    # create heatmap with both feet
    high_res_resolution = (512, 512)
    mat_size = (0.48, 0.48)  # in metres
    left_foot_centre = (0.24, 0.12)  # in metres
    right_foot_centre = (0.24, 0.36)  # in metres

    SCALE_FACTOR = high_res_resolution[0] / mat_size[0] / 1000
    left_foot_centre = (round(left_foot_centre[0] * SCALE_FACTOR * 1000),
                        round(left_foot_centre[1] * SCALE_FACTOR * 1000))
    right_foot_centre = (round(right_foot_centre[0] * SCALE_FACTOR * 1000),
                         round(right_foot_centre[1] * SCALE_FACTOR * 1000))

    left_foot_profile = np.genfromtxt("./InputData/pressure_map.csv", delimiter=',',
                                      skip_header=0, filling_values=np.nan)
    left_foot_profile = zoom(left_foot_profile, zoom=(SCALE_FACTOR, SCALE_FACTOR))

    left_foot_profile = rescale_mass(left_foot_profile, USER_MASS / 2)

    right_foot_profile = np.flip(left_foot_profile, axis=1)

    base_case = move_feet(left_foot_centre, right_foot_centre,
                          left_foot_profile, right_foot_profile, high_res_resolution)
    # Sensor parameters
    R0 = 0.2325  # resistance per metre squared
    k = 1.265535e-8

    # Simulation Settings
    resolution = (8, 8)
    FORCE_RANDOM_OFFSET = 10000 * 64 / (resolution[0] * resolution[1])
    RANDOM_MAP = np.random.uniform(-FORCE_RANDOM_OFFSET, FORCE_RANDOM_OFFSET, size=resolution)
    rescaled_mat_size = (SCALE_FACTOR * mat_size[0], SCALE_FACTOR * mat_size[1])
    pitch_step_size = 2

    sensor_heights = np.array(resolution[0] * [rescaled_mat_size[0] / resolution[0] / 2])
    sensor_widths = np.array(resolution[1] * [rescaled_mat_size[1] / resolution[1] / 2])
    default_pitch_heights = np.array(resolution[0] * [(rescaled_mat_size[0] - sensor_heights.sum()) / resolution[0]])
    default_pitch_widths = np.array(resolution[1] * [(rescaled_mat_size[1] - sensor_widths.sum()) / resolution[1]])

    # Base result
    (absolute_error, x_error, y_error, scenario_errors,
     side_weight_shift_nonuniform_raw_data,
     front_weight_shift_nonuniform_raw_data,
     foot_slide_nonuniform_raw_data) = run_layout_scenarios(sensor_heights, sensor_widths,
                                                            default_pitch_heights, default_pitch_widths,
                                                            USER_MASS, left_foot_profile,
                                                            right_foot_profile, True,
                                                            RANDOM_MAP)

    print("Default Errors")
    print_errors(absolute_error, x_error, y_error, scenario_errors)

    minimum_pitch_height = (rescaled_mat_size[0] - sensor_heights.sum()) / resolution[0] / pitch_step_size
    minimum_pitch_width = (rescaled_mat_size[1] - sensor_widths.sum()) / resolution[1] / pitch_step_size
    track_height = float(sensor_heights[0])
    track_width = float(sensor_widths[0])

    x_min = track_width / 2
    x_max = rescaled_mat_size[1] - track_width / 2
    y_min = track_height / 2
    y_max = rescaled_mat_size[0] - track_height / 2

    # Generate possible discrete locations along an axis
    positions_y = np.arange(y_min, y_max + minimum_pitch_height, minimum_pitch_height)
    positions_x = np.arange(x_min, x_max + minimum_pitch_width, minimum_pitch_width)
    # Iterate over every combination of possible track positions
    valid_pitch_combinations = []
    x_errors = []
    valid_count = 1
    iterations = 0
    round_precision = 5
    # print(f"x combinations: {positions_x}\n f"y combinations: {positions_y}")
    total_x_combinations = math.comb(len(positions_x), resolution[1])
    total_y_combinations = math.comb(len(positions_y), resolution[0])
    total_combinations = total_x_combinations + total_y_combinations
    print("\nNumber of combinations:", total_combinations)
    for x_positions_numpy in itertools.combinations(positions_x, resolution[1]):
        total_width = 0
        iterations += 1
        x_positions = []
        # print(f"Iteration Number: {iterations}/{total_combinations}")
        for x in x_positions_numpy:
            x_positions.append(round(float(x), round_precision))
        pitch_widths = [round(x_positions[0] - track_width / 2, round_precision)]
        for j in range(resolution[1] - 1):
            pitch_widths.append(round(x_positions[j + 1] - x_positions[j] - track_width, round_precision))
        for j in range(0, resolution[1]):
            total_width += pitch_widths[j] + sensor_widths[j]
        if total_width <= rescaled_mat_size[1]:
            # Prevent tracks being next to each other
            if all(pitch_widths[n] > 0 for n in range(1, len(pitch_widths))):
                # Check symmetry
                symmetry_list = pitch_widths.copy()
                symmetry_list.append(round(rescaled_mat_size[1] - x_positions[-1] - x_min, round_precision))
                if symmetry_list == symmetry_list[::-1]:
                    # Valid combination
                    (absolute_error, x_error, y_error, scenario_errors,
                     side_weight_shift_nonuniform_raw_data,
                     front_weight_shift_nonuniform_raw_data,
                     foot_slide_nonuniform_raw_data) = run_layout_scenarios(sensor_heights,
                                                                            sensor_widths,
                                                                            sensor_heights,
                                                                            pitch_widths,
                                                                            USER_MASS,
                                                                            left_foot_profile,
                                                                            right_foot_profile,
                                                                            True,
                                                                            RANDOM_MAP)
                    valid_count += 1
                    valid_pitch_combinations.append([sensor_heights, pitch_widths])
                    x_errors.append(x_error)

    minimum_x_error = min(x_errors)
    minimum_error_index = x_errors.index(minimum_x_error)
    pitch_widths = valid_pitch_combinations[minimum_error_index][1]

    valid_errors = []
    valid_pitch_combinations = []
    combination_errors = []

    for y_positions_numpy in itertools.combinations(positions_y, resolution[0]):
        iterations += 1
        total_height = 0
        y_positions = []
        # print(f"Iteration Number: {iterations}/{total_combinations}")
        for y in y_positions_numpy:
            y_positions.append(round(float(y), round_precision))
        pitch_heights = [y_positions[0] - track_height / 2]
        for i in range(resolution[0] - 1):
            pitch_heights.append(round(y_positions[i + 1] - y_positions[i] - track_height, round_precision))
        # Calculate total width and height of the arrangement
        for i in range(0, resolution[0]):
            total_height += pitch_heights[i] + sensor_heights[i]
        # Check conditions
        if total_height <= rescaled_mat_size[0]:
            # Prevent tracks being next to each other
            if all(pitch_heights[n] > 0 for n in range(1, len(pitch_heights))):
                # Check symmetry
                symmetry_list = pitch_heights.copy()
                symmetry_list.append(round(rescaled_mat_size[0] - y_positions[-1] - y_min, round_precision))
                if symmetry_list == symmetry_list[::-1]:
                    (absolute_error, x_error, y_error, scenario_errors,
                     side_weight_shift_nonuniform_raw_data,
                     front_weight_shift_nonuniform_raw_data,
                     foot_slide_nonuniform_raw_data) = run_layout_scenarios(sensor_heights,
                                                                            sensor_widths,
                                                                            pitch_heights,
                                                                            pitch_widths,
                                                                            USER_MASS,
                                                                            left_foot_profile,
                                                                            right_foot_profile,
                                                                            True,
                                                                            RANDOM_MAP)
                    valid_errors.append([absolute_error, x_error, y_error, scenario_errors])
                    valid_pitch_combinations.append([pitch_heights, pitch_widths])
                    combination_errors.append(absolute_error)
                    valid_count += 1

    minimum_error = min(combination_errors)
    minimum_error_index = combination_errors.index(minimum_error)

    optimal_pitch_heights = valid_pitch_combinations[minimum_error_index][0]
    optimal_pitch_widths = valid_pitch_combinations[minimum_error_index][1]
    print(f"Produced {valid_count} valid combinations")
    print("\nOptimal Errors")
    print_errors(valid_errors[minimum_error_index][0], valid_errors[minimum_error_index][1],
                 valid_errors[minimum_error_index][2], valid_errors[minimum_error_index][3])
    print(f"\nOptimal Heights: {valid_pitch_combinations[minimum_error_index][0]}\n"
          f"Optimal Widths: {valid_pitch_combinations[minimum_error_index][1]}")

    plot_layouts(rescaled_mat_size[0], rescaled_mat_size[1], SCALE_FACTOR,
                 "Default Track Geometry", sensor_heights, sensor_widths,
                 default_pitch_heights, default_pitch_widths,
                 "An Optimal Track Geometry", sensor_heights, sensor_widths,
                 optimal_pitch_heights, optimal_pitch_widths)
