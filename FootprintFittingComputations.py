import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
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
    x = np.sum(x_coords * force_map) / total_pressure
    y = np.sum(y_coords * force_map) / total_pressure

    return x, y


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
    for i in range(0, resolution[0]):
        width_midpoint = _conductor_widths[0] / 2
        for j in range(0, resolution[1]):
            low_res_pressure_map[i][j] = sum_square_section(high_res_heatmap_matrix,
                                                            (height_midpoint, width_midpoint),
                                                            _conductor_widths[j], _conductor_heights[i])
            width_midpoint += _conductor_widths[j - 1] / 2 + _pitch_widths[j] + _conductor_widths[j] / 2
        height_midpoint += _conductor_heights[i - 1] / 2 + _pitch_heights[i] + _conductor_heights[i] / 2

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


def create_big_map(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths, matrix):
    _sensor_widths = (1000 * _sensor_widths).astype(int)
    _sensor_heights = (1000 * _sensor_heights).astype(int)
    _pitch_widths = (1000 * _pitch_widths).astype(int)
    _pitch_heights = (1000 * _pitch_heights).astype(int)
    if not np.all(_pitch_heights == _pitch_heights[0]):
        _pitch_heights = np.hstack((_pitch_heights, _pitch_heights[0]))
    if not np.all(_pitch_widths == _pitch_widths[0]):
        _pitch_widths = np.hstack((_pitch_widths, _pitch_widths[0]))
    big_map = np.zeros((round(_sensor_heights.sum() + _pitch_widths.sum()),
                       round(_sensor_heights.sum() + _pitch_widths.sum())))

    for _i in range(len(_sensor_heights)):
        pitch_height_position = sum(_pitch_heights[0:_i + 1]) + sum(_sensor_heights[0:_i])
        for _j in range(len(_sensor_widths)):
            pitch_width_position = sum(_pitch_widths[0:_j+1]) + sum(_sensor_widths[0:_j])
            for _y in range(_sensor_heights[_i]):
                for _x in range(_sensor_widths[_j]):
                    big_map[_y + pitch_height_position][_x + pitch_width_position] = matrix[_i][_j]

    return big_map


def run_footprint_placement_scenarios(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                                      _left_foot_profile, _right_foot_profile, piezo):
    if np.all(_pitch_heights == _pitch_heights[0]):
        first_pitch_height = _pitch_heights[0]
    else:
        first_pitch_height = 0
    if np.all(_pitch_widths == _pitch_widths[0]):
        first_pitch_width = _pitch_widths[0]
    else:
        first_pitch_width = 0
    time_step = 0.1  # Seconds
    time_steps = np.arange(0, total_time + time_step, time_step)
    number_of_time_stamps = len(time_steps)
    foot_height, foot_width = _left_foot_profile.shape
    left_foot_start = round(foot_width / 2)
    right_foot_end = (1000 * rescaled_mat_size[1]) - round(foot_width / 2)
    _left_foot_centre = (240, 150)  # in mm
    _right_foot_centre = (240, 330)  # in mm
    _left_foot_centre = (round(_left_foot_centre[0] * SCALE_FACTOR), round(_left_foot_centre[1] * SCALE_FACTOR))
    _right_foot_centre = (round(_right_foot_centre[0] * SCALE_FACTOR), round(_right_foot_centre[1] * SCALE_FACTOR))
    left_foot_gradient = 2 * (_left_foot_centre[1] - left_foot_start) / total_time
    right_foot_gradient = 2 * (right_foot_end - _right_foot_centre[1]) / total_time
    animation_matrices = []
    fs_average_x_e = 0
    fs_average_y_e = 0
    fs_average_a_e = 0

    cop_data_sw = np.zeros((number_of_time_stamps, 2))
    cop_data_fw = np.zeros((number_of_time_stamps, 2))
    cop_data_fs = np.zeros((number_of_time_stamps, 2))
    raw_data_sw = np.zeros((number_of_time_stamps, 4))
    raw_data_fw = np.zeros((number_of_time_stamps, 4))
    raw_data_fs = np.zeros((number_of_time_stamps, 4))

    # sliding foot
    for t in time_steps:
        if t < total_time / 2:
            left_foot_position = (_left_foot_centre[0], float(left_foot_gradient * t + left_foot_start))
            right_foot_position = _right_foot_centre
        else:
            left_foot_position = _left_foot_centre
            right_foot_position = (_right_foot_centre[0], float(right_foot_gradient * t
                                   + 2 * _right_foot_centre[1] - right_foot_end))

        high_res_matrix = move_feet(left_foot_position, right_foot_position,
                                    _left_foot_profile, _right_foot_profile, high_res_resolution)
        real_x, real_y = centre_of_pressure(high_res_matrix)

        _, _, _, low_res_matrix, _ = compute_error_for_instance(_sensor_heights, _sensor_widths, _pitch_heights,
                                                                _pitch_widths, high_res_matrix, piezo, RANDOM_MAP)

        resized_low_res_matrix = create_big_map(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                                                low_res_matrix)

        left_half = resized_low_res_matrix.copy()
        right_half = resized_low_res_matrix.copy()
        left_half[:, round(resized_low_res_matrix.shape[1] / 2):] = 0
        right_half[:, :round(resized_low_res_matrix.shape[1] / 2)] = 0
        # extra 0 columns
        extra_columns = np.zeros((resized_low_res_matrix.shape[0], left_foot_profile.shape[1]))
        left_half = np.hstack((extra_columns, left_half, extra_columns))
        right_half = np.hstack((extra_columns, right_half, extra_columns))
        buffer_columns = extra_columns.shape[1]

        best_location_left = fit_profile(left_half.copy(), _left_foot_profile, buffer_columns,
                                         first_pitch_width, first_pitch_height,
                                         left_foot_position[1] + buffer_columns,
                                         left_foot_position[0])
        best_location_right = fit_profile(right_half.copy(), _right_foot_profile, buffer_columns,
                                          first_pitch_width, first_pitch_height,
                                          right_foot_position[1] + buffer_columns,
                                          right_foot_position[0])
        new_high_resolution = (high_res_resolution[0] + 2 * buffer_columns, high_res_resolution[1] + 2 * buffer_columns)
        estimated_matrix = move_feet((best_location_left[0] + buffer_columns,
                                      best_location_left[1] + buffer_columns),
                                     (best_location_right[0] + buffer_columns,
                                      best_location_right[1] + buffer_columns),
                                     _left_foot_profile, _right_foot_profile, new_high_resolution)
        animation_matrices.append(estimated_matrix)
        estimated_x, estimated_y = centre_of_pressure(estimated_matrix)
        estimated_x -= buffer_columns
        estimated_y -= buffer_columns
        _x_e = 100 * abs((real_x - estimated_x) / real_x)
        _y_e = 100 * abs((real_y - estimated_y) / real_y)
        _a_e = compute_absolute_error(_x_e, _y_e)
        print("Instance Error: A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%" % (_a_e, _x_e, _y_e))

        fs_average_x_e += _x_e
        fs_average_y_e += _y_e
        fs_average_a_e += _a_e
        print("Time step: %3.1f/%3.1f" % (t, total_time))

        cop_data_fs[np.where(time_steps == t)[0][0]][0] = _x_e
        cop_data_fs[np.where(time_steps == t)[0][0]][1] = _y_e

        raw_results = [real_x, real_y, estimated_x, estimated_y]
        for _i in range(len(raw_results)):
            raw_data_fs[np.where(time_steps == t)[0][0]][_i] = raw_results[_i]
    np.savetxt('./OutputData/FittedFootSlides.csv', raw_data_fs, delimiter=',')

    fs_average_x_e /= number_of_time_stamps
    fs_average_y_e /= number_of_time_stamps
    fs_average_a_e /= number_of_time_stamps
    print("Sliding Foot: Average Error: A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%" %
          (fs_average_a_e, fs_average_x_e, fs_average_y_e))

    # front weight shift:
    fw_average_x_e = 0
    fw_average_y_e = 0
    fw_average_a_e = 0

    _left_foot_centre = (left_foot_centre[0], left_foot_centre[1])
    _right_foot_centre = (right_foot_centre[0], right_foot_centre[1])
    for t in time_steps:
        if t <= total_time / 2:
            bottom_cut_off = 0
            top_cut_off = 4 * t / (3 * total_time) + 1 / 3
        else:
            bottom_cut_off = 4 / (3 * total_time) * t - 2 / 3
            top_cut_off = 1
        left_foot = redistribute_y_pressure(left_foot_profile,
                                            (bottom_cut_off, top_cut_off), USER_MASS / 2)
        right_foot = redistribute_y_pressure(right_foot_profile,
                                             (bottom_cut_off, top_cut_off), USER_MASS / 2)

        high_res_matrix = move_feet(_left_foot_centre, _right_foot_centre,
                                    left_foot, right_foot, high_res_resolution)

        real_x, real_y = centre_of_pressure(high_res_matrix)

        _, _, _, low_res_matrix, _ = compute_error_for_instance(_sensor_heights, _sensor_widths, _pitch_heights,
                                                                _pitch_widths, high_res_matrix, piezo, RANDOM_MAP)
        resized_low_res_matrix = create_big_map(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                                                low_res_matrix)

        left_half = resized_low_res_matrix.copy()
        right_half = resized_low_res_matrix.copy()
        left_half[:, round(resized_low_res_matrix.shape[1] / 2):] = 0
        right_half[:, :round(resized_low_res_matrix.shape[1] / 2)] = 0
        # extra 0 columns
        extra_columns = np.zeros((resized_low_res_matrix.shape[0], left_foot_profile.shape[1]))
        left_half = np.hstack((extra_columns, left_half, extra_columns))
        right_half = np.hstack((extra_columns, right_half, extra_columns))
        buffer_columns = extra_columns.shape[1]

        best_location_left = fit_profile(left_half.copy(), left_foot, buffer_columns,
                                         first_pitch_width, first_pitch_height, _left_foot_centre[1] + buffer_columns,
                                         _left_foot_centre[0])
        best_location_right = fit_profile(right_half.copy(), right_foot, buffer_columns,
                                          first_pitch_width, first_pitch_height, _right_foot_centre[1] + buffer_columns,
                                          _right_foot_centre[0])
        new_high_resolution = (high_res_resolution[0] + 2 * buffer_columns, high_res_resolution[1] + 2 * buffer_columns)
        estimated_matrix = move_feet((best_location_left[0] + buffer_columns,
                                      best_location_left[1] + buffer_columns),
                                     (best_location_right[0] + buffer_columns,
                                      best_location_right[1] + buffer_columns),
                                     left_foot, right_foot, new_high_resolution)
        animation_matrices.append(estimated_matrix)
        estimated_x, estimated_y = centre_of_pressure(estimated_matrix)
        estimated_x -= buffer_columns
        estimated_y -= buffer_columns
        _x_e = 100 * abs((real_x - estimated_x) / real_x)
        _y_e = 100 * abs((real_y - estimated_y) / real_y)
        _a_e = compute_absolute_error(_x_e, _y_e)
        print("Instance Error: A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%" % (_a_e, _x_e, _y_e))

        fw_average_x_e += _x_e
        fw_average_y_e += _y_e
        fw_average_a_e += _a_e
        print("Time step: %3.1f/%3.1f" % (t, total_time))

        cop_data_fw[np.where(time_steps == t)[0][0]][0] = _x_e
        cop_data_fw[np.where(time_steps == t)[0][0]][1] = _y_e
        raw_results = [real_x, real_y, estimated_x, estimated_y]
        for _i in range(len(raw_results)):
            raw_data_fw[np.where(time_steps == t)[0][0]][_i] = raw_results[_i]
    np.savetxt('./OutputData/FittedFrontWeightShift.csv', raw_data_fw, delimiter=',')

    fw_average_x_e /= number_of_time_stamps
    fw_average_y_e /= number_of_time_stamps
    fw_average_a_e /= number_of_time_stamps
    print("Front Weight Shift: Average Errors: A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%" %
          (fw_average_a_e, fw_average_x_e, fw_average_y_e))

    # Side weight shift
    _left_foot_centre = left_foot_centre
    _right_foot_centre = right_foot_centre
    sw_average_x_e = 0
    sw_average_y_e = 0
    sw_average_a_e = 0
    for t in time_steps:
        left_foot_mass = USER_MASS / total_time * (total_time - t)
        right_foot_mass = USER_MASS - left_foot_mass
        temp_left_foot_profile = rescale_mass(left_foot_profile, left_foot_mass)
        temp_right_foot_profile = rescale_mass(right_foot_profile, right_foot_mass)
        high_res_matrix = move_feet(left_foot_centre, right_foot_centre,
                                    temp_left_foot_profile, temp_right_foot_profile, high_res_resolution)

        real_x, real_y = centre_of_pressure(high_res_matrix)
        _, _, _, low_res_matrix, _ = compute_error_for_instance(_sensor_heights, _sensor_widths, _pitch_heights,
                                                                _pitch_widths, high_res_matrix, piezo, RANDOM_MAP)
        resized_low_res_matrix = create_big_map(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths,
                                                low_res_matrix)

        left_half = resized_low_res_matrix.copy()
        right_half = resized_low_res_matrix.copy()
        left_half[:, round(resized_low_res_matrix.shape[1] / 2):] = 0
        right_half[:, :round(resized_low_res_matrix.shape[1] / 2)] = 0
        # extra 0 columns
        extra_columns = np.zeros((resized_low_res_matrix.shape[0], left_foot_profile.shape[1]))
        left_half = np.hstack((extra_columns, left_half, extra_columns))
        right_half = np.hstack((extra_columns, right_half, extra_columns))
        buffer_columns = extra_columns.shape[1]

        best_location_left = fit_profile(left_half.copy(), temp_left_foot_profile, buffer_columns,
                                         first_pitch_width, first_pitch_height,
                                         _left_foot_centre[1] + buffer_columns,
                                         _left_foot_centre[0])
        best_location_right = fit_profile(right_half.copy(), temp_right_foot_profile, buffer_columns,
                                          first_pitch_width, first_pitch_height,
                                          _right_foot_centre[1] + buffer_columns,
                                          _right_foot_centre[0])
        new_high_resolution = (high_res_resolution[0] + 2 * buffer_columns, high_res_resolution[1] + 2 * buffer_columns)
        estimated_matrix = move_feet((best_location_left[0] + buffer_columns,
                                      best_location_left[1] + buffer_columns),
                                     (best_location_right[0] + buffer_columns,
                                      best_location_right[1] + buffer_columns),
                                     temp_left_foot_profile, temp_right_foot_profile, new_high_resolution)

        animation_matrices.append(estimated_matrix)
        estimated_x, estimated_y = centre_of_pressure(estimated_matrix)
        estimated_x -= buffer_columns
        estimated_y -= buffer_columns
        _x_e = 100 * abs((real_x - estimated_x) / real_x)
        _y_e = 100 * abs((real_y - estimated_y) / real_y)
        _a_e = compute_absolute_error(_x_e, _y_e)
        print("Instance Error: A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%" % (_a_e, _x_e, _y_e))

        sw_average_x_e += _x_e
        sw_average_y_e += _y_e
        sw_average_a_e += _a_e
        print("Time step: %3.1f/%3.1f" % (t, total_time))

        cop_data_sw[np.where(time_steps == t)[0][0]][0] = _x_e
        cop_data_sw[np.where(time_steps == t)[0][0]][1] = _y_e
        raw_results = [real_x, real_y, estimated_x, estimated_y]
        for _i in range(len(raw_results)):
            raw_data_sw[np.where(time_steps == t)[0][0]][_i] = raw_results[_i]

    np.savetxt('./OutputData/FittedSideWeightShift.csv', raw_data_sw, delimiter=',')

    sw_average_a_e /= number_of_time_stamps
    sw_average_x_e /= number_of_time_stamps
    sw_average_y_e /= number_of_time_stamps
    print("Side Weight Shift: Average Errors: A: %5.2f%%, X: %5.2f%%, Y: %5.2f%%" %
          (sw_average_a_e, sw_average_x_e, sw_average_y_e))

    average_a_e = [sw_average_a_e, fw_average_a_e, fs_average_a_e]
    average_x_e = [sw_average_x_e, fw_average_x_e, fs_average_x_e]
    average_y_e = [sw_average_y_e, fw_average_y_e, fs_average_y_e]
    _scenario_errors = [average_a_e, average_x_e, average_y_e]
    return sum(average_a_e) / 3, sum(average_x_e) / 3, sum(average_y_e) / 3, _scenario_errors, animation_matrices


def create_animated_plot(heatmaps):
    # Create real-time plot
    # Set up the figure and axis
    fig, ax = plt.subplots()
    heatmap_line = ax.imshow(heatmaps[0], cmap='viridis', interpolation='none')
    cbar = plt.colorbar(heatmap_line)

    ani = animation.FuncAnimation(fig, update_frame, frames=2 * np.shape(heatmaps)[0], interval=100, blit=True,
                                  fargs=(heatmap_line, heatmaps))

    plt.show()


def subtract_matrices(big_matrix, small_matrix, start_row, start_col):
    end_row = start_row + small_matrix.shape[0]
    end_col = start_col + small_matrix.shape[1]
    big_matrix[start_row:end_row, start_col:end_col] -= small_matrix
    return big_matrix


def fit_profile(matrix, profile, buffer_columns, first_pitch_width=0, first_pitch_height=0,
                centre_x=None, centre_y=None):
    list_of_total_pressures = []
    list_of_locations = []
    if centre_x is None:
        centre_x, _ = centre_of_pressure(matrix)
    if centre_y is None:
        _, centre_y = centre_of_pressure(matrix)
    if np.isnan(centre_x):
        centre_x = matrix.shape[1] // 2
    if np.isnan(centre_y):
        centre_y = matrix.shape[0] // 2
    centre_x = round(centre_x)
    centre_y = round(centre_y)
    top_left_x = centre_x - profile.shape[1] // 2
    top_left_y = centre_y - profile.shape[0] // 2
    x_edge = matrix.shape[1] - profile.shape[1]
    y_edge = matrix.shape[0] - profile.shape[0]
    radius = TEST_RADIUS
    x_search_lower = top_left_x - radius
    y_search_lower = top_left_y - radius
    x_search_upper = top_left_x + radius
    y_search_upper = top_left_y + radius
    if x_search_lower < 0:
        x_search_lower = 0
    if x_search_lower > x_edge:
        x_search_lower = x_edge
    if y_search_lower > y_edge:
        y_search_lower = y_edge
    if y_search_lower < 0:
        y_search_lower = 0
    if x_search_upper > x_edge:
        x_search_upper = x_edge
    if y_search_upper > y_edge:
        y_search_upper = y_edge
    for _i in range(y_search_lower, y_search_upper + 1, 1):
        for _j in range(x_search_lower, x_search_upper + 1, 1):
            subtracted_matrix = subtract_matrices(matrix.copy(), profile.copy(), _i, _j)
            list_of_total_pressures.append(np.sum(subtracted_matrix ** 2))
            list_of_locations.append((_i + profile.shape[0] // 2, _j + profile.shape[1] // 2))
    minimum_area = min(list_of_total_pressures)
    best_location = list_of_locations[list_of_total_pressures.index(minimum_area)]
    print("Movement within search radius:", centre_x - best_location[1], centre_y - best_location[0])
    if minimum_area <= 0:
        if centre_x is not None and centre_y is not None:
            best_location = (centre_y, centre_x)
        else:
            best_location = (matrix.shape[1] // 2, matrix.shape[0] // 2)
    adjusted_best_location = (best_location[0] - 1000 * first_pitch_width,
                              best_location[1] - buffer_columns - 1000 * first_pitch_height)
    return adjusted_best_location


def create_area_map(matrix, threshold=0.0):
    area_matrix = np.zeros(matrix.shape, dtype=np.uint8)
    for i in range(area_matrix.shape[0]):
        for j in range(area_matrix.shape[1]):
            if matrix[i][j] > threshold:
                area_matrix[i][j] = 1
            else:
                area_matrix[i][j] = 0
    return area_matrix


def plot_track_layout(_sensor_heights, _sensor_widths, _pitch_heights, _pitch_widths, _matrix_height, _matrix_width,
                      _scale_factor, _plot_title):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the matrix boundary
    matrix_rect = patches.Rectangle((0, 0), _matrix_width / _scale_factor, _matrix_height / _scale_factor,
                                    linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(matrix_rect)
    track_x = -_sensor_widths[0]
    track_y = -_sensor_heights[0]
    # Draw each track as a rectangle centered on x_positions and y_positions
    for c_w, p_w in zip(_sensor_widths, _pitch_widths):
        # Calculate the bottom-left corner of each track
        track_x += (p_w + c_w) / _scale_factor
        track_rect = patches.Rectangle((track_x, 0), c_w, _matrix_height / _scale_factor,
                                       linewidth=1, edgecolor="None", alpha=0.5, facecolor="orange")
        ax.add_patch(track_rect)

    for c_h, p_h in zip(_sensor_heights, _pitch_heights):
        # Calculate the bottom-left corner of each track
        track_y += (p_h + c_h) / _scale_factor
        track_rect = patches.Rectangle((0, track_y), _matrix_width / _scale_factor, c_h,
                                       linewidth=1, edgecolor="None", alpha=0.5, facecolor="orange")
        ax.add_patch(track_rect)

    # Set axis limits and labels
    ax.set_xlim(-0.01, _matrix_width / _scale_factor + 0.01)
    ax.set_ylim(-0.01, _matrix_height / _scale_factor + 0.01)
    ax.set_aspect('equal')
    ax.set_title(_plot_title)
    plt.xlabel("Width (m)")
    plt.ylabel("Height (m)")
    plt.grid(False)

    plt.show()


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
    TEST_RADIUS = 60  # this controls the size of the radius in which the program fits the known footprint.
    # The larger the number, the longer the optimisation takes, but the more accurate the result.
    resolution = (8, 8)
    FORCE_RANDOM_OFFSET = 10000 * 64 / (resolution[0] * resolution[1])
    RANDOM_MAP = np.random.uniform(-FORCE_RANDOM_OFFSET, FORCE_RANDOM_OFFSET, size=resolution)
    rescaled_mat_size = (SCALE_FACTOR * mat_size[0], SCALE_FACTOR * mat_size[1])
    pitch_step_size = 2

    sensor_heights = np.array(resolution[0] * [rescaled_mat_size[0] / resolution[0] / 2])
    sensor_widths = np.array(resolution[1] * [rescaled_mat_size[1] / resolution[1] / 2])
    pitch_heights = np.array(resolution[0] * [(rescaled_mat_size[0] - sensor_heights.sum()) / resolution[0]])
    pitch_widths = np.array(resolution[1] * [(rescaled_mat_size[1] - sensor_widths.sum()) / resolution[1]])

    # Base result
    (base_absolute_error, base_x_error, base_y_error, base_scenario_errors,
     side_weight_shift_nonuniform_raw_data,
     front_weight_shift_nonuniform_raw_data,
     foot_slide_nonuniform_raw_data) = run_layout_scenarios(sensor_heights, sensor_widths,
                                                            pitch_heights, pitch_widths,
                                                            USER_MASS, left_foot_profile,
                                                            right_foot_profile, True,
                                                            RANDOM_MAP)

    (a_e, x_e, y_e, scenario_errors,
     animation_frames) = run_footprint_placement_scenarios(sensor_heights, sensor_widths,
                                                           pitch_heights, pitch_widths,
                                                           left_foot_profile,
                                                           right_foot_profile, True)
    print("\nDefault Errors")
    print_errors(base_absolute_error, base_x_error, base_y_error, base_scenario_errors)
    print("\nFootprint Fitting Optimisation Errors")
    print_errors(a_e, x_e, y_e, scenario_errors)
    plot_track_layout(sensor_heights, sensor_widths, pitch_heights, pitch_widths,
                      rescaled_mat_size[0], rescaled_mat_size[1], SCALE_FACTOR, "Default Track Geometry")

    create_animated_plot(animation_frames)
