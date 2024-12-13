import re
import numpy as np

# Input text
file_paths = ["fs_data.txt", "fw_data.txt", "sw_data.txt"]  # Replace with your file's path
files = []
for file_path in file_paths:
    with open(file_path, "r") as file:
        data = file.read()
        files.append(data)

# Split the data into lines
lines_of_lines = []
for data in files:
    file_data = data.strip().split("\n")
    lines_of_lines.append(file_data.copy())

# Initialize variables
arrays = []
current_arrays = []

# Process each line
for lines in lines_of_lines:
    for line in lines:
        if "Movement within search radius" in line:
            # Extract the coordinates using regex
            match = re.search(r"Movement within search radius: (-?\d+) (-?\d+)", line)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                if len(current_arrays) < 2:
                    current_arrays.append([[x, y]])
                else:
                    current_arrays[0].append([x, y])
                    current_arrays[1].append([x, y])
        elif "Time step" in line:
            # Append current arrays to the main list and reset for the next timestep
            if current_arrays:
                for i, arr in enumerate(current_arrays):
                    if len(arrays) <= i:
                        arrays.append(arr)
                    else:
                        arrays[i].extend(arr)
                current_arrays = []

# Append any remaining data
if current_arrays:
    for i, arr in enumerate(current_arrays):
        if len(arrays) <= i:
            arrays.append(arr)
        else:
            arrays[i].extend(arr)


if __name__ == "__main__":
    # Result: Two 2D arrays
    left_foot = arrays[0] if len(arrays) > 0 else []
    right_foot = arrays[1] if len(arrays) > 1 else []

    segment = len(left_foot) // 3
    fs_left = left_foot[0:segment]
    fs_right = right_foot[0:segment]
    fw_left = left_foot[segment:2*segment]
    fw_right = right_foot[segment:2*segment]
    sw_left = left_foot[2*segment:]
    sw_right = right_foot[2*segment:]
    # Print the results
    '''
    print("\nFoot Slides")
    print(fs_left)
    print(fs_right)
    print("\nFront Weight Shift")
    print(fw_left)
    print(fw_right)
    print("\nSide Weight Shift")
    print(sw_left)
    print(sw_right)
    '''
    np.save("./Data/fs_left.npy", np.array(fs_left)[:, ::-1])
    np.save("./Data/fs_right.npy", np.array(fs_right)[:, ::-1])
    np.save("./Data/fw_left.npy", np.array(fw_left)[:, ::-1])
    np.save("./Data/fw_right.npy", np.array(fw_right)[:, ::-1])
    np.save("./Data/sw_left.npy", np.array(sw_left)[:, ::-1])
    np.save("./Data/sw_right.npy", np.array(sw_right)[:, ::-1])
