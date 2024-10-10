from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.colors import hex2color

# Load the heatmap image
image_path = 'foot_pressure.png'
image = Image.open(image_path)
image = image.convert('RGB')  # Ensure image is in RGB format

# Define the known colors and pressure values
color_map = {
    '#010164': 0,
    '#3f48cc': 10,
    '#00a8f3': 20,
    '#8cfffb': 30,
    '#c4ff0e': 40,
    '#ffca18': 50,
    '#ff7f27': 60,
    '#ec1c24': 70
}

# Convert hex color codes to RGB values
color_rgb = {hex2color(k): v for k, v in color_map.items()}

# Create arrays for interpolation
colors = np.array(list(color_rgb.keys()))  # Array of RGB values
pressures = np.array(list(color_rgb.values()))  # Corresponding pressures

# Build a k-D tree for efficient nearest-neighbor lookup
tree = cKDTree(colors)

# Convert the image to a numpy array of RGB values
image_data = np.array(image)


# Function to map an RGB color to a pressure value using nearest neighbor interpolation
def get_pressure_from_color(color):
    distance, idx = tree.query(color / 255.0)  # Normalize color to [0,1] range
    return pressures[idx]


# Sample the image and get pressure values
height, width, _ = image_data.shape
pressure_map = np.zeros((height, width))

for i in range(height):
    for j in range(width):
        rgb = image_data[i, j]
        pressure = get_pressure_from_color(rgb)
        pressure_map[i, j] = pressure

# Optionally, you can save the pressure map or analyze it further
# Example: Save as a CSV file
np.savetxt("pressure_map.csv", pressure_map, delimiter=",")

# Assuming `pressure_map` is the 2D array of pressure values obtained from the previous code

# Create a heatmap
plt.imshow(pressure_map, cmap='jet', origin='upper', interpolation='nearest')

# Add a colorbar to represent the pressure levels
plt.colorbar(label='Pressure (units)')

# Set axis labels
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')

# Display the heatmap
plt.title('Pressure Heatmap from Foot Pressure Image')
plt.show()
