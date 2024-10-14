import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from matplotlib.colors import hex2color

# Define the known colors and pressure values
color_map = {
    '#000000': 0,
    '#333333': 10,
    '#666666': 20,
    '#999999': 30,
    '#cccccc': 40,
    '#FFFFFF': 50
}

# Convert hex color codes to RGB values
color_rgb = np.array([hex2color(k) for k in color_map.keys()]) * 255  # Convert to 0-255 range
pressures = np.array(list(color_map.values()))  # Corresponding pressures

# Create an interpolation function for each color channel (R, G, B)
r_interp = interp1d(color_rgb[:, 0], pressures, bounds_error=False, fill_value="extrapolate")
g_interp = interp1d(color_rgb[:, 1], pressures, bounds_error=False, fill_value="extrapolate")
b_interp = interp1d(color_rgb[:, 2], pressures, bounds_error=False, fill_value="extrapolate")


# Function to map an RGB color to a pressure value using interpolation
def get_pressure_from_color(color):
    r, g, b = color
    pressure_r = r_interp(r)
    pressure_g = g_interp(g)
    pressure_b = b_interp(b)
    # Take the average of interpolated pressure values across R, G, and B channels
    return (pressure_r + pressure_g + pressure_b) / 3.0


if __name__ == "__main__":
    # Load the heatmap image
    image_path = 'foot_pressure_4.png'
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure image is in RGB format

    # Convert the image to a numpy array of RGB values
    image_data = np.array(image)

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

    # Dimensions of the image
    height, width = pressure_map.shape

    # Create coordinate grids for x and y
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Compute the total pressure (sum of all pixel pressures)
    total_pressure = np.sum(pressure_map)

    # Compute the weighted sum for the x and y coordinates
    x_COP = np.sum(x_coords * pressure_map) / total_pressure
    y_COP = np.sum(y_coords * pressure_map) / total_pressure

    # Print the computed center of pressure
    print(f"Center of Pressure (x, y): ({x_COP}, {y_COP})")

    # Create a heatmap
    plt.imshow(pressure_map, cmap='jet', origin='upper', interpolation='nearest')

    # Add a colorbar to represent the pressure levels
    plt.colorbar(label='Pressure (units)')

    # Add a red dot to indicate the Center of Pressure (CoP)
    plt.scatter([x_COP], [y_COP], color='red', marker='o', label='Center of Pressure')

    # Set axis labels
    plt.xlabel('Width (pixels)')
    plt.ylabel('Height (pixels)')

    # Display the heatmap
    plt.title('Pressure Heatmap from Foot Pressure Image')
    plt.show()
