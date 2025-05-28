import numpy as np
import matplotlib.pyplot as plt
import csv

def read_csv_to_array(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [list(map(float, row)) for row in reader]
    return np.array(data)

def csv_to_image(data, output_file='output_image.png', base_cmap='viridis'):
    # Create a mask for zero values
    mask = (data == 0)

    # Use masked array to separate zeros from other values
    masked_data = np.ma.masked_where(mask, data)

    # Get base colormap and create a new colormap with black as the "bad" (masked) value
    cmap = plt.get_cmap(base_cmap)
    cmap.set_bad(color='black')
    
    plt.imshow(masked_data, cmap=cmap)
    plt.colorbar()  # Optional: shows color scale
    plt.axis('off')  # Hides axes
    plt.show()
    # plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    # plt.close()
    print(f"Image saved as '{output_file}'")

if __name__ == "__main__":
    input_csv = 'Orientation_angles_72hrs_compact_S1_16x_16x4_tilescan.csv'  # Replace with your filename
    data_array = read_csv_to_array(input_csv)
    csv_to_image(data_array, base_cmap='hsv')