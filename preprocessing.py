import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def interpolate_trajectory_by_distance(trajectory, num_points=50):
    # Calculate cumulative distance
    distances = np.sqrt(np.sum(np.diff(trajectory, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    # Resample based on cumulative distance
    total_distance = cumulative_distances[-1]
    new_distances = np.linspace(0, total_distance, num_points)
    interpolated = interp1d(cumulative_distances, trajectory, axis=0, kind='linear')(new_distances)
    interpolated = pd.DataFrame(interpolated)
    interpolated.columns = ['x', 'y', 'z']
    return interpolated

def normalisation_1 (data):
    data = data.dropna()
    data1 = data - data.mean()
    data1 = data1.to_numpy()
    max_distance = np.max(np.sqrt(np.sum(data1**2, axis=-1)))
    data1 = data1/(max_distance + 1e-8)
    return pd.DataFrame(data1, columns = ['x', 'y', 'z'])

def smoothing (data, type='savgol_filter', window_length = 4):
    if type == 'savgol_filter':
        data['x'] = savgol_filter(data['x'], window_length, polyorder=2)
        data['y'] = savgol_filter(data['y'], window_length, polyorder=2)
        data['z'] = savgol_filter(data['z'], window_length, polyorder=2)
    elif type == 'rolling':
        data['x'] = data['x'].rolling(window=window_length, center=True).mean()
        data['y'] = data['y'].rolling(window=window_length, center=True).mean()
        data['z'] = data['z'].rolling(window=window_length, center=True).mean()
    return data

def transform (data):
    data = interpolate_trajectory_by_distance(data, 30)
    data = smoothing(data, 'savgol_filter', 5) 
    data = smoothing(data, 'rolling', 3)
    data = normalisation_1(data)
    data = smoothing(data, 'savgol_filter', 6) 
    data = interpolate_trajectory_by_distance(data, 300)
    return data 

def process_and_stack(data):
    """
    Processes the data for each (label, obs) group, applies the processing 
    function, and stacks the results into a new DataFrame.
    """
    # List to store processed results
    results = []

    # Group by label and obs
    grouped = data.groupby(['label', 'obs'])

    for (label, obs), group in grouped:
        # Extract x, y, z coordinates
        xyz = group[['x', 'y', 'z']]

        # Apply the processing function to xyz coordinates
        processed_xyz = transform(xyz)

        # Create a new DataFrame for processed points
        processed_df = pd.DataFrame(processed_xyz, columns=['x', 'y', 'z'])
        processed_df['obs'] = obs  # Add observation column
        processed_df['label'] = label  # Add label column

        # Append to results
        results.append(processed_df)
    final_df = pd.concat(results, ignore_index=True)
    return final_df

file_names = {}
for stroke_num in range(10):
    # Generate file names for each stroke
    file_names[stroke_num] = [f"training_data/stroke_{stroke_num}_{str(i).zfill(4)}.csv" for i in range(1, 101)]

# for vertically stacking the data
training_data = pd.DataFrame()
for digit in range(10):
    stroke_0_files = file_names[digit]
    data_0 = stroke_0_files[0]
    data_0 = pd.read_csv(data_0, header = None) 
    data_0.columns = ['x', 'y', 'z']
    data_0['obs'] = 1
    data_0['label'] = digit
    for i in range(1, len(stroke_0_files)):
        data_c = pd.read_csv(stroke_0_files[i], header = None)
        data_c.columns = ['x', 'y', 'z']
        data_c['obs'] = i + 1
        data_c['label'] = digit
        data_0 = pd.concat([data_0, data_c], axis= 0)
    training_data = pd.concat([training_data, data_0], axis=0)

data = process_and_stack(training_data)