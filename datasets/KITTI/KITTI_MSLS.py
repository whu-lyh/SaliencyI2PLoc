import argparse
import os

import numpy as np
import pandas as pd
import yaml


def check_in_test_set(date, tests):
    if date in tests:
        return True
    else:
        return False

def check_in_test_set_region(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing < point[0] + x_width and
                point[1] - y_width < easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set

def reorganize_data(input_file, output_path, sequence, val_sequences, val_region):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Reorganize and reformat the data
    reorganized_df = pd.DataFrame({
        'ID': range(len(df)),
        'ImageFilename': df['img_path'].apply(lambda x: x.split('/')[-1]),  # Extract image file name
        'LiDARFilename': df['submap_path'].apply(lambda x: x.split('/')[-1]),  # Extract lidar file name
        'East': df['easting'].round(3),
        'North': df['northing'].round(3),
        'Height': df['yaw'],  # Assuming 'yaw' represents height
    })

    # Initialize lists for query and database records
    query_records = []
    database_records = []
    query_val_records = []
    database_val_records = []

    # Store the first record in query_records and use it as the starting point
    last_northing, last_easting = reorganized_df.iloc[0]['North'], reorganized_df.iloc[0]['East']
    query_records.append(reorganized_df.iloc[0])

    if check_in_test_set(sequence, val_sequences):
        # Iterate through the DataFrame starting from the second record
        for i in range(1, len(reorganized_df)):
            current_row = reorganized_df.iloc[i]
            current_northing, current_easting = current_row['North'], current_row['East']
            # Calculate the Euclidean distance from the last selected query point
            distance = np.sqrt((current_northing - last_northing) ** 2 + (current_easting - last_easting) ** 2)
            if check_in_test_set_region(current_northing, current_easting, val_region['p'][sequence], val_region['width'], val_region['width']):
                if distance >= 3.0:
                    # Add to query if distance is 3 meters or more
                    query_val_records.append(current_row)
                    last_northing, last_easting = current_northing, current_easting
                else:
                    # Add to database if distance is less than 3 meters
                    database_val_records.append(current_row)
            else:
                if distance >= 3.0:
                    # Add to query if distance is 3 meters or more
                    query_records.append(current_row)
                    last_northing, last_easting = current_northing, current_easting
                else:
                    # Add to database if distance is less than 3 meters
                    database_records.append(current_row)
    else:
        # Iterate through the DataFrame starting from the second record
        for i in range(1, len(reorganized_df)):
            current_row = reorganized_df.iloc[i]
            current_northing, current_easting = current_row['North'], current_row['East']
            # Calculate the Euclidean distance from the last selected query point
            distance = np.sqrt((current_northing - last_northing) ** 2 + (current_easting - last_easting) ** 2)
            if distance >= 3.0:
                # Add to query if distance is 3 meters or more
                query_records.append(current_row)
                last_northing, last_easting = current_northing, current_easting
            else:
                # Add to database if distance is less than 3 meters
                database_records.append(current_row)

    # Convert query and database lists to DataFrames
    query_df = pd.DataFrame(query_records)
    database_df = pd.DataFrame(database_records)
    query_val_df = pd.DataFrame(query_val_records)
    database_val_df = pd.DataFrame(database_val_records)

    # Define the output file paths
    query_file = os.path.join(output_path, "data_odometry_color/sequences", f"{sequence:02}", f"query.csv")
    database_file = os.path.join(output_path, "data_odometry_color/sequences", f"{sequence:02}", f"database.csv")
    query_val_file = os.path.join(output_path, "data_odometry_color/sequences", f"{sequence:02}", f"query_val.csv")
    database_val_file = os.path.join(output_path, "data_odometry_color/sequences", f"{sequence:02}", f"database_val.csv")

    # Save the DataFrames to CSV files
    query_df.reset_index(drop=True, inplace=True)
    query_df.to_csv(query_file, index=True)
    database_df.reset_index(drop=True, inplace=True)
    database_df.to_csv(database_file, index=True)
    query_val_df.reset_index(drop=True, inplace=True)
    query_val_df.to_csv(query_val_file, index=True)
    database_val_df.reset_index(drop=True, inplace=True)
    database_val_df.to_csv(database_val_file, index=True)

    print(f"Saved {len(query_df)} records to {query_file} and {len(database_df)} records to {database_file}")
    print(f"Saved {len(query_val_df)} records to {query_val_file} and {len(database_val_df)} records to {database_val_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize vxp CSV file format.")
    parser.add_argument("--base_path", type=str, default="/public2/KITTI/vxp_loc_data_folder", help="Path to the input CSV file")
    parser.add_argument("--output_path", type=str, default="/public2/KITTI", help="Path to the output CSV file")
    parser.add_argument("--config_path", type=str, default="./val_region.yaml", help="Path to the configuration file")

    args = parser.parse_args()

    with open(args.config_path) as f:
        data = yaml.load(f, Loader=yaml.loader.FullLoader)
    print(data)

    whole_sequences = data['train']
    val_sequences = data['val']

    for sequence in whole_sequences:
        input_file = os.path.join(args.base_path, f"{sequence:02}.csv")
        # Call the function with the arguments
        reorganize_data(input_file, args.output_path, sequence, val_sequences, data['val_region'])
