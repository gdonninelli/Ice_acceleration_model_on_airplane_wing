import os
import glob
import numpy as np
import pandas as pd

dataset_dir = "dataset"
sdf_data_dir = "SDF/data"

summary_files = glob.glob(os.path.join(dataset_dir, "summary_*.txt"))

X_sdf = []
X_scalars = [] # (reynolds, angle)
Y_cl = []
metadata = [] # (naca_id)

for summary_file in summary_files:
    print(f"Processing {summary_file}")
    df = pd.read_csv(summary_file)
    for index, row in df.iterrows():
        csv_path = row['csv_path']
        reynolds = float(row['reynolds'])
        angle = float(row['angle'])
        cl = float(row['mean_last10pct_C_L'])
        
        # Keep only observations with angle of attack between -14 and 14
        if not (-14.0 <= angle <= 14.0):
            continue
        
        # Extract NACA from csv_path
        naca_id = csv_path.split('/')[0]
        
        if naca_id == "naca0012":
            matrix_prefix = "n0012"
        else:
            matrix_prefix = naca_id
            
        if angle.is_integer():
            angle_str = str(int(angle))
        else:
            angle_str = str(angle)
            
        matrix_filename = f"{matrix_prefix}_alpha_{angle_str}_matrix.txt"
        matrix_filepath = os.path.join(sdf_data_dir, matrix_filename)
        
        if not os.path.exists(matrix_filepath):
            print(f"Warning: Matrix not found: {matrix_filepath}")
            continue
            
        # Load the matrix
        try:
            sdf_matrix = np.loadtxt(matrix_filepath)
        except Exception as e:
            print(f"Error loading {matrix_filepath}: {e}")
            continue
            
        X_sdf.append(sdf_matrix)
        X_scalars.append([reynolds, angle])
        Y_cl.append(cl)
        # Only keep the NACA id in metadata, no paths
        metadata.append([naca_id])

X_sdf = np.array(X_sdf, dtype=np.float32)
X_scalars = np.array(X_scalars, dtype=np.float32)
Y_cl = np.array(Y_cl, dtype=np.float32)
metadata = np.array(metadata, dtype=str)

print("Full dataset shapes after filtering [-14, 14]:")
print(f"X_sdf: {X_sdf.shape}")
print(f"X_scalars: {X_scalars.shape}")
print(f"Y_cl: {Y_cl.shape}")

# Split into train (80%) and test (20%)
num_samples = X_sdf.shape[0]
indices = np.random.permutation(num_samples)
split_idx = int(num_samples * 0.8)

train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

# Save Train Dataset
train_path = os.path.join(dataset_dir, "cnn_dataset_train.npz")
np.savez_compressed(train_path, 
                    X_sdf=X_sdf[train_idx], 
                    X_scalars=X_scalars[train_idx], 
                    Y_cl=Y_cl[train_idx], 
                    metadata=metadata[train_idx])

# Save Test Dataset
test_path = os.path.join(dataset_dir, "cnn_dataset_test.npz")
np.savez_compressed(test_path, 
                    X_sdf=X_sdf[test_idx], 
                    X_scalars=X_scalars[test_idx], 
                    Y_cl=Y_cl[test_idx], 
                    metadata=metadata[test_idx])

print(f"Train dataset saved to {train_path} with {len(train_idx)} samples")
print(f"Test dataset saved to {test_path} with {len(test_idx)} samples")
