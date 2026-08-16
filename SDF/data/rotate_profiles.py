import os
import glob
import math

baseline_files = ["naca0006.dat", "naca0008.dat", "naca0010.dat", "n0012.dat", "naca0018.dat"]

# Paths are resolved from this file's location so the script runs from any
# working directory and on any machine.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
data_dir = os.environ.get("SDF_DATA_DIR", os.path.join(repo_root, "SDF", "data"))
dataset_dir = os.environ.get("DATASET_DIR", os.path.join(repo_root, "dataset"))

# Collect unique angles
angles = set()
for summary_file in glob.glob(os.path.join(dataset_dir, "summary_*.txt")):
    with open(summary_file, "r") as f:
        lines = f.readlines()
        for line in lines[1:]: # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 3:
                angle_str = parts[2]
                try:
                    angle_val = float(angle_str)
                    angles.add(angle_val)
                except ValueError:
                    pass

print(f"Found {len(angles)} unique angles to generate.")

for filename in baseline_files:
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filename} not found!")
        continue
        
    with open(filepath, "r") as f:
        lines = f.readlines()
        
    header_count = 3 if filename == "n0012.dat" else 1
    
    for alpha in angles:
        rad = math.radians(alpha)
        
        name_part, ext = os.path.splitext(filename)
        if alpha.is_integer():
            alpha_str = str(int(alpha))
        else:
            alpha_str = str(alpha)
            
        output_filename = f"{name_part}_alpha_{alpha_str}{ext}"
        output_filepath = os.path.join(data_dir, output_filename)
        
        # Skip if the .dat file already exists
        if os.path.exists(output_filepath):
            continue
            
        with open(output_filepath, "w") as out_f:
            # Write header lines unchanged
            for i in range(header_count):
                out_f.write(lines[i])
                
            # Process remaining lines
            for line in lines[header_count:]:
                line_stripped = line.strip()
                if not line_stripped:
                    # Keep empty lines
                    out_f.write(line)
                    continue
                
                parts = line_stripped.split()
                if len(parts) >= 2:
                    try:
                        x = float(parts[0])
                        z = float(parts[1])
                        # Rotation around leading edge (0,0)
                        x_rot = x * math.cos(rad) + z * math.sin(rad)
                        z_rot = -x * math.sin(rad) + z * math.cos(rad)
                        
                        # Preserve formatting style (leading spaces, decimal places)
                        if filename == "n0012.dat":
                            out_f.write(f" {x_rot:.7f} {z_rot:.7f}\n")
                        else:
                            out_f.write(f" {x_rot:.6f} {z_rot:.6f}\n")
                    except ValueError:
                        out_f.write(line)
                else:
                    out_f.write(line)

print("Rotation generation complete!")
