import os
import shutil
import kagglehub

kagglehub.login()
cache_path = kagglehub.dataset_download("giulioenzodonninelli/sdf-symmetric-airfoil-high-reynolds-number")

local_target_path = "./data"


if os.path.exists(local_target_path):
    shutil.rmtree(local_target_path)

shutil.copytree(cache_path, local_target_path)
print(f"Dataset successfully saved to your project path: {os.path.abspath(local_target_path)}")
