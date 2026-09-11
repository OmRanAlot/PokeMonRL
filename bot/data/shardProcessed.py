import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from pathlib import Path
import os
import glob
import tarfile
from tqdm import tqdm
"""
Turns all the .pt tensor files into large chunks so I can easily upload them to google drive

"""

repo_root = Path(__file__).resolve().parent.parent.parent
info_dir = repo_root / "data" / "data" / "processed_data"
output_dir = repo_root / "data" / "data" / "shards_of_processed_data"

max_gb =5.0
max_size_bytes = max_gb * 1024 * 1024 * 1024
current_tar_size = 0
tar_index = 0
# Get a list of all .pt files
pt_files = sorted(glob.glob(os.path.join(info_dir, "*.pt")))

current_tar_path = os.path.join(output_dir, f"dataset_part_{tar_index:03d}.tar")
tar = tarfile.open(current_tar_path, "w")
print(f"Starting {current_tar_path}...")
    
for file_path in tqdm(pt_files, desc="Packing files"):
    file_size = os.path.getsize(file_path)
        
    if current_tar_size + file_size > max_size_bytes and current_tar_size > 0:
        tar.close()
        tar_index += 1
        current_tar_size = 0
            
        current_tar_path = os.path.join(output_dir, f"dataset_part_{tar_index:03d}.tar")
        tar = tarfile.open(current_tar_path, "w")
        print(f"\nStarting {current_tar_path}...")
            
    # Add file to tar
    arcname = os.path.basename(file_path)
    tar.add(file_path, arcname=arcname)
    current_tar_size += file_size

