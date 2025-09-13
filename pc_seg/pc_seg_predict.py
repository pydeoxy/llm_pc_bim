import os
import open3d as o3d
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER
from pyg_pointnet2 import PyGPointNet2NoColor
from pc_label_map import color_map, color_map_dict
import time
import multiprocessing

# ------------------------------
# Config
# ------------------------------

SAVE_PATH = "docs/downpcd_lablled.pcd"
BATCH_SIZE = 32  
NUM_WORKERS = 10  

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

# ------------------------------
# Utility functions
# ------------------------------
def move_to_corner(points):    
    min_xyz = points.min(axis=0)
    return points - min_xyz

def normalize_points_corner(points):
    min_vals = np.min(points, axis=0)
    shifted_points = points - min_vals
    max_vals = np.max(shifted_points, axis=0)
    scale = max_vals.copy()
    scale[scale == 0] = 1
    return shifted_points / scale

def prepare_dataset(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f'Loaded point cloud with {len(pcd.points)} points.')

    moved_points = move_to_corner(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(moved_points)

    downpcd = pcd.voxel_down_sample(voxel_size=0.03)
    print(f'Downsampled to {len(downpcd.points)} points.')

    normalized = normalize_points_corner(np.array(downpcd.points))

    down_points = torch.tensor(np.array(downpcd.points), dtype=torch.float32)  
    down_colors = torch.tensor(np.array(downpcd.colors), dtype=torch.float32)
    down_normalized = torch.tensor(normalized, dtype=torch.float32)

    data = Data(x=down_normalized, pos=down_points)
    dataset = [data]
    return dataset, downpcd

def visualize_pcd(pcd):    
    o3d.visualization.draw_geometries([pcd], point_show_normal=False)    

# ------------------------------
# Main segmentation function
# ------------------------------
def run_segmentation(dataset, downpcd, device, model_file_path):
    message = ""
    custom_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    print('Data prepared for inference.')

    model = PyGPointNet2NoColor(num_classes=13).to(device)
    
    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('Segmentation model loaded.')

    all_labels = []

    with torch.inference_mode():
        start_time = time.time()
        for data in custom_loader:
            data = data.to(device, non_blocking=True)
            preds = model(data)
            labels = preds.argmax(dim=-1)
            all_labels.append(labels.cpu())
    
    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.4f} seconds")    
    
    # Concatenate all predicted labels
    all_labels = torch.cat(all_labels, dim=0).cpu()    
    # Count per label
    unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

    message += "\nPrediction Summary:"
    for lbl, cnt in zip(unique_labels.tolist(), label_counts.tolist()):
        color, name = color_map_dict[lbl]
        message += f"\nLabel {lbl} ({name}): {cnt} points"
    message += f"\nTotal points: {all_labels.numel()}"
   
    # Apply colors
    predicted_colors = color_map[labels.cpu().numpy()]
    downpcd.colors = o3d.utility.Vector3dVector(predicted_colors)
    print('Point cloud labelled.')

    # Save and visualize the result    
    o3d.io.write_point_cloud(SAVE_PATH, downpcd)
    message += f'\nLabelled point cloud saved as {SAVE_PATH}.'

    return downpcd, message, SAVE_PATH

# ------------------------------
# Execution
# ------------------------------
if __name__ == "__main__":
    pcd_path = "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Research/data/smartlab/SmartLab_2024_E57_Single_5mm.pcd"
    model_file_path = "pc_seg/checkpoints/pointnet2_smartlab_sim_finetuned.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, downpcd = prepare_dataset(pcd_path)
    downpcd, message, save_path = run_segmentation(dataset, downpcd, device, model_file_path)
    o3d.visualization.draw_geometries([downpcd])
    #print(message)

