import os
import open3d as o3d
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER
from pyg_pointnet2 import PyGPointNet2
from pc_label_map import color_map,color_map_dict
import time
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------
PCD_PATH = "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Research/data/smartlab/SmartLab_2024_E57_Single_5mm.pcd"
CHECKPOINT_FILE = "pointnet2_s3dis_transform_seg_x6_45_checkpoint.pth"
CHUNK_SIZE = 4096
BATCH_SIZE = 48  # Increase if GPU memory allows
NUM_WORKERS = 20  # 0 for Windows if issues occur

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

def split_point_cloud(points, features, chunk_size):
    """Split the point cloud into smaller Data objects."""
    dataset = []
    for i in range(0, points.shape[0], chunk_size):
        p = points[i:i+chunk_size]
        f = features[i:i+chunk_size]
        dataset.append(Data(x=f, pos=p))
    return dataset

def prepare_dataset(pcd_path, device):
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f'Loaded point cloud with {len(pcd.points)} points.')

    moved_points = move_to_corner(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(moved_points)

    downpcd = pcd.voxel_down_sample(voxel_size=0.03)
    print(f'Downsampled to {len(downpcd.points)} points.')

    normalized = normalize_points_corner(np.array(downpcd.points))

    # Move preprocessing to GPU early
    down_points = torch.tensor(np.array(downpcd.points), dtype=torch.float32)
    down_colors = torch.tensor(np.array(downpcd.colors), dtype=torch.float32)
    down_normalized = torch.tensor(normalized, dtype=torch.float32)

    x = torch.cat([down_colors, down_normalized], dim=1)

    dataset = split_point_cloud(down_points, x, CHUNK_SIZE)
    return dataset, downpcd

# ------------------------------
# Main segmentation function
# ------------------------------
def run_segmentation(dataset, downpcd, device):
    custom_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    print('Data prepared for inference.')

    model = PyGPointNet2(num_classes=13).to(device)
    model_file_path = os.path.join(os.path.dirname(__file__), "checkpoints", CHECKPOINT_FILE)

    checkpoint = torch.load(model_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()    
    print('Segmentation model loaded.')

    all_labels = []

    start_time = time.time()
    with torch.inference_mode():
        for data in tqdm(custom_loader, total=len(custom_loader), desc="Predicting", unit="batch"):
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

    print("\nPrediction Summary:")
    for lbl, cnt in zip(unique_labels.tolist(), label_counts.tolist()):
        color, name = color_map_dict[lbl]
        print(f"Label {lbl} ({name}): {cnt} points")

    print(f"\nTotal points: {all_labels.numel()}")
    

    predicted_colors = color_map[all_labels]
    downpcd.colors = o3d.utility.Vector3dVector(predicted_colors)
    print('Point cloud labelled.')

    return downpcd

# ------------------------------
# Execution
# ------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, downpcd = prepare_dataset(PCD_PATH, device)
    downpcd = run_segmentation(dataset, downpcd, device)

    # Optional visualization
    # o3d.visualization.draw_geometries([downpcd])
