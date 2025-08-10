import os
import open3d as o3d
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.typing import WITH_TORCH_CLUSTER
from pyg_pointnet2 import PyGPointNet2
from pc_label_map import color_map
import time
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------
PCD_PATH = "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Research/data/smartlab/SmartLab_2024_E57_Single_5mm.pcd"
CHECKPOINT_FILE = "pointnet2_s3dis_transform_seg_x6_45_checkpoint.pth"

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
    print('point cloud loaded.')

    moved_points = move_to_corner(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(moved_points)
    print('point cloud moved.')

    downpcd = pcd.voxel_down_sample(voxel_size=0.03)
    print('downsample done.')

    normalized = normalize_points_corner(np.array(downpcd.points))
    print('normalized done.')

    down_points = torch.tensor(np.array(downpcd.points), dtype=torch.float32)  
    down_colors = torch.tensor(np.array(downpcd.colors), dtype=torch.float32)
    down_normalized = torch.tensor(normalized, dtype=torch.float32)
    print('features prepared.')

    x = torch.cat([down_colors, down_normalized], dim=1)
    data = Data(x=x, pos=down_points)

    return [data], downpcd

# ------------------------------
# Main segmentation function
# ------------------------------
def run_segmentation(dataset, downpcd):
    num_workers = 10
    batch_size = 32

    custom_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    print('data loaded.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyGPointNet2(num_classes=13).to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(script_dir, "checkpoints", CHECKPOINT_FILE)

    checkpoint = torch.load(model_file_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    model.eval()
    print('seg model loaded.')

    labels = None
    with torch.no_grad():
        start_time = time.time()
        print('segmentation started.')
        for data in tqdm(custom_loader, total=len(custom_loader), desc="Predicting", unit="batch"):
            data = data.to(device)
            predictions = model(data)
            labels = predictions.argmax(dim=-1)
            unique_labels, label_counts = torch.unique(labels, return_counts=True)        
            result_labels = torch.stack((unique_labels, label_counts), dim=1).cpu()
            print("Label counts:")
            print(result_labels)
        end_time = time.time()
        print(f"Total inference time: {end_time - start_time:.4f} seconds")

    # Apply colors
    predicted_colors = color_map[labels.cpu().numpy()]
    downpcd.colors = o3d.utility.Vector3dVector(predicted_colors)
    print('point cloud labelled.')

    return downpcd

# ------------------------------
# Execution
# ------------------------------
if __name__ == "__main__":
    dataset, downpcd = prepare_dataset(PCD_PATH)
    downpcd = run_segmentation(dataset, downpcd)

# Optional visualization
# o3d.visualization.draw_geometries([downpcd])