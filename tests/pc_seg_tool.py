import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, knn_interpolate
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.data import Data

import open3d as o3d
import numpy as np

from sklearn.linear_model import RANSACRegressor
from scipy.spatial import ConvexHull

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 6, 64, 64, 128])) # 3 (pos) + 6 (x)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 6, 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, num_classes)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x).log_softmax(dim=-1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_classes=13).to(device)

# Load the checkpoint dictionary
checkpoint = torch.load(".\checkpoints\pointnet2_s3dis_seg_x6_30_checkpoint.pth", map_location=device)
# Extract the model state dictionary
model_state_dict = checkpoint['model_state_dict']

model.load_state_dict(model_state_dict)
model.eval()

pcd_path = "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Research/data/smartlab/SmartLab_2024_E57_Single_5mm.pcd"
#pcd_path = "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Research/data/smartlab/SmartLab_2024_E57_Single_5mm.ply"
pcd = o3d.io.read_point_cloud(pcd_path)

def center_points(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Move the points so the centroid is at (0, 0, 0)
    centered_points = points - centroid
    
    return centered_points

centered_points = center_points(np.array(pcd.points))

pcd.points = o3d.utility.Vector3dVector(centered_points)

downpcd = pcd.voxel_down_sample(voxel_size=0.02)

def normalize_points(points):
    # Step 1: Center the points around the origin
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Step 2: Scale to fit within the [0, 1] interval
    min_vals = np.min(centered_points, axis=0)
    max_vals = np.max(centered_points, axis=0)
    scale = max_vals - min_vals

    # Avoid division by zero in case of flat dimension
    scale[scale == 0] = 1  # Set zero scales to 1 to keep that dimension as 0.5 after normalization

    normalized_points = (centered_points - min_vals) / scale

    return normalized_points

normalized = normalize_points(np.array(downpcd.points))

# Extract coordinates and colors from the point cloud
down_points = torch.tensor(np.array(downpcd.points), dtype=torch.float32)  
down_colors = torch.tensor(np.array(downpcd.colors), dtype=torch.float32)
down_normalized = torch.tensor(normalized, dtype=torch.float32)

# Concatenate coordinates and colors to form the input features
x = torch.cat([down_colors, down_normalized], dim=1)

data = Data(x=x, pos=down_points)
dataset = [data] 
custom_loader = DataLoader(dataset, batch_size=12, shuffle=False,
                         num_workers=8)

model.eval()
with torch.no_grad():
    for data in custom_loader:
        data = data.to(device)
        predictions = model(data)
        labels = predictions.argmax(dim=-1)
        # Process the labels as needed
        labels_arr = labels.cpu().numpy()
        # Count occurrences of labels
        unique_labels, label_counts = np.unique(labels_arr, return_counts=True)
        # Combine and print
        result_labels = np.array(list(zip(unique_labels, label_counts)))
        print("Label counts:")
        print(result_labels)    

# Define the number of classes in your model's predictions
num_classes = 13  # Adjust based on your number of classes


# Define a fixed color map for 13 labels
color_map = np.array([
    [1.0, 0.0, 0.0],  # Label 0: Red,  'ceiling'
    [0.0, 1.0, 0.0],  # Label 1: Green, 'floor'
    [0.0, 0.0, 1.0],  # Label 2: Blue,  'wall'
    [1.0, 1.0, 0.0],  # Label 3: Yellow, 'beam'
    [1.0, 0.0, 1.0],  # Label 4: Magenta, 'column'
    [0.0, 1.0, 1.0],  # Label 5: Cyan, 'window'
    [0.5, 0.5, 0.5],  # Label 6: Gray, 'door'
    [1.0, 0.5, 0.0],  # Label 7: Orange, 'chair'
    [0.5, 0.0, 1.0],  # Label 8: Purple, 'table'
    [0.5, 1.0, 0.5],  # Label 9: Light Green, 'bookcase'
    [0.5, 0.5, 1.0],  # Label 10: Light Blue, 'sofa'
    [1.0, 0.5, 0.5],  # Label 11: Pink, 'board'
    [0.0, 0.0, 0.0]   # Label 12: Black, 'clutter'
    ])

predicted_colors = color_map[labels.cpu().numpy()] 

downpcd.colors = o3d.utility.Vector3dVector(predicted_colors)
