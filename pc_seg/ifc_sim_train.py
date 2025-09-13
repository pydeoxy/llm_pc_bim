import ifcopenshell
import time
import numpy as np
from ifcopenshell import geom
import open3d as o3d
import h5py
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import Subset
from torchmetrics import JaccardIndex

from torch_geometric.typing import WITH_TORCH_CLUSTER
if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

from pc_seg.pc_label_map import color_map, color_map_dict
from pc_seg.pc_dataset import H5PCDataset
from pc_seg.pyg_pointnet2 import PyGPointNet2NoColor

# ------------------------------
# Config
# ------------------------------

settings = geom.settings()
settings.set(settings.USE_WORLD_COORDS, True) 

ifc_label_map = {
  "IfcCovering": [
       ("1125344", 3), # "beam", one frame of ceiling used to complete 13 labels
       0, # "ceiling"
      ], 
  "IfcSlab": [
       ("roof", 0), # "ceiling"
       1, # "floor"
      ], 
  "IfcStair": 1, # "floor"
  "IfcWallStandardCase": 2, # "wall"
  "IfcBeam": 3, # "beam"
  "IfcColumn": 4, # "column"
  "IfcWindow": 5, # "window"
  "IfcDoor": 6, # "door"
  "IfcFurnishingElement": [
      ("table", 7), #'table'
      ("chair", 8), # 'chair'
      ("case", 10), #'bookcase'
      ("cabinet", 11), # 'board'
      12  # Default for unmatched furnishings
      ], #12, # "clutter"
  "IfcBuildingElementProxy": [
      ("plaza", 9), # 'sofa'
      12 # Default for unmatched proxies
      ], #12, # "clutter"
  "IfcDistributionElement": 12, # "clutter"  
}

SIM_PC_PATH = './docs/smartLab_sim.ply'
SIM_H5_PATH = './docs/smartLab_sim_dataset.h5'
PRTRAINED_CKPT_PATH = './pc_seg/checkpoints/pointnet2_s3dis_transform_seg_x3_45_checkpoint.pth'          
SIM_CKPT_PATH = './pc_seg/checkpoints/pointnet2_smartlab_sim_finetuned.pth'

BATCH_SIZE=32
NUM_WORKERS=0
EPOCHS = 25

# ------------------------------
# Utility functions
# ------------------------------

def load_ifc(ifc_file_path):
    """
    Load IFC file from its path and get all building elements according to ifc_label_map.
    - file_path: path of the IFC file.
    Returns: loaded elements.
    """
    ifc_file = ifcopenshell.open(ifc_file_path) 
    elements = []
    for tp in ifc_label_map.keys():
        elements += ifc_file.by_type(tp)
    print(f"{len(elements)} building elements loaded from the IFC file.")
    return elements

def sample_points_on_mesh(verts, faces, spacing_mm=10, noise_ratio=0.25, noise_fraction=0.2):
    """
    Sample points on a mesh with a target spacing and optionally add noise to a fraction of them.
    - verts: (N, 3) mesh vertices.
    - faces: (M, 3) triangular face indices.
    - spacing_mm: Desired spacing in mm.
    - noise_ratio: Noise amplitude as a fraction of spacing_mm.
    - noise_fraction: Fraction of points to which noise will be applied (0.0–1.0).
    Returns: (K, 3) sampled points.
    """
    if isinstance(faces, tuple):
        faces = np.array(faces)
    if len(faces.shape) == 1:
        faces = faces.reshape(-1, 3)

    triangles = verts[faces]
    vec1 = triangles[:, 1] - triangles[:, 0]
    vec2 = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(vec1, vec2), axis=1)

    total_area_mm2 = np.sum(areas)
    points_per_mm2 = 1 / (spacing_mm ** 2)
    num_points = int(total_area_mm2 * points_per_mm2)
    num_points = max(num_points, 1)

    probs = areas / areas.sum()
    sampled_tri_indices = np.random.choice(len(faces), size=num_points, p=probs)
    sampled_tris = triangles[sampled_tri_indices]

    u = np.random.rand(num_points, 1)
    v = np.random.rand(num_points, 1)
    mask = (u + v) > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v)

    sampled_points = (u * sampled_tris[:, 0]) + (v * sampled_tris[:, 1]) + (w * sampled_tris[:, 2])

    # Decide which points will get noise
    num_noisy = int(num_points * noise_fraction)
    noisy_indices = np.random.choice(num_points, size=num_noisy, replace=False)

    # Apply noise only to selected points
    noise_amplitude = spacing_mm * noise_ratio
    noise = np.random.uniform(-noise_amplitude, noise_amplitude, size=(num_noisy, 3))
    sampled_points[noisy_indices] += noise

    return sampled_points

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

def preprocess_points(pcd):
    points = np.asarray(pcd.points) 
    colors = np.asarray(pcd.colors)      
    moved_points = move_to_corner(np.array(pcd.points))
    pcd.points = o3d.utility.Vector3dVector(moved_points)    
    return pcd

# ------------------------------
# IFC -> point cloud sampling
# ------------------------------

def ifc_sample_points(ifc_file_path, sim_pc_path):
    """
    Sample points on meshes from the IFC elements.
    - ifc_file_path: path of the IFC file.
    - sim_pc_path: path to save the simulated point cloud.
    Returns: point cloud with colors according to ifc_label_map.
    """
    # Track sampling time
    start_time = time.perf_counter()    
        
    elements = load_ifc(ifc_file_path)

    points = []
    labels = []

    for element in elements:
        if element.Representation is None:
            print(f"Skipping {element.GlobalId}: No representation")
            continue
        # Get geometry
        shape = geom.create_shape(settings, element)
        verts = shape.geometry.verts  # Vertex coordinates (flat list)
        faces = shape.geometry.faces  # Triangular faces (indices)
        
        # Reshape vertices into (N, 3) array
        verts = np.array(verts).reshape(-1, 3)
        
        # Generate points on the mesh surface (see Step 4)
        spacing_mm = 0.01  # Points every 10mm
        sampled_points = sample_points_on_mesh(verts, faces, spacing_mm=spacing_mm)
        points.extend(sampled_points)
        
        # Assign label
        element_type = element.is_a()
        label_spec = ifc_label_map.get(element_type, -1)  # Default to -1 for unknown classes
        label = -1

        # Handle list-based specifications with keyword matching
        if isinstance(label_spec, list):
            default = None
            for item in label_spec:
                if isinstance(item, tuple):
                    # Get element name in lowercase (handle potential None values)
                    element_name = element.Name.lower()
                    if item[0] in element_name:
                        label = item[1]
                        break  # First match wins
                elif isinstance(item, int):
                    default = item  # Store potential default value
            
            # Use default if no matches found and default exists
            if label == -1 and default is not None:
                label = default
        elif isinstance(label_spec, int):
            # Direct integer mapping
            label = label_spec

        labels.extend([label] * len(sampled_points))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color_map[labels])    
    moved_points = preprocess_points(pcd)
    sampling_time = time.perf_counter() - start_time
    print("Simulated point cloud generated.")
    print(f"Time used: {sampling_time:.2f}s")
    # Save the point cloud
    o3d.io.write_point_cloud(sim_pc_path, moved_points)
    print(f"Simulated point cloud saved as {sim_pc_path}.")

    return moved_points

# ------------------------------
# Point cloud -> HDF5 dataset
# ------------------------------

def pcd_to_h5(pcd, sim_data_path):
    # Track processing time
    start_time = time.perf_counter()    

    points = np.asarray(pcd.points) 
    colors = np.asarray(pcd.colors)             
    normalized = normalize_points_corner(np.array(pcd.points))

    features = np.concatenate([points, colors, normalized], axis=1)

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    block_size = 1.0  # in meters
    num_blocks_x = int(np.ceil((max_bound[0] - min_bound[0]) / block_size))
    num_blocks_y = int(np.ceil((max_bound[1] - min_bound[1]) / block_size))
    num_blocks_z = int(np.ceil((max_bound[2] - min_bound[2]) / block_size))

    block_features_list = []
    block_labels_list  = []

    for ix in range(num_blocks_x):
        for iy in range(num_blocks_y):
            for iz in range(num_blocks_z):
                # Define the spatial boundaries for this block
                x_min = min_bound[0] + ix * block_size
                x_max = x_min + block_size
                y_min = min_bound[1] + iy * block_size
                y_max = y_min + block_size
                z_min = min_bound[2] + iz * block_size
                z_max = z_min + block_size

                # Find indices of points within the block
                in_block = np.where(
                    (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] < z_max)
                )[0]

                if len(in_block) == 0:
                    continue  # Skip empty blocks

                block_features = features[in_block, :]
                # If label is directly extractable, substitute here. If not, infer from color.
                block_colors = colors[in_block]
                block_labels = np.argmin(np.linalg.norm(colors[in_block][:, None] - color_map, axis=2), axis=1)

                # --- Handling Block Size (4096 points) ---
                # If there are more than 4096 points, randomly sample 4096.
                # If there are fewer, perform random duplication (or padding with zeros) to reach 4096.
                num_points = block_features.shape[0]
                target_points = 4096

                if num_points >= target_points:
                    idx = np.random.choice(num_points, target_points, replace=False)
                else:
                    # Duplicate some points
                    idx = np.concatenate([
                        np.arange(num_points),
                        np.random.choice(num_points, target_points - num_points, replace=True)
                    ])
                block_features = block_features[idx, :]
                block_labels = block_labels[idx]

                block_features_list.append(block_features)
                block_labels_list.append(block_labels)

    data_array = np.stack(block_features_list, axis=0)  
    label_array = np.stack(block_labels_list, axis=0) 

    # Save to HDF5
    with h5py.File(sim_data_path, 'w') as f:
        f.create_dataset('data', data=data_array, compression='gzip')
        f.create_dataset('label', data=label_array, compression='gzip')

    processing_time = time.perf_counter() - start_time
    print(f"H5PY dataset from simulated point cloud saved as {sim_data_path}.")
    print(f"Time used: {processing_time:.2f}s")

# ------------------------------
# Training dataset preparation
# ------------------------------

class SelectLast3Features:
        def __call__(self, data):
            # If data.x is defined, select only its last 3 features.
            if data.x is not None:
                data.x = data.x[:, -3:]
            return data

class AugmentedSubset(Subset):
    def __init__(self, subset, transform):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return self.transform(data)

def load_h5_dataset(sim_data_path):    

    transform = T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
        ])

    pre_transform =  T.Compose([
        SelectLast3Features()
        ])

    # Create the dataset
    full_dataset = H5PCDataset(sim_data_path, pre_transform = pre_transform)

    # Define split sizes (e.g., 80% training and 20% validation)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    # Randomly split the dataset
    train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

    train_dataset = AugmentedSubset(train_subset, transform)
    test_dataset = test_subset 

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    return train_loader, test_loader

# ------------------------------
# Training functions
# ------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = correct = total = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct      += out.argmax(1).eq(data.y).sum().item()
        total        += data.num_nodes

    # Average loss & accuracy for this epoch
    epoch_loss = running_loss / len(loader) 
    epoch_acc  = correct / total

    return epoch_loss, epoch_acc

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    jaccard = JaccardIndex(num_classes=loader.dataset.dataset.num_classes, task="multiclass").to(device)
    
    for data in loader:
        data = data.to(device)
        outs = model(data)
        preds = outs.argmax(dim=-1)
        jaccard.update(preds, data.y)
    
    return jaccard.compute().item()
    
# ------------------------------
# Main Finetuning Training 
# ------------------------------

def finetuning_train(ifc_file_path):   
    # Convert IFC to training dataset
    pcd = ifc_sample_points(ifc_file_path, SIM_PC_PATH)
    pcd_to_h5(pcd, SIM_H5_PATH)
    train_loader, test_loader = load_h5_dataset(SIM_H5_PATH)

    # Initialize the pretrained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyGPointNet2NoColor(num_classes=13).to(device)
    checkpoint = torch.load(PRTRAINED_CKPT_PATH, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict, strict=True)  
    model.eval()
    print("Pretrained segmentation model loaded.")

    optimizer = torch.optim.Adam(
        model.parameters(),  
        lr=1e-4,
        weight_decay=0.01
    )    

    # Fine-tuning training    
    for epoch in range(1, EPOCHS+1):
        # Track epoch time
        start_time = time.perf_counter()
        loss, acc = train_one_epoch(model, train_loader, optimizer, device)  
        iou = test(model, test_loader, device)
        epoch_time = time.perf_counter() - start_time
        
        # Print results with time
        print(f"Epoch {epoch:02d} | "
            f"Loss: {loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"IoU: {iou:.4f} | "
            f"Time: {epoch_time:.2f}s")

    # Save model, optimizer state, and any other info needed
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'loss': loss,
        #'test_accuracy': test_acc
    }, SIM_CKPT_PATH)

    print(f"Checkpoint saved as {SIM_CKPT_PATH}.")

# Training Test
if __name__ == "__main__":
    ifc_file_path = './docs/smartLab.ifc'
    #pcd = ifc_sample_points(ifc_file_path, SIM_PC_PATH)
    #o3d.visualization.draw_geometries([pcd])
    #pcd_to_h5(pcd, SIM_H5_PATH)

    #finetuning_train(ifc_file_path)

    '''device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyGPointNet2NoColor(num_classes=13).to(device)
    checkpoint = torch.load(PRTRAINED_CKPT_PATH, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict, strict=True)  
    model.eval()'''
    