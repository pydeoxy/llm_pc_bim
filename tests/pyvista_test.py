import pyvista as pv
import open3d as o3d
import numpy as np

pcd_path = "C:/Users/yanpe/OneDrive - Metropolia Ammattikorkeakoulu Oy/Research/data/smartlab/SmartLab_2024_E57_Single_5mm.pcd"

import pyvista as pv
from pyvistaqt import BackgroundPlotter

pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
point_cloud = pv.PolyData(points)
if colors.size:    
    point_cloud['RGB'] = colors
  

# Create a non-blocking background plotter
plotter = BackgroundPlotter()
plotter.add_points(point_cloud, scalars='RGB', rgb=True, point_size=5)

# Main thread continues execution
print("Visualization is running in the background. Main thread is free...")

# Keep the program alive (e.g., run other tasks or wait)
input("Press Enter to exit...")
