import open3d as o3d
import numpy as np
import trimesh

# Create a box
box = trimesh.creation.box(extents=[1, 1, 1])

# Convert trimesh to open3d format
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(np.array(box.vertices))
mesh.triangles = o3d.utility.Vector3iVector(np.array(box.faces))

# Compute vertex normals for better shading
mesh.compute_vertex_normals()

# Visualize with interactive controls
o3d.visualization.draw_geometries([mesh])
