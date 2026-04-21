# import open3d as o3d

# # Load PLY
# pcd = o3d.io.read_point_cloud(r"C:\Users\ve00yn139\Downloads\archive (1)\pc_cad\0088\00884693.ply")

# # Show
# o3d.visualization.draw_geometries([pcd])
import open3d as o3d
import numpy as np

# Load point cloud
pcd = o3d.io.read_point_cloud(r"C:\Users\ve00yn139\Downloads\archive (1)\pc_cad\0088\00884693.ply")

# Step 1 — estimate normals (needed for mesh reconstruction)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(100)

# Step 2 — Poisson surface reconstruction → gives a solid mesh
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9
)

# Step 3 — remove low density vertices (cleans up loose surfaces)
densities = np.asarray(densities)
keep = densities > np.percentile(densities, 10)  # raise to 20-30 if still noisy
mesh.remove_vertices_by_mask(~keep)

# Step 4 — clean up
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.6, 0.7, 0.85])  # steel blue, looks like CAD

# Step 5 — render
vis = o3d.visualization.Visualizer()
vis.create_window(width=1200, height=900, window_name="CAD Solid")
vis.add_geometry(mesh)

opt = vis.get_render_option()
opt.background_color = np.array([1, 1, 1])   # white background
opt.mesh_show_back_face = True
opt.light_on = True

vis.run()
vis.capture_screen_image("solid_cad.png")
vis.destroy_window()