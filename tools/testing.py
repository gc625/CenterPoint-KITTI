import numpy as np
import open3d as o3d
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils 
import torch

with open('/workspaces/CenterPoint-KITTI/lidar_pts.npy', 'rb') as f:
    lidar_pts= np.load(f)

with open('/workspaces/CenterPoint-KITTI/radar_pts.npy', 'rb') as f:
    radar_pts= np.load(f)






lidar_for_radar = pointnet2_utils.ball_query(0.8,1,torch.from_numpy(lidar_pts).to('cuda'),torch.from_numpy(radar_pts).to('cuda'))


idx = lidar_for_radar.squeeze(2).squeeze(0).cpu().numpy()

sampled_lidar_points = lidar_pts.squeeze(0)[idx]

np.save('sampled_lidar_points',sampled_lidar_points)

C = [
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [0,0.5,0.5]
]

pts = [sampled_lidar_points,lidar_pts.squeeze(0),radar_pts.squeeze(0)]

geometries = []
for i in range(len(pts)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[i])
    colors = np.zeros_like(pts[i])
    colors[:] = C[i]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # geometries += [pcd]
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 6.0
    geometries.append({'name': f'pcd{i}', 'geometry': pcd, 'material': mat})


o3d.visualization.draw(geometries,show_skybox=False)
print("")