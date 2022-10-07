import open3d as o3d
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
import os
from os.path import join as osjoin
import copy
from tqdm import tqdm
import shutil

def csv2geometry(fname, fix_height=None):

    radar_data = np.loadtxt(
        fname,
        delimiter=',',
        ndmin=1,
        skiprows=1,
        dtype=[
            ('id', 'u4'),
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('vx', 'f4'), ('vy', 'f4'),
            ('fPower', 'f4'), ('fRCS', 'f4'), ('fSpeed', 'f4')
        ]
    )

    radar_points_xyz = np.column_stack((radar_data['x'], radar_data['y'], radar_data['z']))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(radar_points_xyz)
    return pcd

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

def get_rotation(arr):
    x,y,_ = arr[:3]
    yaw = np.arctan(y/x)
    angle = np.array([0, 0, yaw])
    r = R.from_euler('XYZ', angle)
    return r.as_matrix()

def get_bbx_param(obj_info, scale=1.1):

    center = obj_info[2:5] 
    extent = obj_info[5:8] * 1.3
    extent[2] += 10
    
    angle = obj_info[8:]

    rot_m = get_rotation(angle)
    
    obbx = o3d.geometry.OrientedBoundingBox(center.T, rot_m, extent.T)
    return obbx

ext_params = {
    '0118': {
        'lidar': get_matrix_from_ext([0.00, 0.0, -0.3, -2.5, 0.0, 0]),
        'radar': get_matrix_from_ext([0.06, -0.2, 0.7, -3.5, 2.0, 180]),
    },
    '0126': {
        'lidar': get_matrix_from_ext([0.00, 0.0, 0.0, -1.0, 2.0, 0]),
        'radar': get_matrix_from_ext([0.06, -0.2, 0.2, -1.0, 2.0, 180]),
    }
}

def get_date_key(ts):
    return '0118' if ts < 1643000000000 else '0126'

def filter_radar_label(save_dir, filter_num=3):
    
    radar_files = sorted(glob(osjoin(save_dir, 'inhouse_format', 'radar/*.csv')))
    gt_files = sorted(glob(osjoin(save_dir, 'inhouse_format', 'gt/*.csv')))
    filtered_path = osjoin(save_dir, 'inhouse_format', 'radar_gt')

    if not os.path.exists(filtered_path):
        os.mkdir(filtered_path)
    for idx in tqdm(range(len(radar_files))):
        gt_name = gt_files[idx].split('/')[-1]
        ts = int(gt_name[:-4])
        save_gt_name = os.path.join(filtered_path, gt_name)
        radar_pcd = csv2geometry(radar_files[idx])
        collection_date = get_date_key(ts)
        radar_pcd.transform(ext_params[collection_date]['radar'])
        box_list = []
        gt_name = gt_files[idx].split('/')[-1]
        save_gt_name = os.path.join(filtered_path, gt_name)
        gt_data = np.loadtxt(gt_files[idx])
        
        if gt_data.size == 0:
            # save empty array
            np.savetxt(save_gt_name, gt_data, fmt='%s')
            continue
        if len(gt_data.shape) == 1:
            gt_data = gt_data.reshape(1, -1)
        point_flag = np.array([])
        for obj_idx, obj_info in enumerate(gt_data):
            obj_bbx = get_bbx_param(obj_info)
            num_list = obj_bbx.get_point_indices_within_bounding_box(radar_pcd.points)
            if len(num_list) > filter_num:
                gt_data[obj_idx][-1] = len(num_list)
                point_flag = np.append(point_flag, True)
            else:
                point_flag = np.append(point_flag, False)

        new_gt_data = gt_data[point_flag.astype(bool)]
        if len(new_gt_data.shape) == 1:
            new_gt_data = new_gt_data.reshape(1, -1)
        for new_obj_info in new_gt_data:
            obj_bbx = get_bbx_param(new_obj_info)
            box_list += [obj_bbx]
        
        np.savetxt(save_gt_name, new_gt_data, fmt='%s')

if __name__ == '__main__':
    filter_radar_label('/public/shangqi/')