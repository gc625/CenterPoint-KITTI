# sync and concat multi-radar data to lidar scans with pose compensation
import csv
import open3d
import numpy as np
from math import sin, cos, pi
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from glob import glob
from os.path import join as osp
import pandas as pd
from pose_extract import get_trans_from_gnssimu, get_matrix_from_ext, get_interpolate_pose
import argparse

def get_cor_radar_idx(lidar_ts, radar_ts):
    
    f_cor = []
    l_cor = []
    r_cor = []
    cor_idx = {"front_idx": [], "left_idx": [], "right_idx": []}
    cor_ts = {"front_ts": [], "left_ts": [], "right_ts": []}
    for target in lidar_ts:
        for key in radar_ts:
            diff = abs(radar_ts[key] - target)
            idx = diff.argmin()
            cor_idx[key[:-3]+"_idx"].append(idx)
            cor_ts[key].append(radar_ts[key][idx])
            
    for key in cor_idx:
        cor_idx[key]=np.array(cor_idx[key])
    for key in cor_ts:
        cor_ts[key]=np.array(cor_ts[key])
   
    return cor_idx,cor_ts

def get_cor_gt_idx(lidar_ts, gt_ts):
    
    cor_idx = []
    cor_ts = []
    
    for target in lidar_ts:
        diff = abs(gt_ts - target)
        idx = diff.argmin()
        cor_idx.append(idx)
        cor_ts.append(gt_ts[idx])
    
    cor_idx = np.array(cor_idx)
    cor_ts = np.array(cor_ts)
    
    return cor_idx, cor_ts

def radar_temp_comp(data, data_ts, lidar_ts, pose, pose_ts, radar_trans, target_trans):
    
    xyz = np.concatenate((data[:3],np.ones((1,len(data[0])))),axis=0)
    src_diff = abs(pose_ts-data_ts)
    src_idx = src_diff.argmin()
    src_pose = pose[src_idx]
    tgt_diff = abs(pose_ts-lidar_ts)
    tgt_idx = tgt_diff.argmin()
    tgt_pose = pose[tgt_idx]
    # transform from radar to ego
    ego_xyz = radar_trans @ xyz 
    # transform from source to global
    glb_xyz = src_pose @ ego_xyz
    # tranform from global to target
    tgt_xyz = np.linalg.inv(tgt_pose) @ glb_xyz
    # transform from target timestamp to target sensor
    front_xyz = np.linalg.inv(target_trans) @ tgt_xyz
    # concate transformed xyz with other data
    data_comp = np.concatenate((front_xyz[:3],data[3:]),axis=0)
    
    return data_comp
    
def concat_radars(base_ts_utc, front_files,left_files,right_files,cor_idx,cor_ts,lidar_ts, ego_pose,pose_ts,sensor_T,save_path):
    
    
    data_len = len(cor_ts["front_ts"])
    for i in tqdm(range(data_len)):
        f_data = pd.read_table(front_files[cor_idx["front_idx"][i]], sep=",", header=None).values[1:,1:].T.astype(np.float32)
        #l_data = pd.read_table(left_files[cor_idx["left_idx"][i]], sep=",", header=None).values[1:,1:].T.astype(np.float32)
        #r_data = pd.read_table(right_files[cor_idx["right_idx"][i]], sep=",", header=None).values[1:,1:].T.astype(np.float32)
        f_comp = radar_temp_comp(f_data, cor_ts["front_ts"][i], lidar_ts[i], ego_pose,pose_ts, sensor_T['radar_front'],sensor_T['radar_front'])
        #l_comp = radar_temp_comp(l_data, cor_ts["left_ts"][i], lidar_ts[i], ego_pose,pose_ts, sensor_T['radar_left'],sensor_T['radar_front'])
        #r_comp = radar_temp_comp(r_data, cor_ts["right_ts"][i], lidar_ts[i], ego_pose,pose_ts, sensor_T['radar_right'],sensor_T['radar_front'])
        #concat_comp = np.concatenate((f_comp,l_comp,r_comp),axis=1)
        concat_comp = f_comp
        path = os.path.join(save_path, str(round(lidar_ts[i]*1e3)+base_ts_utc).zfill(13) + ".csv")
        data = pd.DataFrame(concat_comp.T)
        data.to_csv(path)

def save_sync_gt(save_gt,gt_cor_idx,base_ts_utc,lidar_ts,gt_files):
    
    data_len = len(lidar_ts)
    for i in tqdm(range(data_len)):
        ts = lidar_ts[i]
        gt_ts = int(ts*1e3+base_ts_utc)
        gt_obj = np.loadtxt(gt_files[gt_cor_idx[i]], ndmin=2)  # ndmin=2 handles single-line input
        gt_fname = str(gt_ts).zfill(13) + ".csv"
        full_fname = osp(save_gt, gt_fname)
        np.savetxt(full_fname, gt_obj, fmt="%s")        

def _get_lidar_path(load_dir):
    lidar_path = os.path.join(load_dir, 'input', 'lidar')
    path_options = os.listdir(lidar_path)
    if len(path_options) != 1:
        raise ValueError(f'"{lidar_path}" contains {path_options} files. Expecting exactly 1 subdirectory.')
    return os.path.join(lidar_path, path_options[0])

def pcl_sync(load_dir, save_dir, base_ts: dict):
    # save path of concatenated radar point clouds
    save_path = os.path.join(save_dir, 'inhouse_format', "radar")
    # save path of sync gt files
    save_gt = os.path.join(save_dir, 'inhouse_format', "gt")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_gt):
        os.mkdir(save_gt)
    # initialize the lidar and radar timestamps and files
    radar_path = os.path.join(save_dir, 'inhouse_format', "radar_raw")
    radar_paths = {'front': radar_path, 
                  'left': radar_path,
                  'right': radar_path
                  }
    lidar_path = _get_lidar_path(load_dir)
    pose_path = os.path.join(load_dir, "output", "online", "sample", "gnssimu-sample-v6@2.csv")
    gt_path = os.path.join(save_dir, 'inhouse_format', "gt_raw")
    
    base_ts_utc = base_ts['utc'] ## utc base timestamp, for lidar and robosense
    base_ts_local = base_ts['local'] ## local base timestamp, for radar and pose

    front_files = sorted(glob(os.path.join(radar_paths["front"], '*.csv')))
    left_files = sorted(glob(os.path.join(radar_paths["left"], '*.csv')))
    right_files = sorted(glob(os.path.join(radar_paths["right"], '*.csv')))
    lidar_files = sorted(glob(os.path.join(lidar_path, '*.pcd')) )
    gt_files = sorted(glob(os.path.join(gt_path, '*.csv')))
    
    radar_ts = {'front_ts': [], "left_ts" : [], "right_ts": []}
    lidar_ts = []
    gt_ts = []
    gt_bias = 18000 + 100 + 25
    
    
    for file in front_files:
        radar_ts["front_ts"].append((float(file.split('/')[-1].split('.')[0])-base_ts_local)/1e3)
    for file in left_files:
        radar_ts["left_ts"].append((float(file.split('/')[-1].split('.')[0])-base_ts_local)/1e3)
    for file in right_files:
        radar_ts["right_ts"].append((float(file.split('/')[-1].split('.')[0])-base_ts_local)/1e3)
    for file in lidar_files:
        lidar_ts.append((float(file.split('/')[-1].split('.')[0])-base_ts_utc)/1e3)
    for file in gt_files:
        gt_ts.append((float(file.split('/')[-1].split('.')[0])-base_ts_utc-gt_bias)/1e3)
    
    gt_ts = np.array(gt_ts)
        
    for key in radar_ts:
        radar_ts[key] = np.array(radar_ts[key])
    
    # get transformation matrix of sensors from extrinsics
    radar_front_ext = np.array([0.06, -0.2, 0.7, -3.5, 2, 180])
    radar_left_ext = np.array([0.06, -0.2, 0.7, -3.5, 2, 180])
    radar_right_ext = np.array([0.06, -0.2, 0.7, -3.5, 2, 180])
    lidar_ext = np.array([0, 0, -0.3, -2.5, 0, 0])
    
    radar_front_T = get_matrix_from_ext(radar_front_ext)
    radar_left_T = get_matrix_from_ext(radar_left_ext)
    radar_right_T = get_matrix_from_ext(radar_right_ext)
    lidar_T = get_matrix_from_ext(lidar_ext)
    
    sensor_T = {"radar_front": radar_front_T, "radar_left": radar_left_T,
                "radar_right": radar_right_T, "lidar": lidar_T}
    
    # get the high-frenquency vehicle pose data (100Hz)
    ego_pose, pose_ts = get_interpolate_pose(pose_path,scale=10)
    pose_ts = (pose_ts-base_ts_local)/1e3
    # get the corresponding timestamp (sync with lidar) of gt
    gt_cor_idx, gt_cor_ts = get_cor_gt_idx(lidar_ts, gt_ts)
    # get the corresponding timestamp (sync with lidar) of each radar sensor 
    cor_idx, cor_ts = get_cor_radar_idx(lidar_ts, radar_ts)
    
    # concat multi-radar data with temporal compensation using pose data
    concat_radars(base_ts_utc, front_files,left_files,right_files,cor_idx,cor_ts, lidar_ts, ego_pose,pose_ts,sensor_T,save_path)
    # save sync gt to new folder, no need pose data, just to match gt and lidar
    save_sync_gt(save_gt,gt_cor_idx,base_ts_utc,lidar_ts,gt_files)
    
def main():
    parser = argparse.ArgumentParser(description='Sync radar and LiDAR pointclouds')
    parser.add_argument('data_root', help='path to root of directory containing unprocessed data')
    args = parser.parse_args()
    pcl_sync(args.data_root)
    

if __name__ == '__main__':
    main()