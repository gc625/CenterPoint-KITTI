import os
from turtle import color
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path as P
import pickle
from matplotlib.patches import Rectangle as Rec
import cv2

root_dir = P('/root/dj/code/CenterPoint-KITTI/output/IA-SSD/initial_ctr256/eval/eval_with_train')

frame_ids = root_dir / 'frame_ids.txt'
frame_ids = np.loadtxt(frame_ids, delimiter=',', dtype=str)
dt_annos = []
gt_annos = []

dt_file = root_dir / 'result.pkl'
gt_file = root_dir / 'gt.pkl'

with open(dt_file, 'rb') as f:
    infos = pickle.load(f)
    dt_annos.extend(infos)

with open(gt_file, 'rb') as f:
    infos = pickle.load(f)
    gt_annos.extend(infos)

radar_pcd_path = P('/root/data/public/shangqi/radar_strict/kitti_format/radar')

def getpcd(fname):
    return np.fromfile(str(fname), dtype=np.float32).reshape(-1, 6)

def anno2plt(anno, color, lw):
    dim = anno['dimensions']
    loc = anno['location']
    angle = anno['rotation_y'] * 180 / 3.14
    rec_list = []
    for idx in range(dim.shape[0]):
        x, y, _ = loc[idx]
        w, l, _ = dim[idx]
        ax = x - w/2
        ay = y - l/2
        ang = -angle[idx]
        rec_list += [Rec((ax, ay), w, l, ang, fill=False, color=color,lw=lw)]
    return rec_list

def bin2pts(fname):
    pts = getpcd(fname)
    pts_xy = pts[:,:2]
    return pts_xy

def drawBEV(pcd_fname, dt_anno, gt_anno, save_name):
    pts_2d = bin2pts(pcd_fname)
    dt_rec_list = anno2plt(dt_anno, 'red', lw=3)
    gt_rec_list = anno2plt(gt_anno, 'green', lw=2)
    x = pts_2d[:,0]
    y = pts_2d[:,1]
    plt.scatter(x, y, c='black', s=0.1)
    plt.xlim(-0,75)
    plt.ylim(-30,30)
    for dt_rec in dt_rec_list:
        plt.gca().add_patch(dt_rec)

    for gt_rec in gt_rec_list:
        plt.gca().add_patch(gt_rec)
    plt.savefig(save_name)

# test
# cur_id = frame_ids[0]
# cur_bin = str(radar_pcd_path/ (cur_id+'.bin'))
# dt_anno = dt_annos[0]
# gt_anno = gt_annos[0]
save_dir = root_dir/'bev_img'
save_dir.mkdir(exist_ok=True)

plt.figure()
from tqdm import tqdm
for idx in tqdm(range(len(frame_ids))):
    cur_id = frame_ids[idx]
    cur_bin = str(radar_pcd_path/(cur_id + '.bin'))
    dt_anno = dt_annos[idx]
    gt_anno = gt_annos[idx]
    save_name = str(save_dir/(cur_id + '.png'))
    drawBEV(cur_bin, dt_anno=dt_anno, gt_anno=gt_anno, save_name=save_name)
    plt.clf()

