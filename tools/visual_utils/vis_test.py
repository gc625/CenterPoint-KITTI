from posixpath import abspath
import numpy as np
from pathlib import Path as P
import pickle
# from visualize_utils import make_vid
import cv2
from vod.visualization.settings import label_color_palette_2d
from matplotlib.lines import Line2D

from visualize_point_based import drawBEV
import matplotlib.pyplot as plt
from tqdm import tqdm

from glob import glob
import argparse

def saveODImgs(frame_ids, anno, data_path, img_path, color_dict, is_radar=True, title='pred'):
    print('=================== drawing images ===================')
    plt.rcParams['figure.dpi'] = 150
    for fid in tqdm(frame_ids):
        pcd_fname = data_path / (fid + '.bin')
        vis_pcd = get_radar(pcd_fname) if is_radar else get_lidar(pcd_fname)
        vis_pcd = pcd_formating(vis_pcd)
        ax = plt.gca()
        drawBEV(ax, vis_pcd, None, anno[fid], color_dict, fid, title)
        plt.xlim(-0,75)
        plt.ylim(-30,30)
        img_fname = img_path / (fid + '.png')
        plt.savefig(str(img_fname))
        plt.cla()

def get_radar(fname):
    assert fname.exists()
    radar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 7)
    return radar_point_cloud

def get_lidar(fname):
    assert fname.exists()
    radar_point_cloud = np.fromfile(str(fname), dtype=np.float32).reshape(-1, 4)
    return radar_point_cloud

def pcd_formating(pcd):
    num_pts = pcd.shape[0]
    zeros_pad = np.zeros([num_pts, 1])
    final_pcd = np.concatenate((zeros_pad, pcd), axis=1)
    return final_pcd

def make_vid(imgs, vid_fname, fps=15):
    print('=================== making videos ===================')
    out = None
    for fname in tqdm(imgs):
        i = cv2.imread(fname)
        if out is None:
            h, w, _ = i.shape
            size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(vid_fname), fourcc, fps, size)
            
        out.write(i)
    out.release()
    

if __name__ == '__main__':

    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    path_dict = {
        'CFAR-radar':'output/IA-SSD-GAN-vod-aug/radar48001_512all/eval/best_epoch_checkpoint',
        'radar-rcsv':'',
        'radar-rcs':'',
        'radar-v':'',
        'radar':'',
        'lidar-i':'',
        'lidar':'',
        'CFAR-lidar-rcsv':'',
        'CFAR-lidar-rcs':'',
        'CFAR-lidar-v':'',
        'CFAR-lidar':''
    }

    draw_gt = False
    is_radar = True
    exp_tag = ''
    modality = 'radar' if is_radar else 'lidar'
    tag = 'CFAR-radar'

    result_path = base_path / path_dict[tag]
    data_path = base_path/ '/CenterPoint-KITTI/data/vod_%s/training/velodyne'%modality 
    

    dt_img_path = result_path/'dt_img'
    dt_img_path.mkdir(exist_ok=True)

    data_ids = np.loadtxt(str(result_path / 'frame_ids.txt'), delimiter=',', dtype=str)[:-1]

    with open(str(result_path / 'gt.pkl'), 'rb') as f:
        gt = pickle.load(f)

    # load det
    with open(str(result_path / 'dt.pkl'), 'rb') as f:
        dt = pickle.load(f)

    keys = list(gt.keys())
    cls_name = ['Car','Pedestrian', 'Cyclist', 'Others']
    color_dict = {}
    for i, v in enumerate(cls_name):
        color_dict[v] = label_color_palette_2d[v]

    saveODImgs(data_ids, dt, data_path, dt_img_path, \
    color_dict, is_radar=True, title='pred CFAR')

    dt_imgs = sorted(glob(str(dt_img_path/'*.png')))

    make_vid(dt_imgs, result_path/'dt.mp4', fps=10)


    if draw_gt:
        gt_img_path = result_path/'gt_img'
        gt_img_path.mkdir(exist_ok=True)
        saveODImgs(data_ids, gt, data_path, gt_img_path, \
            color_dict, is_radar=True, title='gt')
        gt_imgs = sorted(glob(str(gt_img_path/'*.png')))
        make_vid(gt_imgs, result_path/'gt.mp4', fps=10)