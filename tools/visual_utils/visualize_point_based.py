# %%
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path as P
from matplotlib.patches import Rectangle as Rec
import numpy as np
from pcdet.utils import calibration_kitti
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm
from vod.visualization.settings import label_color_palette_2d



def transform_anno(loc, frame_id):
    x,y, z = loc[0], loc[1], loc[2]
    calib_path = "/root/dj/code/CenterPoint-KITTI/data/vod_radar/training/calib/{0}.txt".format(frame_id)
    calib = calibration_kitti.Calibration(calib_path)
    loc = np.array([[x,y,z]])
    loc_lidar = calib.rect_to_lidar(loc)
    x,y,z = loc_lidar[0]
    return x,y,z


def get_rot_corner(x,y,l,w,a):

    s,c = np.sin(a),np.cos(a)

    corner_x = x - l/2
    corner_y = y - w/2

    corner_x -= x
    corner_y -= y

    new_corner_x = corner_x*c - corner_y*s 
    new_corner_y = corner_x*s + corner_y*c

    new_corner_x += x
    new_corner_y += y

    return new_corner_x,new_corner_y




def anno2plt(anno, color_dict, lw, frame_id, xz=False):
    dim = anno['dimensions']
    loc = anno['location']
    # angle = anno['rotation_y'] * 180 / 3.14
    angle = -(anno['rotation_y']+ np.pi / 2) 
    rec_list = []
    cls = anno['name']
    for idx in range(dim.shape[0]):
        name = cls[idx]
        # print(name)
        if name not in color_dict:
            color = 'gray'
        else:
            color = color_dict[name]
            # print(color)
    
        if xz:

            x, _, y = transform_anno(loc[idx], frame_id)
            # w, _, l = dim[idx]
            l, w, _ = dim[idx]  # 
            ang = -angle[idx]* 0
        else:
            # print(loc[idx])
            x, y, z = transform_anno(loc[idx], frame_id)
            # print(x,y)
            
            ### X -> LENGTH
            ### Y -> WIDTH 
            ### Z -> HEIGHT, not used. 
            # x,y,z = loc[idx]
            # w, l, _ = dim[idx]
            # print(dim[idx]) 
            l, h, w  = dim[idx] # <-- SHOULD BE CORRECT? 
            ang = angle[idx]
            # ang = 0
            # print(l,w,ang)
            # print("="*40)

            ax,ay = get_rot_corner(x,y,l,w,ang)
            ang = ang * 180 / 3.14
            # ax = x - (l/4)
            # ay = y - (w/4)

        rec_list += [Rec((ax, ay), l, w, ang, fill=False, color=color,lw=lw)]
    return rec_list



def drawBEV(ax, pts, centers, annos, color_dict, frame_id, ax_title):


    # 3. draw bbx
    try:
        rec_list = anno2plt(annos, color_dict, 2, frame_id=frame_id, xz=False)
    except:
        rec_list = anno2plt(annos[0], color_dict, 2, frame_id=frame_id, xz=False)
    
    for rec in rec_list:
        ax.add_patch(rec)
    # 1. draw original points if exist
    if pts is not None:
        x = pts[:, 1]
        y = pts[:, 2]
        ax.scatter(x, y, c='black', s=0.1)
    # 2. overlay centers
    if centers is not None:
        cx = centers[:, 1]
        cy = centers[:, 2]
        ax.scatter(cx, cy, c='red', s=0.1)

    legend_elements = [Patch(facecolor='white', edgecolor=v, label=k) for i, (k, v) in enumerate(color_dict.items())]
    legend_elements += [Line2D([0], [0], marker='o', color='w', label='FG points',
                          markerfacecolor='r', markersize=10)]
    ax.legend(handles=legend_elements, loc=1)
    ax.set_title(ax_title)


def draw_rectangle(ax, anno, color_dict, xz=False):
    recs = anno2plt(anno, color_dict, lw=2, xz=xz)
    for rec in recs:
        ax.add_patch(rec)


if __name__ == '__main__':
    
    cls_name = ['Car','Pedestrian', 'Cyclist', 'Others']

    
    
    # trans-ssd
    root_path = P('/root/dj/code/CenterPoint-KITTI/output/IA-SSD-vod-radar/iassd_128_vcomp/eval/checkpoint_epoch_100')
    # ia-ssd
    # root_path = P('/root/dj/code/CenterPoint-KITTI/output/IA-SSD-vod-radar/iassd_128_all/eval/checkpoint_epoch_100')
    
    # root_path = P('/root/dj/code/CenterPoint-KITTI/output/pointpillar_vod_lidar/filter5/eval/eval_with_train/epoch_80/val')

    color_dict = {}

    gt_save_dir = root_path / 'GT_all_bev'
    pred_save_dir = root_path / 'pred_bev'
    gt_save_dir.mkdir(exist_ok=True)
    pred_save_dir.mkdir(exist_ok=True)

    for i, v in enumerate(cls_name):
        color_dict[v] = label_color_palette_2d[v]
    # load gt
    with open(str(root_path / 'gt.pkl'), 'rb') as f:
        gt = pickle.load(f)

    # load det
    with open(str(root_path / 'dt.pkl'), 'rb') as f:
        dt = pickle.load(f)

    # load centers
    with open(str(root_path / 'centers.pkl'), 'rb') as f:
        centers = pickle.load(f)

    # load centers_origin
    with open(str(root_path / 'centers_origin.pkl'), 'rb') as f:
        centers_origin = pickle.load(f)

    # load input points (after sampling to 512 points)
    with open(str(root_path / 'points.pkl'), 'rb') as f:
        points = pickle.load(f)
    
    data_dict = {}
    def load_data(name):
        with open(str(root_path / (name + '.pkl')), 'rb') as f:
            data = pickle.load(f)
        return data
    # save_name_list = ('centers', 'centers_origin', 'points', 'match', 'lidar_center', 'lidar_preds', 'radar_preds', 'radar_label')
    # for name in save_name_list:
    #     data_dict[name] = load_data(name)
    # test_recs = anno2plt(dt['00000'][0], color_dict, 2, 0)

    # print(type(points))

    plt.rcParams['figure.dpi'] = 150
    # fig, ax = plt.subplots()
    plt.xlim(-0,75)
    plt.ylim(-30,30)
    ax = plt.gca()

    # 1. image with center origin and gt annos
    ids = list(gt.keys())
    for id in tqdm(ids):
        img_fname = str(id) + '.png'
        
        # print(id,points[id])
        # draw gt
        drawBEV(ax, points[id], centers_origin[id], gt[id], color_dict, id, 'GT')
        gt_img_full_fname = str(gt_save_dir / img_fname)
        plt.xlim(-0,75)
        plt.ylim(-30,30)
        plt.savefig(gt_img_full_fname)
        ax.clear()
        
        # draw pred
        # drawBEV(ax, None, centers_origin[id], dt[id], color_dict, id, 'pred')
        # pred_img_full_fname = str(pred_save_dir / img_fname)
        # plt.xlim(-0,75)
        # plt.ylim(-30,30)
        # plt.savefig(pred_img_full_fname)
        # ax.clear()
        
    
    # plt.show()


# %%
