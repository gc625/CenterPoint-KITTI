import io as sysio
from mmap import MAP_ANON

import numba
import numpy as np
import pickle
from pathlib import Path as P
from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
from pcdet.datasets.kitti.kitti_object_eval_python.eval import clean_data,_prepare_data,eval_class,get_mAP,get_mAP_R40
from pcdet.datasets.kitti.kitti_object_eval_python.kitti_common import get_label_annos
from vod.visualization.settings import label_color_palette_2d
import matplotlib.pyplot as plt


def draw_iou_results(iou_thresholds,
                car_AP,
                pedestrian_AP,
                cyclist_AP,
                result_dir,
                is_distance=False,
                fig_name=None,
                xlabel=None):
    fig, ax = plt.subplots(1)

    car_color = label_color_palette_2d['Car']
    ped_color = label_color_palette_2d['Pedestrian']
    cyclist_color = label_color_palette_2d['Cyclist']

    mAP = np.mean([car_AP,pedestrian_AP,cyclist_AP],axis=0)
    
    ax.scatter(iou_thresholds,car_AP,color=car_color,clip_on=False)
    ax.scatter(iou_thresholds,pedestrian_AP,color=ped_color,clip_on=False)
    ax.scatter(iou_thresholds,cyclist_AP,color=cyclist_color,clip_on=False)
    ax.scatter(iou_thresholds,mAP,color='black',clip_on=False)
    
    ax.plot(iou_thresholds,car_AP,color=car_color,label='Car')
    ax.plot(iou_thresholds,pedestrian_AP,color=ped_color,label='Pedestrian')
    ax.plot(iou_thresholds,cyclist_AP,color=cyclist_color,label='Cyclist')
    ax.plot(iou_thresholds,mAP,color='black',label='mAP')

    if xlabel is not None:
        ax.set_xlabel(xlabel) 
    else:
        ax.set_xlabel('IoU threshold (3D)') 
    ax.set_ylabel('AP (3D IoU)')

    ax.set_yticks(np.arange(0,110,10))
    
    if not is_distance:
        ax.set_xticks(np.arange(0,1,0.1))
    
    ax.grid(axis = 'y')
    
    ax.legend()

    for label in ax.get_yticklabels()[1::2]:
        label.set_visible(False)
        plt.xlim(xmin=0) 

    plt.ylim(ymin=0,ymax=100)

    if fig_name is not None:
        fig_path = result_dir / (fig_name+'.png')
    else:
        fig_path = result_dir / 'iou_threshold.png'

    fig.savefig(fig_path)


def draw_one(x,car,ped,cyclist,tag,ax,linestyle):
    car_color = label_color_palette_2d['Car']
    ped_color = label_color_palette_2d['Pedestrian']
    cyclist_color = label_color_palette_2d['Cyclist']
    mAP = np.mean([car,ped,cyclist],axis=0)

    ax.scatter(x,car,color=car_color,clip_on=False)
    ax.scatter(x,ped,color=ped_color,clip_on=False)
    ax.scatter(x,cyclist,color=cyclist_color,clip_on=False)
    ax.scatter(x,mAP,color='black',clip_on=False)

    ax.plot(x,car,color=car_color,label=f'{tag:} Car',linestyle=linestyle)
    ax.plot(x,ped,color=ped_color,label=f'{tag:} Pedestrian',linestyle=linestyle)
    ax.plot(x,cyclist,color=cyclist_color,label=f'{tag:} Cyclist',linestyle=linestyle)
    ax.plot(x,mAP,color='black',label=f'{tag:} mAP',linestyle=linestyle)

def compare(x,results_a,results_b,tag_a,tag_b,result_dir):


    
    fig, ax = plt.subplots(1)
    
    car_a = results_a['Car']['mAP_3d']
    pedestrian_a = results_a['Pedestrian']['mAP_3d']
    cyclist_a = results_a['Cyclist']['mAP_3d']

    car_b = results_b['Car']['mAP_3d']
    pedestrian_b = results_b['Pedestrian']['mAP_3d']
    cyclist_b = results_b['Cyclist']['mAP_3d']

    draw_one(x,car_a,pedestrian_a,cyclist_a,tag_a,ax,linestyle="solid")
    draw_one(x,car_b,pedestrian_b,cyclist_b,tag_b,ax,linestyle="dashed")

    ax.grid(axis = 'y')
    plt.ylim(ymin=0,ymax=100)
    ax.legend()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - box.height * 0.01,
                 box.width, box.height * 0.9])

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,2,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # ax.legend(handles, labels)                 
    ax.legend(handles, labels, bbox_to_anchor=(-0.05, 1.05),loc='lower left',
          fancybox=True, shadow=True, ncol=4, prop={'size': 6})

    ax.set_xlabel('Distance from ego-vehicle') 
    ax.set_ylabel('AP (3D bounding box)')

    fig_path = result_dir / (f'{tag_a}_and_{tag_b}'+'.png')
    
    fig.savefig(fig_path)


def main():
    path_dict = {
        'CFAR_radar':'output/IA-SSD-GAN-vod-aug/radar48001_512all/eval/best_epoch_checkpoint',
        'radar_rcsv':'output/IA-SSD-vod-radar/iassd_best_aug_new/eval/best_epoch_checkpoint',
        'radar_rcs':'output/IA-SSD-vod-radar/iassd_rcs/eval/best_epoch_checkpoint',
        'radar_v':'output/IA-SSD-vod-radar/iassd_vcomp_only/eval/best_epoch_checkpoint',
        'radar':'output/IA-SSD-vod-radar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'lidar_i':'output/IA-SSD-vod-lidar/all_cls/eval/checkpoint_epoch_80',
        'lidar':'output/IA-SSD-vod-lidar-block-feature/only_xyz/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcsv':'output/IA-SSD-GAN-vod-aug-lidar/to_lidar_5_feat/eval/best_epoch_checkpoint',
        'CFAR_lidar_rcs':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_rcs_only/eval/best_epoch_checkpoint',
        'CFAR_lidar_v':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_vcomp_only/eval/best_epoch_checkpoint',
        'CFAR_lidar':'output/IA-SSD-GAN-vod-aug-lidar/cls80_attach_xyz_only/eval/best_epoch_checkpoint'
    }

    tag_a = 'CFAR_radar'
    tag_b = 'radar_rcsv'
    
    abs_path = P(__file__).parent.resolve()
    base_path = abs_path.parents[1]
    result_path_a = base_path / path_dict[tag_a]
    result_path_b = base_path / path_dict[tag_b]

    save_path = base_path /'output' / 'vod_vis' / 'comparisons'
    save_path.mkdir(parents=True,exist_ok=True)

    with open(result_path_a / 'all_iou_results.pkl', 'rb') as f:
        iou_results_a = pickle.load(f)

    with open(result_path_b / 'all_iou_results.pkl', 'rb') as f:
        iou_results_b = pickle.load(f)

    with open(result_path_a / 'distance_results.pkl', 'rb') as f:
        distance_results_a = pickle.load(f)

    with open(result_path_b / 'distance_results.pkl', 'rb') as f:
        distance_results_b = pickle.load(f)
    

    distance_range = distance_results_a['distances']

    compare(
        distance_range,
        distance_results_a,
        distance_results_b,
        tag_a,
        tag_b,
        save_path
        )

    
    
    
    


if __name__ == "__main__":
    main()




