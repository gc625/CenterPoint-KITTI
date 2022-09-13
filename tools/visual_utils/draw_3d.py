# %%
import matplotlib.pyplot as plt
import pickle 
from pathlib import Path as P
from matplotlib.patches import Rectangle as Rec
import numpy as np
from pcdet.utils import calibration_kitti
from vod.visualization.settings import label_color_palette_2d
from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameLabels, FrameTransformMatrix
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# %%
def get_pred_dict(dt_file):
    '''
    reads results.pkl file

    returns dictionary with str(frame_id) as key, and list of strings, 
    where each string is a predicted box in kitti format.
    '''
    dt_annos = []

    # load detection dict
    # print(dt_file)
    with open(dt_file, 'rb') as f:
        infos = pickle.load(f)
        dt_annos.extend(infos)      
    # print(dt_annos)
    labels_dict = {}
    for j in range(len(dt_annos)):
        labels = []
        curr = dt_annos[j]
        frame_id = curr['frame_id']
        
        # no predicted 
        if len(dt_annos[j]['name']) == 0: 
            labels += []
        
        else:
            for i in range(len(dt_annos[j]['name'])):       
                # extract the relevant info and format it 
                line = [str(curr[x][i]) if not isinstance(curr[x][i],np.ndarray) else [y for y in curr[x][i]]  for x in list(curr.keys())[:-2]]
                flat = [str(num) for item in line for num in (item if isinstance(item, list) else (item,))]
                
                # L,H,W -> H,W,L 
                flat[9],flat[10] = flat[10],flat[9]
                flat[8],flat[10] = flat[10],flat[8]
                
                labels += [" ".join(flat)]

        labels_dict[frame_id] = labels
    
    return dt_annos, labels_dict

def vod_to_o3d(vod_bbx,frame_id,is_radar,is_test=False):

    box_list = []
    modality = 'radar' if is_radar else 'lidar'
    split = 'testing' if is_test else 'training'
    calib_path = "../../data/vod_%s/%s/calib/%s.txt"%(modality, split, frame_id)
    calib = calibration_kitti.Calibration(calib_path)

    for box in vod_bbx:
        xyz = np.array([[box['x'],box['y'],box['z']]])
        loc_lidar = calib.rect_to_lidar(xyz)
        new_xyz = loc_lidar[0]
        
        angle = -(box['rotation']+ np.pi / 2) 
        angle = np.array([0, 0, angle])
        rot_matrix = R.from_euler('XYZ', angle).as_matrix()
        extent = np.array([[box['l'],box['w'],box['h']]])

        obbx = o3d.geometry.OrientedBoundingBox(new_xyz.T, rot_matrix, extent.T)

        box_list += [obbx]

    return obbx

def process_dt(pred_dict,vod_data_path,frame_id,is_radar):
    frame_ids = list(pred_dict.keys())
    kitti_locations = KittiLocations(root_dir=vod_data_path,
                                output_dir="output/",
                                frame_set_path="",
                                pred_dir="",
                                )
    frame_data = FrameDataLoader(kitti_locations,
                             frame_ids[frame_id],pred_dict)

    radar_points = frame_data.radar_data
    lidar_points = frame_data.lidar_data

    vod_preds = FrameLabels(frame_data.get_predictions()).labels_dict
    vod_labels = FrameLabels(frame_data.get_labels()).labels_dict

    o3d_predictions = vod_to_o3d(vod_preds,frame_ids[frame_id],is_radar)
    o3d_labels = vod_to_o3d(vod_labels,frame_ids[frame_id],is_radar)

    return radar_points,lidar_points,o3d_predictions,o3d_labels

# %%
def main():

    root_path = P('/root/gabriel/code/parent/CenterPoint-KITTI/output/IA-SSD-vod-radar/iassd_best_aug_new/eval/best_epoch_checkpoint')
    dt_path = str(root_path / 'result.pkl')
    dt_annos, pred_dict = get_pred_dict(dt_path)

    # gt_path = str(root_path / 'radar_preds.pkl')
    # print(pred_dict)
    vod_data_path = '/mnt/12T/public/view_of_delft'
    radar_points,lidar_points,o3d_predictions,o3d_labels = process_dt(
                                                                pred_dict,
                                                                vod_data_path,
                                                                333,
                                                                is_radar=True)
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(radar_points[:,0:3])
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([o3d_labels])

    o3d.io.write_point_cloud("test.pcd", pcd)
    
    

    


#%%
if __name__ == "__main__":
    main()
# %%
