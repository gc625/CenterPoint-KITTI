# Modify converty kitti format to real kitti format
import glob
import os
from tabnanny import check
from typing import no_type_check
import numpy as np
from scipy.spatial.transform import Rotation as R

def write_split_file(path, timestamps):
    with open(path, 'w') as f:
        f.writelines(ts.split('.')[0] + '\n' for ts in timestamps)

def check_if_empty_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        return True
    return False

def get_matrix_from_ext(ext):
    rot = R.from_euler('ZYX', ext[3:], degrees=True)
    rot_m = rot.as_matrix()
    x, y, z = ext[:3]
    tr = np.eye(4)
    tr[:3,:3] = rot_m
    tr[:3, 3] = np.array([x, y, z]).T
    return tr

def create_label_link(source_path, inhouse_source_path, target_path, test_set_imageset, train_set_imageset, val_set_imageset):
    source_label_path = os.path.join(source_path, 'label')
    target_label_path = os.path.join(target_path,'training', 'label_2')
    if not os.path.exists(target_label_path):
        os.makedirs(target_label_path)

    no_found = 0
    empty = 0

    test_label_seq = []
    with open(test_set_imageset,'r') as f:
        test_labels_files = f.read().splitlines()
    for file in test_labels_files:
        label_tgt = file+'.txt'

        if not os.path.isfile(os.path.join(inhouse_source_path,label_tgt)):
            print(f'[TEST]: not found {label_tgt}, using shangqi gt')
            no_found += 1
            continue
        
        if check_if_empty_label(os.path.join(inhouse_source_path,label_tgt)):
            print(f'[TEST]: {label_tgt} is empty, using shangqi gt')
            empty += 1
            continue

        os.symlink(
            os.path.join(inhouse_source_path,label_tgt),
            os.path.join(target_label_path,label_tgt)
        )
        test_label_seq.append(label_tgt)
    assert(len(test_label_seq) == len(glob.glob(target_label_path + '/*.txt'))), 'test labels doesnt match'

    train_label_seq = []
    with open(train_set_imageset,'r') as f:
        train_labels_files = f.read().splitlines()
    for file in train_labels_files:
        label_tgt = file+'.txt'

        if not os.path.isfile(os.path.join(source_label_path,label_tgt)):
            print(f'[TRAIN]: not found {label_tgt}, skip')
            no_found += 1
            continue

        if check_if_empty_label(os.path.join(source_label_path,label_tgt)):
            print(f'[TRAIN]: {label_tgt} is empty, skip')
            empty += 1
            continue
        
        os.symlink(
            os.path.join(source_label_path,label_tgt),
            os.path.join(target_label_path,label_tgt)
        )
        train_label_seq.append(label_tgt)

    val_label_seq = []
    with open(val_set_imageset,'r') as f:
        val_labels_files = f.read().splitlines()
    for file in val_labels_files:
        label_tgt = file+'.txt'

        if not os.path.isfile(os.path.join(source_label_path,label_tgt)):
            print(f'[VAL]: not found {label_tgt}, skip')
            no_found += 1
            continue

        if check_if_empty_label(os.path.join(source_label_path,label_tgt)):
            print(f'[VAL]: {label_tgt} is empty, skip')
            empty += 1
            continue
        
        os.symlink(
            os.path.join(source_label_path,label_tgt),
            os.path.join(target_label_path,label_tgt)
        )
        val_label_seq.append(label_tgt)
    
    print('Found valid labels: ', len(train_label_seq), len(val_label_seq), len(test_label_seq))
    print('Not found labels: ', no_found)
    print('Empty labels: ', empty)
    return train_label_seq, val_label_seq, test_label_seq
    

def transform_lidar(input_path, output_path, lidar_tr):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    lidar_files = glob.glob(input_path + '/*.bin')
    for lidar_file in lidar_files:
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        pc = lidar[:, :3]
        pc = np.hstack((pc, np.ones((pc.shape[0], 1),dtype=np.float32)))
        pc = np.dot(pc, lidar_tr.T)
        pc.astype(np.float32)
        lidar[:, :3] = pc[:, :3]
        lidar.tofile(os.path.join(output_path, lidar_file.split('/')[-1]))


def create_lidar(source_path, in_house_label_path, target_path):
    lidar_path = os.path.join(target_path, 'lidar')
    if not os.path.exists(lidar_path):
        os.makedirs(lidar_path)
    training_path = os.path.join(lidar_path, 'training')
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    testing_path = os.path.join(lidar_path, 'testing')
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    imageset_path = os.path.join(lidar_path, 'ImageSets')
    if not os.path.exists(imageset_path):
        os.makedirs(imageset_path)

    train_set, test_set, val_set = os.path.join(source_path,'ImageSets/train.txt'),os.path.join(source_path,'ImageSets/test.txt'), os.path.join(source_path,'ImageSets/val.txt')
    
    os.symlink(
        os.path.join(source_path, 'calib'),
        os.path.join(training_path, 'calib')
    )

    os.symlink(
        os.path.join(source_path, 'calib'),
        os.path.join(testing_path, 'calib')
    )

    lidar_ext = [-2.5, 0, 2.03, 4.9, -1.5, 0]
    lidar_tr = get_matrix_from_ext(lidar_ext)
    transform_lidar(os.path.join(source_path, 'lidar'), os.path.join(training_path, 'velodyne'), lidar_tr)

    os.symlink(
        os.path.join(training_path, 'velodyne'),
        os.path.join(testing_path, 'velodyne')
    )

    train_label_seq, val_label_seq, test_label_seq = create_label_link(source_path, in_house_label_path, lidar_path, test_set, train_set, val_set)
    write_split_file(os.path.join(imageset_path, 'train.txt'), train_label_seq)
    write_split_file(os.path.join(imageset_path, 'val.txt'), val_label_seq)
    write_split_file(os.path.join(imageset_path, 'trainval.txt'), train_label_seq + val_label_seq)
    write_split_file(os.path.join(imageset_path, 'test.txt'), test_label_seq)

    os.symlink(
        os.path.join(training_path, 'label_2'),
        os.path.join(testing_path, 'label_2')
    )

def create_radar(source_path, target_path):
    radar_path = os.path.join(target_path, 'radar')
    if not os.path.exists(radar_path):
        os.makedirs(radar_path)
    training_path = os.path.join(radar_path, 'training')
    if not os.path.exists(training_path):
        os.makedirs(training_path)
    testing_path = os.path.join(radar_path, 'testing')
    if not os.path.exists(testing_path):
        os.makedirs(testing_path)
    
    os.symlink(
        os.path.join(target_path,'lidar', 'ImageSets'),
        os.path.join(radar_path, 'ImageSets')
    )

    os.symlink(
        os.path.join(source_path, 'calib'),
        os.path.join(training_path, 'calib')
    )

    os.symlink(
        os.path.join(source_path, 'calib'),
        os.path.join(testing_path, 'calib')
    )

    os.symlink(
        os.path.join(source_path, 'radar'),
        os.path.join(training_path, 'velodyne')
    )

    os.symlink(
        os.path.join(source_path, 'radar'),
        os.path.join(testing_path, 'velodyne')
    )

    os.symlink(
        os.path.join(target_path,'lidar', 'training','label_2'),
        os.path.join(training_path, 'label_2')
    )

    os.symlink(
        os.path.join(target_path,'lidar', 'training','label_2'),
        os.path.join(testing_path, 'label_2')
    )

def create_attach(source_path, target_path):
    radar_path = os.path.join(target_path, 'radar')
    if not os.path.exists(radar_path):
        os.makedirs(radar_path)
    radar_training_path = os.path.join(radar_path, 'training')
    if not os.path.exists(radar_training_path):
        os.makedirs(radar_training_path)
    
    lidar_path = os.path.join(target_path, 'lidar')
    if not os.path.exists(radar_path):
        os.makedirs(radar_path)
    lidar_training_path = os.path.join(lidar_path, 'training')
    if not os.path.exists(lidar_training_path):
        os.makedirs(lidar_training_path)
    
    os.symlink(
        os.path.join(source_path, 'radar'),
        os.path.join(lidar_training_path, 'attach_lidar')
    )
    
    os.symlink(
        os.path.join(source_path, 'calib'),
        os.path.join(lidar_training_path, 'attach_calib')
    )
    
    os.symlink(
        os.path.join(source_path, 'lidar'),
        os.path.join(radar_training_path, 'attach_lidar')
    )
    
    os.symlink(
        os.path.join(source_path, 'calib'),
        os.path.join(radar_training_path, 'attach_calib')
    )
    
def main():
    print('start')
    # source_path = '/mnt/12T/hantao/shangqi/kitti_format'
    in_house_label_path = '/mnt/12T/hantao/shangqi/inhouse_labels'
    # target_path = '/mnt/12T/hantao/shangqi/inhouse_final_in_kitti'
    source_path = '/mnt/12T/hantao/shangqi_new/kitti_format'
    # in_house_label_path = '/mnt/12T/hantao/shangqi_new/inhouse_labels'
    target_path = '/mnt/12T/hantao/shangqi_new/inhouse_final_in_kitti'
    create_lidar(source_path, in_house_label_path, target_path)
    create_radar(source_path, target_path)
    create_attach(source_path, target_path)

if __name__ == '__main__':
    main()