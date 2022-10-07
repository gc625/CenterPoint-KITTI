import os
import glob
import mmcv
from pathlib import Path

def copy_lidar(load_dir, save_dir):
    # NB: not actually copying files, but creating symlinks to save space
    # figure out the source directory
    lidar_source = os.path.join(load_dir, 'input', 'lidar')
    path_options = os.listdir(lidar_source)
    if len(path_options) != 1:
        raise ValueError(f'"{lidar_source}" contains {path_options} files. Expecting exactly 1 subdirectory.')
    lidar_source = os.path.join(lidar_source, path_options[0])
    # copy source files to processed directory to allow for easier downstream processing
    lidar_filenames = os.path.join(lidar_source, '*.pcd')
    dest_dir = os.path.join(save_dir, 'inhouse_format', 'lidar')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for _, src_path in enumerate(mmcv.track_iter_progress(glob.glob(lidar_filenames))):
        dest_path = os.path.join(dest_dir, Path(src_path).name)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.symlink(src_path, dest_path)
        
