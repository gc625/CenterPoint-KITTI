import argparse
import glob
import shutil
import os
from pathlib import Path
import lxml.etree
import read_raw, extract_gt, pcl_sync, copy_lidar, partition, vis_pcl, filter_radar_label

def _purge(paths):
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f'deleted "{path}"')

def _get_data_dirs(root, assume_yes=False):
    options = os.listdir(root)
    dirs = []
    for option in options:
        path = os.path.join(root, option)
        if os.path.isdir(path):
            dirs.append(path)
    if len(dirs) == 0:
        raise ValueError('No top-level directories found')
    print(f'Found {len(dirs)} potential data directories:')
    for i, dir in enumerate(dirs):
        print(f'\t{i + 1}.\t {dir}')
    if not assume_yes:
        ans = input('Is this the correct set of directories containing data? [y/n] ')
        if ans.lower() != 'y':
            raise ValueError('Preprocessing stopped. Answer "y"/"Y" next time.')
    _validate_timestamps(dirs)
    return dirs

def _validate_timestamps(dirs):
    timestamps = set()
    for dir in dirs:
        dname = Path(dir).name
        lidar_paths = os.path.join(dir, 'input', 'lidar', f'{dname}_C', '*.pcd')
        for lidar_path in glob.glob(lidar_paths):
            lidar_ts = Path(lidar_path).stem
            if lidar_ts in timestamps:
                raise ValueError(f'timestamp {lidar_ts} already encountered')
            timestamps.add(lidar_ts)
    print(f'Directories contain {len(timestamps)} unique lidar timestamps.')

def _get_base_timestamps(dir):
    """Extract the base timestamps needed for following steps from {dir}/meta.xml
    returns: 
        utc (for lidar and robosense), 
        local (for radar and pose)
    """
    metafile = os.path.join(dir, 'meta.xml')
    root = lxml.etree.parse(metafile).getroot()
    return {
        'utc': round(float(root.attrib['start_posix_utc'])),
        'local': round(float(root.attrib['start_posix_local']))
    }

def main():
    parser = argparse.ArgumentParser(description='Read raw radar sweeps')
    parser.add_argument('load_dir', help='path to root of directory containing unprocessed data')
    parser.add_argument('save_dir', help='path to root of directory to store processed data')
    parser.add_argument('--skip_read_raw', action='store_true', help='skip read_raw.py')
    parser.add_argument('--skip_extract_gt', action='store_true', help='skip extract_gt.py')
    parser.add_argument('--skip_pcl_sync', action='store_true', help='skip pcl_sync.py')
    parser.add_argument('--skip_copy_lidar', action='store_true', help='skip copy_lidar.py')
    parser.add_argument('--skip_partition', action='store_true', help='skip partition.py')
    # parser.add_argument('--skip_vis_pcl', action='store_true', help='skip vis_pcl.py')
    parser.add_argument('--skip_label_filter', action='store_true', help='skip radar_label_filter.py')
    parser.add_argument('-y', action='store_true', help='assume yes')
    args = parser.parse_args()

    _raw_paths = [
        os.path.join(args.save_dir, 'inhouse_format', 'radar_raw'),
        os.path.join(args.save_dir, 'inhouse_format', 'gt_raw'),
    ]

    print('*** STEP 0 - gather top-level directories and metadata ***')
    dirs = _get_data_dirs(args.load_dir, args.y)
    base_timestamps = {}
    for dir in dirs:
        base_timestamps[dir] = _get_base_timestamps(dir)
        print(f'Using {base_timestamps[dir]} as base timestamps (from {dir}/meta.xml)')
    # import ipdb
    # ipdb.set_trace()
    print('*** STEP 1 - read_raw.py ***')
    if not args.skip_read_raw:
        for dir in dirs:
            print(f'*** {dir} ***')
            read_raw.read_raw(dir, args.save_dir, base_timestamps[dir])
    else: print('(skipped)')

    print('*** STEP 2 - extract_gt.py ***')
    if not args.skip_extract_gt:
        for dir in dirs:
            print(f'*** {dir} ***')
            extract_gt.extract_gt(dir, args.save_dir)
    else: print('(skipped)')

    print('*** STEP 3 - pcl_sync.py ***')
    if not args.skip_pcl_sync:
        for dir in dirs:
            print(f'*** {dir} ***')
            pcl_sync.pcl_sync(dir, args.save_dir, base_timestamps[dir])
        _purge(_raw_paths)  # remove unsynced gt and radar data to save storage
    else: print('(skipped)')

    print('*** STEP 4 - copy_lidar.py ***')
    if not args.skip_copy_lidar: 
        for dir in dirs:
            print(f'*** {dir} ***')
            copy_lidar.copy_lidar(dir, args.save_dir)
    else: print('(skipped)')

    print('*** STEP 5 - filter_radar_label.py ***')
    if not args.skip_label_filter: 
        filter_radar_label.filter_radar_label(args.save_dir)
    else: print('(skipped)')


    print('*** STEP 6 - partition.py ***')
    if not args.skip_partition: 
        partition.partition(args.save_dir)
        _purge(_raw_paths)  # remove unsynced gt and radar data to save storage
    else: print('(skipped)')
    # print('*** STEP 6 - vis_pcl.py ***')
    # if not args.skip_vis_pcl: 
    #     for dir in dirs:
    #         print(f'*** {dir} ***')
    #         vis_pcl.vis_pcl(dir, args.save_dir)
    # else: print('(skipped)')

    print('*** PREPROCESSING COMPLETED ***')

if __name__ == '__main__':
    main()
