import os
import glob
import random
from pathlib import Path

random.seed(3248723)

# https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _partition_chunk(chunk, split, root_train, root_val, root_test):
    N = len(chunk)
    train_end = int(round(split[0] * N))
    val_end = int(round((split[0] + split[1]) * N))
    # partition without shuffling
    train = chunk[:train_end]
    val = chunk[train_end:val_end]
    test = chunk[val_end:]
    for f, s in zip(split, [train, val, test]):
        if int(round(f * N)) != len(s):
            print(f'WARN: expected {int(round(f * N))} elements, got {len(s)}. difference: {int(round(f * N)) - len(s)}')
    # shuffle elements in sets
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    # add to root sets
    root_train += train
    root_val += val
    root_test += test



def _create_partition(elems, split):
    # assert timestamps are sorted
    all(elems[i] <= elems[i+1] for i in range(len(elems) - 1))
    # assert splits
    if not isinstance(split, list) or sum(split) != 1.0 or not all(val > 0.0 for val in split):
        raise ValueError(f'{split} must be a list whose values sum to 1.0, with no negative values.')
    assert isinstance(split, list)
    assert sum(split) == 1.0
    N = len(elems)
    assert len(set(elems)) == N  # no duplicates
    # divide into 100 chunks (~45secs each)
    chunk_size = N // 100
    elem_chunks = chunks(elems, chunk_size)
    # for each chunk, partition it and add it to the train/val/test split
    root_train, root_val, root_test = [], [], []
    for chunk in elem_chunks:
        _partition_chunk(chunk, split=split, root_train=root_train, root_val=root_val, root_test=root_test)
    # shuffle elements in sets
    random.shuffle(root_train)
    random.shuffle(root_val)
    random.shuffle(root_test)
    # sanity checks
    assert len(root_train) + len(root_val) + len(root_test) == N
    assert len(set(root_train + root_val + root_test)) == N, 'partition contains duplicates'
    for f, s in zip(split, [root_train, root_val, root_test]):
        if int(round(f * N)) != len(s):
            print(f'WARN: expected {int(round(f * N))} elements, got {len(s)}. difference: {int(round(f * N)) - len(s)}')
    counts = [len(s) for s in [root_train, root_val, root_test]]
    print(f'INFO: split "{split}" resulted in the following subset sizes: {counts}')
    return {
        'train': root_train,
        'val': root_val,
        'test': root_test,
    }

def _partition_chunk(chunk, split, root_train, root_val):
    N = len(chunk)
    train_end = int(round(split[0] * N))
    val_end = int(round((split[0] + split[1]) * N))
    # partition without shuffling
    train = chunk[:train_end]
    val = chunk[train_end:]
    for f, s in zip(split, [train, val]):
        if int(round(f * N)) != len(s):
            print(f'WARN: expected {int(round(f * N))} elements, got {len(s)}. difference: {int(round(f * N)) - len(s)}')
    # shuffle elements in sets
    random.shuffle(train)
    random.shuffle(val)
    # random.shuffle(test)
    # add to root sets
    root_train += train
    root_val += val
    # root_test += test

def _create_trainval_partition(elems, split):
    # assert timestamps are sorted
    all(elems[i] <= elems[i+1] for i in range(len(elems) - 1))
    # assert splits
    if not isinstance(split, list) or sum(split) != 1.0 or not all(val > 0.0 for val in split):
        raise ValueError(f'{split} must be a list whose values sum to 1.0, with no negative values.')
    assert isinstance(split, list)
    assert sum(split) == 1.0
    N = len(elems)
    assert len(set(elems)) == N  # no duplicates
    # divide into 100 chunks (~45secs each)
    chunk_size = N // 100
    elem_chunks = chunks(elems, chunk_size)
    # for each chunk, partition it and add it to the train/val/test split
    root_train, root_val = [], []
    for chunk in elem_chunks:
        _partition_chunk(chunk, split=split, root_train=root_train, root_val=root_val)
    # shuffle elements in sets
    random.shuffle(root_train)
    random.shuffle(root_val)
    # random.shuffle(root_test)
    # sanity checks
    assert len(root_train) + len(root_val) == N
    assert len(set(root_train + root_val)) == N, 'partition contains duplicates'
    for f, s in zip(split, [root_train, root_val]):
        if int(round(f * N)) != len(s):
            print(f'WARN: expected {int(round(f * N))} elements, got {len(s)}. difference: {int(round(f * N)) - len(s)}')
    counts = [len(s) for s in [root_train, root_val]]
    print(f'INFO: split "{split}" resulted in the following subset sizes: {counts}')
    return {
        'train': root_train,
        'val': root_val,
    }


def _extract_values(subset):
    return [subset.dataset[i] for i in subset.indices]

def _get_timestamps(data_dir):
    gt_files = os.path.join(data_dir,'inhouse_format', 'gt', '*.csv')
    return [Path(file).stem for file in sorted(glob.glob(gt_files))]

def _remove_test_frames(timestamps,test_set_dir):
    test_files = [Path(file).stem for file in sorted(glob.glob(test_set_dir))]

    counter = 0 
    for file in test_files: 
        if file in timestamps:
            timestamps.remove(file)
            counter += 1

    assert counter == len(test_set_dir), "number of test frame removed doesn't match"
    return timestamps

def _write_split_file(path, timestamps):
    with open(path, 'w') as f:
        f.writelines(ts + '\n' for ts in timestamps)

def partition(data_dir):

    test_set_dir = '/mnt/12T/public/labeling/lidar_subset'

    split_files_dir = os.path.join(data_dir, 'ImageSets')
    timestamps = _get_timestamps(data_dir)
    trainval_timestamps = _remove_test_frames(timestamps,test_set_dir)
    
    
    sets = _create_trainval_partition(trainval_timestamps, [0.8, 0.2])

    train, val= sets['train'], sets['val']
    trainval = train + val
    test = _get_timestamps(test_set_dir)

    assert len(set(train)) + len(set(val)) + len(set(test)) == len(timestamps)
    assert len(set(trainval)) == len(train) + len(val)
    if not os.path.exists(split_files_dir):
        os.makedirs(split_files_dir)
    _write_split_file(os.path.join(split_files_dir, 'train.txt'), train)
    _write_split_file(os.path.join(split_files_dir, 'val.txt'), val)
    _write_split_file(os.path.join(split_files_dir, 'trainval.txt'), trainval)
    _write_split_file(os.path.join(split_files_dir, 'test.txt'), test)

def filter_missing():
    lidar = '/mnt/12T/public/labeling/lidar'
    gt = '/mnt/12T/public/labeling/label_2/*'    
    output = '/mnt/12T/public/labeling/lidar_subset'

    gt_files = glob.glob(gt)
    lidar_files = glob.glob(os.path.join(lidar,"*"))
    for f in gt_files:
        os.symlink(os.path.join(lidar,Path(f).stem+".bin"),os.path.join(output,Path(f).stem+".bin"))
    assert(len(glob.glob(os.path.join(output,'*'))) == len(gt_files))

def _createlinks(imageset,output_dir,root):
    lidar_files = root + '/lidar_subset'
    label_files = root + '/label_2'

    lidar_output = output_dir + '/velodyne'
    label_output = output_dir + '/label_2'

    with open(imageset,'r') as f:
        list_files = f.read().splitlines()


    for file in list_files: 
        lidar_tgt = file+'.bin'
        label_tgt = file+'.txt'
    
        assert os.path.isfile(
            os.path.join(lidar_files,lidar_tgt)
        ), f'LIDAR FILE MISSING {lidar_tgt}'
        assert os.path.isfile(
            os.path.join(label_files,label_tgt)
        ), f'label FILE MISSING {label_tgt}'


        os.symlink(
            os.path.join(lidar_files,lidar_tgt),
            os.path.join(lidar_output,lidar_tgt)
        )
        
        os.symlink(
            os.path.join(label_files,label_tgt),
            os.path.join(label_output,label_tgt)
        )

    assert(len(glob.glob(lidar_files)) == len(glob.glob(lidar_output))), 'lidar doesnt match'
    assert(len(glob.glob(label_files)) == len(glob.glob(label_output))), 'labels doesnt match'

def softlinking():
    root = '/mnt/12T/public/labeling'
    train_set, test_set = os.path.join(root,'ImageSets/trainval.txt'),os.path.join(root,'ImageSets/test.txt')
    train, test = os.path.join(root,'training'),os.path.join(root,'testing')
    _createlinks(train_set,train,root)
    _createlinks(test_set,test,root)

def unlink_directory(dir):
    if dir[-1] != '*':
        dir = os.path.join(dir,'*')
    files = glob.glob(dir)
    for f in files:
        os.unlink(f)    




