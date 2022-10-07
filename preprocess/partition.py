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

def _extract_values(subset):
    return [subset.dataset[i] for i in subset.indices]

def _get_timestamps(data_dir):
    gt_files = os.path.join(data_dir, 'inhouse_format', 'gt', '*.csv')
    return [Path(file).stem for file in sorted(glob.glob(gt_files))]

def _write_split_file(path, timestamps):
    with open(path, 'w') as f:
        f.writelines(ts + '\n' for ts in timestamps)

def partition(data_dir):
    split_files_dir = os.path.join(data_dir, 'kitti_format', 'ImageSets')
    timestamps = _get_timestamps(data_dir)
    sets = _create_partition(timestamps, [0.65, 0.15, 0.20])
    train, val, test = sets['train'], sets['val'], sets['test']
    trainval = train + val
    assert len(set(train)) + len(set(val)) + len(set(test)) == len(timestamps)
    assert len(set(trainval)) == len(train) + len(val)
    if not os.path.exists(split_files_dir):
        os.makedirs(split_files_dir)
    _write_split_file(os.path.join(split_files_dir, 'train.txt'), train)
    _write_split_file(os.path.join(split_files_dir, 'val.txt'), val)
    _write_split_file(os.path.join(split_files_dir, 'trainval.txt'), trainval)
    _write_split_file(os.path.join(split_files_dir, 'test.txt'), test)
