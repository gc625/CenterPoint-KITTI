from distutils.log import debug
import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, runtime_gt=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    frame_ids = []
    sum_duration = 0

    # collect centers, centers_origin for visualization
    center_dict = {}
    center_origin_dict = {}
    ip_dict = {}
    det_dict = {}
    match_dict = {}
    lidar_center_dict = {}
    lidar_preds_dict = {}
    radar_preds_dict = {}
    radar_label_dict = {}
    init_flag = False
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            start_run_time = time.time()
            pred_dicts, ret_dict = model(batch_dict)
            duration = time.time() - start_run_time
            frame_ids += list(batch_dict['frame_id'])
            if hasattr(model, 'debug'):
                debug = model.debug
            else:
                debug = False
            save_center = ('centers' in batch_dict)

            if save_center:
                centers = batch_dict['centers'].cpu().numpy()
                centers_origin = batch_dict['centers_origin'].cpu().numpy()
                points = batch_dict['points'].cpu().numpy()


                center_dict[frame_ids[-1]] = centers
                center_origin_dict[frame_ids[-1]] = centers_origin
                ip_dict[frame_ids[-1]] = points
                # pointwise classification
                if debug:
                    radar_idx = batch_dict['radar_idx'].cpu().numpy().reshape([-1, 1])
                    lidar_idx = batch_dict['lidar_idx'].cpu().numpy().reshape([-1, 1])
                    mask = batch_dict['mask'].cpu().numpy().reshape([-1, 1])
                    matches = np.concatenate((radar_idx, lidar_idx, mask), axis=1)
                    lidar_center = batch_dict['lidar_centers'].cpu().numpy()
                    lidar_preds = batch_dict['lidar_preds'][2]
                    radar_cls_label = batch_dict['sa_ins_labels']
                    radar_preds = batch_dict['sa_ins_preds'][2]
                    # print('saving debug result')

                    match_dict[frame_ids[-1]] = matches
                    lidar_center_dict[frame_ids[-1]] = lidar_center
                    lidar_preds_dict[frame_ids[-1]] = lidar_preds
                    radar_preds_dict[frame_ids[-1]] = radar_preds
                    radar_label_dict[frame_ids[-1]] = radar_cls_label
                

        disp_dict = {}
        sum_duration += duration
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        det_dict[frame_ids[-1]] = annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    peak_memory = torch.cuda.max_memory_allocated() / 1024 # convert to KByte
    
    logger.info('Peak memory usage: %.4f KB.' % peak_memory)
    peak_memory = peak_memory/1024 # convert to MByte
    logger.info('Peak memory usage: %.4f MB.' % peak_memory)
    logger.info('Average run time per scan: %.4f ms' % (sum_duration / len(dataloader.dataset) * 1000))
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    logger.info('******************Saving result to dir: ' + str(result_dir) + '**********************')

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    # gt pkl
    import copy

    gt_dict = {}
    for info in dataset.kitti_infos:
        frame_id = copy.deepcopy(info['image']['image_idx'])
        gt_anno = copy.deepcopy(info['annos'])
        gt_dict[frame_id] = gt_anno
        pass
    # gt_annos = [copy.deepcopy(info['annos']) for info in dataset.kitti_infos]
    gt_annos = []
    for id in frame_ids:
        gt_annos += [gt_dict[id]]
    with open(result_dir / 'gt.pkl', 'wb') as f:
        pickle.dump(gt_dict, f)

    # save detection
    with open(result_dir / 'dt.pkl', 'wb') as f:
        pickle.dump(det_dict, f)
    
    if save_center:

        save_name_list = ('centers', 'centers_origin', 'points', 'match', 'lidar_center', 'lidar_preds', 'radar_preds', 'radar_label')
        save_dict_list = (center_dict, center_origin_dict, ip_dict, match_dict, lidar_center_dict, lidar_preds_dict, radar_preds_dict, radar_label_dict)
        '''
        center_dict = {}
        center_origin_dict = {}
        ip_dict = {}
        match_dict = {}
        lidar_center_dict = {}
        lidar_preds_dict = {}
        radar_preds_dict = {}
        radar_label_dict = {}
        '''
        # # save centers 
        # with open(result_dir / 'centers.pkl', 'wb') as f:
        #     pickle.dump(center_dict, f)
        # # save centers_origin
        # with open(result_dir / 'centers_origin.pkl', 'wb') as f:
        #     pickle.dump(center_origin_dict, f)
        # # save input points
        # with open(result_dir / 'points.pkl', 'wb') as f:
        #     pickle.dump(ip_dict, f)

        for i, name in enumerate(save_name_list):
            save_data = save_dict_list[i]
            save_name = result_dir / (name + '.pkl')
            with open(save_name, 'wb') as f:
                pickle.dump(save_data, f)
            


    # save frame ids
    with open(result_dir / 'frame_ids.txt', 'w') as f:
        for id in frame_ids: 
            f.write(str(id) + ',')

    try:
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names, gt_annos=gt_annos,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )
    except:
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )

    logger.info(result_str)
    ret_dict.update(result_dict)
    # save gt, prediction, final points origin, final points new coordinate
    
    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
