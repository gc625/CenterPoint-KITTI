from turtle import forward
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pcdet.utils.SSD as SSD
from . import pointnet2_utils
import pcdet.ops.pointnet2.pointnet2_3DSSD.pointnet2_utils as pointnet2_3DSSD

class _PointnetSAModuleBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None
        self.pool_method = 'max_pool'

    def calc_square_dist(self, a, b, norm=True):
        """
        Calculating square distance between a and b
        a: [bs, n, c]
        b: [bs, m, c]
        """
        n = a.shape[1]
        m = b.shape[1]
        num_channel = a.shape[-1]
        a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
        b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
        a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
        b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
        a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
        b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

        coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

        if norm:
            dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
            # dist = torch.sqrt(dist)
        else:
            dist = a_square + b_square - 2 * coor
            # dist = torch.sqrt(dist)
        return dist

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, N, C) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if new_xyz is None:
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped,
                pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            ).transpose(1, 2).contiguous() if self.npoint is not None else None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            if self.pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    new_features, kernel_size=[1, new_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)

class PoolingWeight(nn.Module):
    def __init__(self):
        super().__init__()
        # shape of weights [B, 1, Npoints, Nsamples]
        # created during inference, save gradient only
        self.weights = None
        self.device = None
        pass

    def init_weights(self, features):
        B, _, N, ns = features.shape
        self.weights = torch.ones([B, 1, N, ns])
        self.weights = self.weights.to(features.device)
        self.device = features.device
        self.weights.requires_grad = True
    
    def clear_grad(self):
        try:
            self.weights.grad.zero_()
        except:
            pass
                

    def reset_weights(self):
        self.weights.fill_(1)

    def forward(self, x):
        '''
        @param: x: input features 
        '''
        return self.weights * x
        


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool'):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint[0]
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

class PointnetSAModuleMSG_SSD(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with multiscale grouping"""

    def __init__(self, *, npoint: int, radii: List[float], nsamples: List[int], mlps: List[List[int]], bn: bool = True,
                 use_xyz: bool = True, pool_method='max_pool', out_channle=-1, fps_type='D-FPS', fps_range=-1,
                 dilated_group=False):
        """
        :param npoint: int
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__()
        self.fps_types = fps_type
        self.fps_ranges = fps_range
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)
        # print(f'SA module {npoint}')
        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))

        self.pool_method = pool_method

        if out_channle != -1 and len(self.mlps) > 0:
            in_channel = 0
            for mlp_tmp in mlps:
                in_channel += mlp_tmp[-1]
            shared_mlps = []
            shared_mlps.extend([
                nn.Conv1d(in_channel, out_channle, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channle),
                nn.ReLU()
            ])
            self.out_aggregation = nn.Sequential(*shared_mlps)
            pass

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, new_xyz=None, ctr_xyz=None) -> (torch.Tensor, torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param new_xyz:
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, npoint, \sum_k(mlps[k][-1])) tensor of the new_features descriptors
        """
        # print(f'in SA module, pt shape: {xyz.shape}')
        # print(f'in SA module, feature shape: {features.shape}')
        new_features_list = []
        features = features.contiguous()

        xyz_flipped = xyz.transpose(1, 2).contiguous()

        if ctr_xyz is None:
            last_fps_end_index = 0
            fps_idxes = []
            for i in range(len(self.fps_types)):
                fps_type = self.fps_types[i]
                fps_range = self.fps_ranges[i]
                npoint = self.npoint[i]
                if npoint == 0:
                    continue
                if fps_range == -1:
                    xyz_tmp = xyz[:, last_fps_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:, :]
                else:
                    xyz_tmp = xyz[:, last_fps_end_index:fps_range, :]
                    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:fps_range, :]
                    last_fps_end_index += fps_range

                # print(f'xyztemp: {xyz_tmp.shape}')
                
                if fps_type == 'D-FPS':
                    fps_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
                elif fps_type == 'F-FPS':
                    # features_SSD = xyz_tmp
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                elif fps_type == 'FS':
                    # features_SSD = xyz_tmp
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    fps_idx_1 = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    fps_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    fps_idx = torch.cat([fps_idx_1, fps_idx_2], dim=-1)  # [bs, npoint * 2]
                # print(fps_idx)
                fps_idxes.append(fps_idx)
            fps_idxes = torch.cat(fps_idxes, dim=-1)
            # print(fps_idxes)
            new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped, fps_idxes
            ).transpose(1, 2).contiguous() if self.npoint is not None else None
        else:
            new_xyz = ctr_xyz
        # print('-'*70)
        # print(self.groupers)
        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):

                new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                # print(f'in SA module, new feature shape idx [{i}]: {new_features.shape}')
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                # print(f'in SA module, new feature shape postmlp: {new_features.shape}')
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError
                # print(f'in SA module, new feature shape after maxpool: {new_features.shape}')
                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)
                # print('-'*70)
            new_features = torch.cat(new_features_list, dim=1)
            # print(f'in SA module, new feature shape after cat: {new_features.shape}')
            new_features = self.out_aggregation(new_features)
            # print(self.out_aggregation)
            # print(f'in SA module, new feature shape after agg: {new_features.shape}')
        else:
            new_features = pointnet2_utils.gather_operation(features, fps_idxes).contiguous()
        # print('-'*70)
        # print(f'in SA module, new pt shape: {new_xyz.shape}')
        # print(f'in SA module, new feature shape: {new_features.shape}')
        # print('='*70)
        return new_xyz, new_features


class FPSampler():
    def __init__(self, npoint, fps_type) -> None:
        self.npoint = npoint
        self.fps_type = fps_type
        
    def sample(self, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param npoint: number of sampled points
        :param type: FPS type, option: D-FPS, F-FPS
        """
        npoint = self.npoint
        fps_type = self.fps_type

        features = features.contiguous()

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        last_fps_end_index = 0
        # npoint = 1
        xyz_tmp = xyz[:, last_fps_end_index:, :]
        feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:, :]
        if fps_type == 'D-FPS':
            fps_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
        elif fps_type == 'F-FPS':
            features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
            features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
            features_for_fps_distance = features_for_fps_distance.contiguous()
            fps_idx = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
        new_xyz = pointnet2_utils.gather_operation(
                xyz_flipped, fps_idx
            ).transpose(1, 2).contiguous() if npoint is not None else None # (B, N, 3)
        return new_xyz, fps_idx

def FPS(xyz: torch.Tensor, features: torch.Tensor = None, npoint=None, fps_type='D-FPS'):
    """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param npoint: number of sampled points
        :param type: FPS type, option: D-FPS, F-FPS
    """
    features = features.contiguous()

    xyz_flipped = xyz.transpose(1, 2).contiguous()
    last_fps_end_index = 0
    # npoint = 1
    xyz_tmp = xyz[:, last_fps_end_index:, :]
    feature_tmp = features.transpose(1, 2)[:, last_fps_end_index:, :]
    if fps_type == 'D-FPS':
        fps_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
    elif fps_type == 'F-FPS':
        features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
        features_for_fps_distance = SSD.calc_square_dist(features_SSD, features_SSD)
        features_for_fps_distance = features_for_fps_distance.contiguous()
        fps_idx = pointnet2_3DSSD.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
    new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, fps_idx
        ).transpose(1, 2).contiguous() if npoint is not None else None # (B, N, 3)
    return new_xyz, fps_idx


class PointnetSAModuleMSG_WithSampling(_PointnetSAModuleBase):
    """Pointnet set abstraction layer with specific downsampling and multiscale grouping """

    def __init__(self, *,
                 npoint_list: List[int],
                 sample_range_list: List[int],
                 sample_type_list: List[str],
                 radii: List[float],
                 nsamples: List[int],
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 reverse=False,
                 pool_method='max_pool',
                 sample_idx=False,
                 use_pooling_weights=False,
                 aggregation_mlp: List[int],
                 confidence_mlp: List[int],
                 num_class):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param sample_range_list: list of list of int, sample index range [left, right] for every sampling type
        :param sample_type_list: list of str, list of used sampling type, d-fps or f-fps
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param sample_idx: whether to return sample_idx
        :param dilated_group: whether to use dilated group
        :param reverse: if True, return bg samples instead of fg samples
        :param aggregation_mlp: list of int, spec aggregation mlp
        :param confidence_mlp: list of int, spec confidence mlp
        :param num_class: int, class for process
        """
        super().__init__()
        self.sample_type_list = sample_type_list
        self.sample_range_list = sample_range_list
        self.dilated_group = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        self.use_pool_weights = use_pooling_weights
        self.pool_weights = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.reverse = reverse
        out_channels = 0
        self.sample_idx = sample_idx
        # ===========================================
        # counter for naming saved features files
        self.saving_cnt = 0
        # ===========================================
        # initialize a module to get weights
        # pooling weights module is required for each SA_Layer
        # ===========================================
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                self.groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
                if self.use_pool_weights:
                    self.pool_weights.append(PoolingWeight())
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )

                if self.use_pool_weights:
                    self.pool_weights.append(PoolingWeight())

            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]

        self.pool_method = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(self.mlps) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            self.aggregation_layer = nn.Sequential(*shared_mlp)
        else:
            self.aggregation_layer = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels,
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            self.confidence_layers = nn.Sequential(*shared_mlp)
        else:
            self.confidence_layers = None

    def save_features(self, features, coords, xyz, save_path, frame_id):
        ''' 
        :param features: (B, mlp[-1], npoint, nsample)
        :param coords: (B, 3, npoint, nsample)
        :param xyz: group center
        :param save_path: path/to/save
        :param frame_id: name the file using frame id
        '''
        import numpy as np
        from pathlib import Path
        B, _, _, _ = features.shape
        save_features = features.cpu().detach().numpy()
        save_coords = coords.cpu().detach().numpy()
        center_coords = xyz.cpu().detach().numpy()
        save_data = np.concatenate((save_coords, save_features), axis=1)  
        fname = Path(save_path) / (frame_id + '_group.npy')
        np.save(str(fname), save_data)

        fname = Path(save_path) / (frame_id + '_center.npy')
        np.save(str(fname), center_coords)

        self.saving_cnt += 1

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None, cls_features: torch.Tensor = None, \
        new_xyz=None, ctr_xyz=None, save_features_dir=None, frame_id=None, pooling_weights=None):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param cls_features: (B, N, num_class) tensor of the descriptors of the the confidence (classification) features 
        :param new_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        :param ctr_xyz: tensor of the xyz coordinates of the centers 
        :param save_features_dir: path to save intermediate features during inference
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(self.sample_type_list)):
                sample_type = self.sample_type_list[i]
                sample_range = self.sample_range_list[i]
                npoint = self.npoint_list[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    # if self.reverse: 
                    #     score_pred = 1 - score_pred # background sampling
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'mix' in sample_type.lower():
                    fps_npoints = int(npoint/2)
                    ctr_npoints = npoint - fps_npoints
                    sample_idx_fps = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
                    # ==========
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    # if self.reverse: 
                    #     score_pred = 1 - score_pred # background sampling
                    score_picked, sample_idx_ctr = torch.topk(score_pred, ctr_npoints, dim=-1)           
                    sample_idx_ctr = sample_idx_ctr.int()
                    sample_idx = torch.cat([sample_idx_fps, sample_idx_ctr], dim=-1)  # [bs, npoint * 2]

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) 
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()

        else:
            new_xyz = ctr_xyz




        if len(self.groupers) > 0:
            for i in range(len(self.groupers)):
                # ================================== this features can be saved for analysis =======================

                if (self.training is False) & (save_features_dir is not None):
                    new_features = self.groupers[i](xyz, new_xyz, features, save_abs_coord=True)  # (B, C+3, npoint, nsample) point coordinate included        
                    save_coords = new_features[:, -3:, : ,:]
                    new_features = new_features[:, :-3, :, :]
                # save coordinate from here
                else:
                    new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample) point coordinate included
                # save_coords = new_features[:,:3, :, :]
                # save pre-pooling features from here
                # =================================================
                # add a learnable weights here before pooling for visualization
                # =================================================
                new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                # ==================================
                # function to save features before further operations
                # ==================================
                if self.use_pool_weights:
                    p_weights = self.pool_weights[i]
                    if p_weights.weights is None:
                        p_weights.init_weights(new_features)
                    p_weights.clear_grad()
                    new_features = p_weights(new_features)
                # ==================================
                if self.training is False:
                    if save_features_dir is not None:
                        self.save_features(new_features, save_coords, new_xyz, save_features_dir, frame_id)
                if self.pool_method == 'max_pool':
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                elif self.pool_method == 'avg_pool':
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)]
                    )  # (B, mlp[-1], npoint, 1)
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
                new_features_list.append(new_features)

            new_features = torch.cat(new_features_list, dim=1)

            if self.aggregation_layer is not None:
                new_features = self.aggregation_layer(new_features)
        else:
            new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        if self.confidence_layers is not None:
            cls_features = self.confidence_layers(new_features).transpose(1, 2)
            
        else:
            cls_features = None

        if self.sample_idx:
            return new_xyz, new_features, cls_features, sampled_idx_list
        else:
            return new_xyz, new_features, cls_features


class DoubleSAModuleMSG_WithSampling(_PointnetSAModuleBase):
    def __init__(self, *,
                 npoint_list: List[List[int]],
                 sample_range_list: List[List[int]],
                 sample_type_list: List[List[str]],
                 radii: List[List[float]],
                 nsamples: List[List[int]],
                 mlps: List[List[List[int]]],                 
                 use_xyz: List[bool] = [True,True],
                 dilated_group= [False,False],
                 reverse=  [False,False],
                 pool_method=['max_pool','maxpool'],
                 sample_idx=[False,False],
                 use_pooling_weights=[False,False],
                 aggregation_mlp: List[List[int]],
                 confidence_mlp: List[List[int]],
                 num_class):


        self.MSGs = []
        for i in range(len(npoint_list)):
            self.MSGs.append(PointnetSAModuleMSG_WithSampling(
                npoint_list=npoint_list[i],
                sample_range_list=sample_range_list[i],
                sample_type_list=sample_type_list[i],
                radii=radii[i],
                nsamples=nsamples[i],
                mlps=mlps[i],
                use_xyz=use_xyz[i],
                dilated_group=dilated_group[i],
                reverse=reverse[i],
                pool_method=pool_method[i],
                sample_idx=sample_idx[i],
                use_pooling_weights=use_pooling_weights[i],
                aggregation_mlp=aggregation_mlp[i],
                confidence_mlp=confidence_mlp[i],
                num_class=num_class,
                )
            )

    def forward(self,
            xyz: List[torch.Tensor],
            features: List[torch.Tensor],
            cls_features: List[torch.Tensor],
            new_xyz=[None,None],
            ctr_xyz=[None,None]):


        results = []
        for i in range(len(xyz)):
            ret = self.MSGs[i](
                xyz=xyz[i],
                features=features[i],
                cls_features=cls_features[i],
                new_xyz=new_xyz[i],)




class MMSAModuleMSG_WithSampling(_PointnetSAModuleBase):


    def build_sa_modules(self,
                npoint_list: List[int],
                sample_range_list: List[int],
                sample_type_list: List[str],
                radii: List[float],
                nsamples: List[int],
                mlps: List[List[int]],                 
                use_xyz: bool = True,
                dilated_group=False,
                reverse=False,
                pool_method='max_pool',
                sample_idx=False,
                use_pooling_weights=False,
                aggregation_mlp: List[int] = None,
                confidence_mlp: List[int] = None,
                num_class: int = 3) -> dict:
        
        SA_MODULES = nn.ModuleList()
        GROUPING_MODULES = nn.ModuleList()
        POOL_WEIGHTS = nn.ModuleList()
        MLPS = nn.ModuleList()
        NSAMPLES = nsamples

        SAMPLE_TYPE_LIST = sample_type_list
        SAMPLE_RANGE_LIST = sample_range_list
        DILATED_GROUP = dilated_group

        assert len(radii) == len(nsamples) == len(mlps)

        NPOINT_LIST = npoint_list
        USE_POOLING_WEIGHTS = use_pooling_weights
        
        REVERSE = reverse
        out_channels = 0
        SAMPLE_IDX = sample_idx



        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            if DILATED_GROUP:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                GROUPING_MODULES.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
                if USE_POOLING_WEIGHTS:
                    POOL_WEIGHTS.append(PoolingWeight())
            else:
                GROUPING_MODULES.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )

                if USE_POOLING_WEIGHTS:
                    POOL_WEIGHTS.append(PoolingWeight())

            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv2d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            MLPS.append(nn.Sequential(*shared_mlps).to('cuda'))
            out_channels += mlp_spec[-1]

        POOL_METHOD = pool_method

        if (aggregation_mlp is not None) and (len(aggregation_mlp) != 0) and (len(MLPS) > 0):
            shared_mlp = []
            for k in range(len(aggregation_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels*2, #! MULT 2 HERE FOR EXTRA ATTENTION FEATURE
                              aggregation_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(aggregation_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = aggregation_mlp[k]
            AGGREGATION_LAYER = nn.Sequential(*shared_mlp).to('cuda')
        else:
            AGGREGATION_LAYER = None

        if (confidence_mlp is not None) and (len(confidence_mlp) != 0):
            shared_mlp = []
            for k in range(len(confidence_mlp)):
                shared_mlp.extend([
                    nn.Conv1d(out_channels, 
                              confidence_mlp[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(confidence_mlp[k]),
                    nn.ReLU()
                ])
                out_channels = confidence_mlp[k]
            shared_mlp.append(
                nn.Conv1d(out_channels, num_class, kernel_size=1, bias=True),
            )
            CONFIDENCE_LAYERS = nn.Sequential(*shared_mlp).to('cuda')
        else:
            CONFIDENCE_LAYERS = None


        ALL_MODULES = {
            "SA_MODULES":SA_MODULES,
            "GROUPING_MODULES":GROUPING_MODULES,
            "POOL_WEIGHTS":POOL_WEIGHTS,
            "MLPS":MLPS,
            "AGGREGATION_LAYER":AGGREGATION_LAYER,
            "CONFIDENCE_LAYERS":CONFIDENCE_LAYERS,
            "SAMPLE_TYPE_LIST":SAMPLE_TYPE_LIST,
            "SAMPLE_RANGE_LIST":SAMPLE_RANGE_LIST,
            "NPOINT_LIST":NPOINT_LIST,
            "REVERSE":REVERSE,
            "SAMPLE_IDX":SAMPLE_IDX,
            "POOL_METHOD":POOL_METHOD,
            'NSAMPLES': NSAMPLES,
            'RADII': radii
        }

        return ALL_MODULES 

    def single_sample_and_gather(self,
                            MODULE_INFO: dict,
                            xyz: torch.Tensor, 
                            features: torch.Tensor = None, 
                            cls_features: torch.Tensor = None,
                            new_xyz=None, 
                            ctr_xyz=None,):

        SAMPLE_TYPE_LIST = MODULE_INFO['SAMPLE_TYPE_LIST']
        SAMPLE_RANGE_LIST = MODULE_INFO['SAMPLE_RANGE_LIST']
        NPOINT_LIST = MODULE_INFO['NPOINT_LIST']


        new_features_list = []
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sampled_idx_list = []
        if ctr_xyz is None:
            last_sample_end_index = 0
            
            for i in range(len(SAMPLE_TYPE_LIST)):
                sample_type = SAMPLE_TYPE_LIST[i]
                sample_range = SAMPLE_RANGE_LIST[i]
                npoint = NPOINT_LIST[i]

                if npoint <= 0:
                    continue
                if sample_range == -1: #全部
                    xyz_tmp = xyz[:, last_sample_end_index:, :]
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:, :].contiguous()  
                    cls_features_tmp = cls_features[:, last_sample_end_index:, :] if cls_features is not None else None 
                else:
                    xyz_tmp = xyz[:, last_sample_end_index:sample_range, :].contiguous()
                    feature_tmp = features.transpose(1, 2)[:, last_sample_end_index:sample_range, :]
                    cls_features_tmp = cls_features[:, last_sample_end_index:sample_range, :] if cls_features is not None else None 
                    last_sample_end_index += sample_range

                if xyz_tmp.shape[1] <= npoint: # No downsampling
                    sample_idx = torch.arange(xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32) * torch.ones(xyz_tmp.shape[0], xyz_tmp.shape[1], device=xyz_tmp.device, dtype=torch.int32)

                elif ('cls' in sample_type) or ('ctr' in sample_type):
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    # if self.reverse: 
                    #     score_pred = 1 - score_pred # background sampling
                    score_picked, sample_idx = torch.topk(score_pred, npoint, dim=-1)           
                    sample_idx = sample_idx.int()

                elif 'D-FPS' in sample_type or 'DFS' in sample_type:
                    sample_idx = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)

                elif 'mix' in sample_type.lower():
                    fps_npoints = int(npoint/2)
                    ctr_npoints = npoint - fps_npoints
                    sample_idx_fps = pointnet2_utils.furthest_point_sample(xyz_tmp.contiguous(), npoint)
                    # ==========
                    cls_features_max, class_pred = cls_features_tmp.max(dim=-1)
                    score_pred = torch.sigmoid(cls_features_max) # B,N
                    # if self.reverse: 
                    #     score_pred = 1 - score_pred # background sampling
                    score_picked, sample_idx_ctr = torch.topk(score_pred, ctr_npoints, dim=-1)           
                    sample_idx_ctr = sample_idx_ctr.int()
                    sample_idx = torch.cat([sample_idx_fps, sample_idx_ctr], dim=-1)  # [bs, npoint * 2]

                elif 'F-FPS' in sample_type or 'FFS' in sample_type:
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)

                elif sample_type == 'FS':
                    features_SSD = torch.cat([xyz_tmp, feature_tmp], dim=-1)
                    features_for_fps_distance = self.calc_square_dist(features_SSD, features_SSD)
                    features_for_fps_distance = features_for_fps_distance.contiguous()
                    sample_idx_1 = pointnet2_utils.furthest_point_sample_with_dist(features_for_fps_distance, npoint)
                    sample_idx_2 = pointnet2_utils.furthest_point_sample(xyz_tmp, npoint)
                    sample_idx = torch.cat([sample_idx_1, sample_idx_2], dim=-1)  # [bs, npoint * 2]
                elif 'Rand' in sample_type:
                    sample_idx = torch.randperm(xyz_tmp.shape[1],device=xyz_tmp.device)[None, :npoint].int().repeat(xyz_tmp.shape[0], 1)
                elif sample_type == 'ds_FPS' or sample_type == 'ds-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        radii = per_xyz.norm(dim=-1) -5 
                        storted_radii, indince = radii.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)
                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                elif sample_type == 'ry_FPS' or sample_type == 'ry-FPS':
                    part_num = 4
                    xyz_div = []
                    idx_div = []
                    for i in range(len(xyz_tmp)):
                        per_xyz = xyz_tmp[i]
                        ry = torch.atan(per_xyz[:,0]/per_xyz[:,1])
                        storted_ry, indince = ry.sort(dim=0, descending=False)
                        per_xyz_sorted = per_xyz[indince]
                        per_xyz_sorted_div = per_xyz_sorted.view(part_num, -1 ,3)

                        per_idx_div = indince.view(part_num,-1)
                        xyz_div.append(per_xyz_sorted_div)
                        idx_div.append(per_idx_div)
                    xyz_div = torch.cat(xyz_div ,dim=0)
                    idx_div = torch.cat(idx_div ,dim=0)
                    idx_sampled = pointnet2_utils.furthest_point_sample(xyz_div, (npoint//part_num))

                    indince_div = []
                    for idx_sampled_per, idx_per in zip(idx_sampled, idx_div):                    
                        indince_div.append(idx_per[idx_sampled_per.long()])
                    index = torch.cat(indince_div, dim=-1)

                    sample_idx = index.reshape(xyz.shape[0], npoint).int()

                sampled_idx_list.append(sample_idx)

            sampled_idx_list = torch.cat(sampled_idx_list, dim=-1) 
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sampled_idx_list).transpose(1, 2).contiguous()

        else:
            new_xyz = ctr_xyz

        return new_xyz,sampled_idx_list


    def single_gather_features(self,
                            MODULE_INFO,
                            xyz,
                            new_xyz,
                            features,
                            sampled_idx_list):
        groupers = MODULE_INFO['GROUPING_MODULES']
        mlps = MODULE_INFO['MLPS']
        pool_method = MODULE_INFO['POOL_METHOD']
        confidence_layers = MODULE_INFO['CONFIDENCE_LAYERS']
        sample_idx = MODULE_INFO['SAMPLE_IDX']
        radii = MODULE_INFO['RADII']
        nsamples = MODULE_INFO['NSAMPLES']


        new_features_list = []
        # sampled_features_indices = []
        if len(groupers) > 0:
            for i in range(len(groupers)):
                radius = radii[i]
                nsample = nsamples[i]
                
                grouped_features = groupers[i](xyz,new_xyz,features)
                new_features = mlps[i](grouped_features)
                new_features_list += [new_features]
                # features = 


                # feature_indices = pointnet2_utils.ball_query(
                #     radius,
                #     nsample,
                #     xyz,
                #     new_xyz
                # ).type(torch.int)
                
                # group_features = pointnet2_utils.grouping_operation(
                #     features,
                #     feature_indices
                # )
                # new_features_list += [group_features]
                # sampled_features_indices += [feature_indices]
                ## lol this does the same thing as group_features :(

                # #! (B,C,npoints) -> (B,npoints,C)
                # features_swapped = features.permute(0,2,1)
                # F = features_swapped.shape[2] 

                # B,N,S = feature_indices.shape
                # #! (B,N,S) -> (B,N*S,1)
                # idx_squished = feature_indices.reshape(B,N*S).unsqueeze(2)
                # idx_expanded = idx_squished.expand((B,N*S,F)).long()

                # features_gathered = torch.gather(features_swapped,-1,idx_expanded)

                # final_features = features_gathered.reshape(B,N,S,F)
        else:
            print('this bit might not be working as intended')
            raise RuntimeError('reached wrong gathering step')
            # new_features = pointnet2_utils.gather_operation(features, sampled_idx_list)
            # new_features_list += [new_features]

        # if len(groupers) > 0:
        #     for i in range(len(groupers)):
        #         new_features = groupers[i](xyz, new_xyz, features)                
        #         new_features = mlps[i](new_features) 

        #         if pool_method == 'max_pool':
        #             new_features = F.max_pool2d(
        #                 new_features, kernel_size=[1, new_features.size(3)]
        #             )  # (B, mlp[-1], npoint, 1)
        #         elif pool_method == 'avg_pool':
        #             new_features = F.avg_pool2d(
        #                 new_features, kernel_size=[1, new_features.size(3)]
        #             )  # (B, mlp[-1], npoint, 1)
        #         else:
        #             raise NotImplementedError
        #         new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
        #         new_features_list.append(new_features)
        #     new_features = torch.cat(new_features_list, dim=1)
        # else:
        #     new_features = pointnet2_utils.gather_operation(features, sampled_idx_list).contiguous()

        # if confidence_layers is not None:
        #     cls_features = confidence_layers(new_features).transpose(1, 2)
            
        # else:
        #     cls_features = None

        # if sample_idx:
        #     return new_xyz, new_features, cls_features, sampled_idx_list
        # else:
        #     return new_xyz, new_features, cls_features
        return new_features_list


    def build_attention_modules(self):
        
        self.radarCrosslidar = nn.ModuleList()
        self.lidarCrossradar = nn.ModuleList()

        #! Assumes out MLP is same for lidar and radar
        MLPS = self.RADAR_MODULES['MLPS']
        for i in range(len(MLPS)):
            # dim = MLPS[i]
            dim = MLPS[i][6].out_channels
            self.radarCrosslidar.append(nn.MultiheadAttention(dim,8))
            self.lidarCrossradar.append(nn.MultiheadAttention(dim,8))            
                
    def attention_and_pool(self,radar_features:List[torch.Tensor],lidar_features:List[torch.Tensor]):

        


        B = radar_features[0].shape[0] # num batches
        G = radar_features[0].shape[2] # number of groups/clusters


        radar_outputs = []
        lidar_outputs = []
        for i in range(len(radar_features)):
            
            radar_attn_list = []
            lidar_attn_list = []
            for batch in range(B):            
                    # dims, Groups(batch), npoints -> npoints, groups, dims
                r2l_attn = self.radarCrosslidar[i]
                l2r_attn = self.lidarCrossradar[i]

                radar_feats = radar_features[i][batch].permute(2,1,0)
                lidar_feats = lidar_features[i][batch].permute(2,1,0)


                r2l_output, _ = r2l_attn(radar_feats,lidar_feats,lidar_feats)
                l2r_output, _ = l2r_attn(lidar_feats,radar_feats,radar_feats)
                radar_attn_list += [r2l_output.permute(1,2,0)]
                lidar_attn_list += [l2r_output.permute(1,2,0)]
            
            radar_outputs += [torch.stack(radar_attn_list)]
            lidar_outputs += [torch.stack(lidar_attn_list)]

        
        return lidar_outputs,radar_outputs



    def single_attention(self,q_features,kv_features,attention_modules):


        B = q_features[0].shape[0]

        ret = []
        for i in range(len(q_features)):
            attn_list = []

            for batch in range(B):
                Q = q_features[i][batch].permute(2,1,0)
                KV = kv_features[i][batch].permute(2,1,0)

                attn, weights = attention_modules[i](Q,KV,KV)
                
                attn_list += [attn.permute(2,1,0)]
            
            ret += [torch.stack(attn_list)]
        

        return ret

    def __init__(self, *,
                radar_settings,
                lidar_settings):
        super().__init__()

        assert len(radar_settings) == len(lidar_settings), "setting lengths dont match"


        self.RADAR_MODULES = self.build_sa_modules(*radar_settings) 
        self.LIDAR_MODULES = self.build_sa_modules(*lidar_settings)
        
        # Builds attention modules 
        self.build_attention_modules()


    def match_clusters(self,lidar_new_xyz,radar_new_xyz):


        
                                                                                    # selecting lidar point FOR EACH radar
        lidar_for_radar = pointnet2_utils.ball_query(self.RADAR_MODULES['RADII'][-1],1,lidar_new_xyz,radar_new_xyz)
                                                                                    # selecting radar point FOR EACH lidar
        radar_for_lidar = pointnet2_utils.ball_query(self.RADAR_MODULES['RADII'][-1],1,radar_new_xyz,lidar_new_xyz)
        
    
        return radar_for_lidar,lidar_for_radar

    def single_pool_and_aggregate(self,feature_list,pool_method,aggregation_layer):

        new_feature_list = []
        for i in range(len(feature_list)):
            features = feature_list[i]
            if pool_method == 'max_pool':
                new_features = F.max_pool2d(
                    features, kernel_size=[1, features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif pool_method == 'avg_pool':
                new_features = F.avg_pool2d(
                    features, kernel_size=[1, features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            new_feature_list += [new_features.squeeze(-1)]


        final_features = torch.cat(new_feature_list,dim=1) 
        final_features = aggregation_layer(final_features)
        
        return final_features

    def forward(self,
                lidar_xyz: torch.Tensor,
                radar_xyz: torch.Tensor,
                lidar_features: torch.Tensor = None,
                lidar_cls_features: torch.Tensor = None,
                lidar_new_xyz = None,
                lidar_ctr_xyz = None,
                radar_features: torch.Tensor = None,
                radar_cls_features: torch.Tensor = None,
                radar_new_xyz = None,
                radar_ctr_xyz = None,    
                ):

        '''
        lidar_xyz: [B,npoints_l,3]
        radar_xyz: [B,npoints_r,3]
        lidar_features: [B,1,npoints_l]
        radar_features: [B,2,npoints_r]
        '''

        

        #@ [B,512,3]         [B,512]
        radar_new_xyz,radar_sampled_idx_list = self.single_sample_and_gather(
                                                        self.RADAR_MODULES,
                                                        radar_xyz,
                                                        radar_features,
                                                        radar_cls_features,
                                                        radar_new_xyz,
                                                        radar_ctr_xyz)
        #@ [B,4096,3]        [B,4096]
        lidar_new_xyz,lidar_sampled_idx_list = self.single_sample_and_gather(
                                                        self.LIDAR_MODULES,
                                                        lidar_xyz,
                                                        lidar_features,
                                                        lidar_cls_features,
                                                        lidar_new_xyz,
                                                        lidar_ctr_xyz)
        
        # import numpy as np
        # np.save('lidar_xyz',lidar_xyz.cpu().numpy())
        # np.save('lidar_new_xyz',lidar_new_xyz.cpu().numpy())
        # np.save('radar_xyz',radar_xyz.cpu().numpy())
        # np.save('radar_new_xyz',radar_new_xyz.cpu().numpy())

        #@ [B,4096,1]     [B,512,1]
        radar_for_lidar,lidar_for_radar = self.match_clusters(lidar_new_xyz,radar_new_xyz)        


        # OUTPUT IS list of num_radii tensors


        #@ LIST[ (B,32,512,16),(B,64,512,32)]
        radar_gathered_features = self.single_gather_features(
                                            self.RADAR_MODULES,
                                            xyz=radar_xyz,
                                            new_xyz=radar_new_xyz,
                                            features=radar_features,
                                            sampled_idx_list = radar_sampled_idx_list)


        #@ LIST[ (B,32,4096,16),(B,64,4096,32)]
        lidar_gathered_features = self.single_gather_features(
                                            self.LIDAR_MODULES,
                                            xyz=lidar_xyz,
                                            new_xyz=lidar_new_xyz,
                                            features=lidar_features,
                                            sampled_idx_list = lidar_sampled_idx_list)

        assert len(radar_gathered_features) == len(lidar_gathered_features), "gathering error"

        ##!!!!! ONLY WORKS FOR BATCHSIZE 1 
        # 
        print('FEATURE MATCHING ONLY WORKS WITH BATCHSIZE 1 ') 
        lidar_idx = lidar_for_radar.squeeze(2).squeeze(0)
        radar_idx = radar_for_lidar.squeeze(2).squeeze(0)

        lidar_features_for_radar = [feature[:,:,lidar_idx.long(),:] for feature in lidar_gathered_features] 
        radar_features_for_lidar = [feature[:,:,radar_idx.long(),:] for feature in radar_gathered_features]



        lidar_features_for_radar_post_attn = self.single_attention(radar_gathered_features,lidar_features_for_radar,self.radarCrosslidar)
        radar_features_for_lidar_post_attn = self.single_attention(lidar_gathered_features,radar_features_for_lidar,self.lidarCrossradar)
        
        
        radar_final_features = [
            torch.concat((radar_gathered_features[i],lidar_features_for_radar_post_attn[i]),1) for i in range(len(lidar_features_for_radar_post_attn))
        ]
        
        lidar_final_features = [
            torch.concat((lidar_gathered_features[i],radar_features_for_lidar_post_attn[i]),1) for i in range(len(radar_features_for_lidar_post_attn))
        ]
        

        radar_final_features = self.single_pool_and_aggregate(
            radar_final_features,
            self.RADAR_MODULES['POOL_METHOD'],
            self.RADAR_MODULES['AGGREGATION_LAYER'])

        lidar_final_features = self.single_pool_and_aggregate(
            lidar_final_features,
            self.RADAR_MODULES['POOL_METHOD'],
            self.RADAR_MODULES['AGGREGATION_LAYER'])

        

        if self.RADAR_MODULES['CONFIDENCE_LAYERS'] is not None:
            lidar_cls_features = self.LIDAR_MODULES['CONFIDENCE_LAYERS'](lidar_final_features).transpose(1,2)
            radar_cls_features = self.RADAR_MODULES['CONFIDENCE_LAYERS'](radar_final_features).transpose(1,2)
            
        else:
            lidar_cls_features = None
            radar_cls_features = None
        # lidar_feat_post_attn,radar_feat_post_attn = self.attention_and_pool(radar_gathered_features,lidar_gathered_features)










        ret_dict = {
            'lidar': (lidar_new_xyz,lidar_final_features,lidar_cls_features),
            'radar': (radar_new_xyz,radar_final_features,radar_cls_features),
        }
        

        return ret_dict







class Vote_layer3DSSD(nn.Module):
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        shared_mlps = []
        for i in range(len(mlp_list)):
            shared_mlps.extend([
                nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                nn.BatchNorm1d(mlp_list[i]),
                nn.ReLU()
            ])
            pre_channel = mlp_list[i]
        self.mlp_modules = nn.Sequential(*shared_mlps)

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.min_offset = torch.tensor(max_translate_range).float().view(1, 1, 3)
        

    def forward(self, xyz, features):
        # print(f'mlp modules: {self.mlp_modules}')
        # print(f'ctr_reg mod: {self.ctr_reg}')

        # print(f'in Vote module, pt shape: {xyz.shape}')
        # print(f'in Vote module, feature shape: {features.shape}')

        new_features = self.mlp_modules(features)
        ctr_offsets = self.ctr_reg(new_features)
        ctr_offsets = ctr_offsets.transpose(1, 2)

        min_offset = self.min_offset.repeat((xyz.shape[0], xyz.shape[1], 1)).to(xyz.device)

        limited_ctr_offsets = torch.where(ctr_offsets < min_offset, min_offset, ctr_offsets)
        min_offset = -1 * min_offset
        limited_ctr_offsets = torch.where(limited_ctr_offsets > min_offset, min_offset, limited_ctr_offsets)
        xyz = xyz + limited_ctr_offsets
        # print('-'*70)
        # print(f'in Vote module, new pt (shd be same) shape: {xyz.shape}')
        # print(f'in Vote module, new feature shape: {new_features.shape}')
        # print(f'in Vote module, ctr_offset shape: {ctr_offsets.shape}')
        # print('='*70)
        return xyz, new_features, ctr_offsets







class Vote_layer(nn.Module):
    """ Light voting module with limitation"""
    def __init__(self, mlp_list, pre_channel, max_translate_range):
        super().__init__()
        self.mlp_list = mlp_list
        if len(mlp_list) > 0:
            for i in range(len(mlp_list)):
                shared_mlps = []

                shared_mlps.extend([
                    nn.Conv1d(pre_channel, mlp_list[i], kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_list[i]),
                    nn.ReLU()
                ])
                pre_channel = mlp_list[i]
            self.mlp_modules = nn.Sequential(*shared_mlps)
        else:
            self.mlp_modules = None

        self.ctr_reg = nn.Conv1d(pre_channel, 3, kernel_size=1)
        self.max_offset_limit = torch.tensor(max_translate_range).float() if max_translate_range is not None else None
       

    def forward(self, xyz, features):
        xyz_select = xyz
        features_select = features

        if self.mlp_modules is not None: 
            new_features = self.mlp_modules(features_select) #([4, 256, 256]) ->([4, 128, 256])            
        else:
            new_features = features_select
        
        ctr_offsets = self.ctr_reg(new_features) #[4, 128, 256]) -> ([4, 3, 256])

        ctr_offsets = ctr_offsets.transpose(1, 2)#([4, 256, 3])
        feat_offets = ctr_offsets[..., 3:]
        new_features = feat_offets
        ctr_offsets = ctr_offsets[..., :3]
        if torch.isnan(ctr_offsets).sum() > 0:
            raise RuntimeError('Nan in ctr_offsets')
        if self.max_offset_limit is not None:
            max_offset_limit = self.max_offset_limit.view(1, 1, 3)            
            max_offset_limit = self.max_offset_limit.repeat((xyz_select.shape[0], xyz_select.shape[1], 1)).to(xyz_select.device) #([4, 256, 3])
      
            limited_ctr_offsets = torch.where(ctr_offsets > max_offset_limit, max_offset_limit, ctr_offsets)
            min_offset_limit = -1 * max_offset_limit
            limited_ctr_offsets = torch.where(limited_ctr_offsets < min_offset_limit, min_offset_limit, limited_ctr_offsets)
            vote_xyz = xyz_select + limited_ctr_offsets
        else:
            vote_xyz = xyz_select + ctr_offsets
        if torch.isnan(vote_xyz).sum() > 0:
            raise RuntimeError('Nan in vote_xyz')
        return vote_xyz, new_features, xyz_select, ctr_offsets



class Double_Vote_layer(nn.Module):
    """
    Alias for voting module with two point clouds
    """

    def __init__(self,
                lidar_mlp_list,
                lidar_prechannel,
                lidar_max_translate_range,
                radar_mlp_list,
                radar_prechannel,
                radar_max_translate_range,
                ):

        super().__init__()
        self.lidar_vote_layer = Vote_layer(lidar_mlp_list,lidar_prechannel,lidar_max_translate_range)
        self.radar_vote_layer = Vote_layer(radar_mlp_list,radar_prechannel,radar_max_translate_range)
        


    def forward(self, lidar_xyz, radar_xyz, lidar_features, radar_features):

        lidar_vote_xyz, lidar_new_features, lidar_xyz_select, lidar_ctr_offsets = self.lidar_vote_layer(lidar_xyz,lidar_features)
        radar_vote_xyz, radar_new_features, radar_xyz_select, radar_ctr_offsets = self.radar_vote_layer(radar_xyz,radar_features)


        ret_dict = {
            'lidar': (lidar_vote_xyz, lidar_new_features, lidar_xyz_select, lidar_ctr_offsets),
            'radar': (radar_vote_xyz, radar_new_features, radar_xyz_select, radar_ctr_offsets)
        }

        return ret_dict



class Fusion_Layer(nn.Module):
    def __init__(self,
                *,
                npoints: int,
                sample_type: str,
                nsamples = 5,
                mlps: List[int],
                use_xyz: bool = True,
                aggregation_mlp: List[int],
                confidence_mlp: List[int],
                num_class):
        '''
        :param 
        '''
        super().__init__()
        self.num_ctr = npoints
        self.sample_type = sample_type
        self.nsamples = nsamples
        self.mlp = nn.ModuleList()
        shared_mlps = []
        for k in range(len(mlps) - 1):
            shared_mlps.extend([
                    nn.Conv2d(mlps[k], mlps[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(mlps[k + 1]),
                    nn.ReLU()
                ])
        self.mlps.append(nn.Sequential(*shared_mlps))

    def forward(self, fg_xyz, fg_features, bg_xyz, bg_features):
        '''
        :param fg_xyz [B, N, 3]
        :param fg_features [B, N, C]
        :param bg_xyz [B, N, 3]
        :param bg_features [B, N, C]
        '''

        # take bg features and xyz, sample it using D-FPS, F-FPS or ctr-aware
        # cat each fg_feat with selected bg_feat
        # using MLP to calculate weight
        # new_fg_feat = weighted sum of bg + itself
        fg_xyz_flipped = fg_xyz.transpose(1, 2).contiguous() 
        bg_xyz_flipped = bg_xyz.transpose(1, 2).contiguous() 
        bg_feat_flipped = bg_features.transpose(1, 2).contiguous()
        if self.sample_type == 'D-FPS':
            sample_idx = pointnet2_utils.furthest_point_sample(bg_xyz_flipped, self.nsamples)
        elif ('cls' in self.sample_type) or ('ctr' in self.sample_type):
            pass
        else:
            raise NotImplementedError
        # B x nsample x 3 
        sample_bg_xyz = pointnet2_utils.gather_operation(bg_xyz_flipped, sample_idx).transpose(1, 2).contiguous()
        # B x C x nsample
        sample_bg_feat = pointnet2_utils.gather_operation(bg_feat_flipped, sample_idx).contiguous()
        # B x C x nsample -> B x C x nsample x N_ctr
        sample_bg_feat = sample_bg_feat.unsqueeze(-1).repeat([1, 1, 1, self.num_ctr])
        
        fg_features_self = fg_features_self.unsqueeze(2).repeat([1, 2, 1, 1]) # B x 2C x 1 x N_ctr
        fusion_feats = torch.cat((sample_bg_feat, fg_features), dim=1) # B x 2C x nsample x N_ctr
        fusion_feats = torch.cat((fusion_feats, fg_features_self), dim=2) # B x 2C x nsample+1 x N_ctr
        weights = self.weights_mlp(fusion_feats) # B x 1 x nsample+1 x N_ctr
        
        fg_features_expand = fg_features.unsqueeze(2) # B x C x 1 x N_ctr
        fused_feat_origin = torch.cat((fg_features_expand, sample_bg_feat), dim=2) # B x C x nsample+1 x N_ctr
        final_feats = weights * fused_feat_origin
        final_feats = final_feats.sum(dim=2).unsqueeze(2) # B x C x N_ctr
        return fg_xyz, final_feats




class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstraction layer"""

    def __init__(self, *, mlp: List[int], npoint: int = None, radius: float = None, nsample: int = None,
                 bn: bool = True, use_xyz: bool = True, pool_method='max_pool'):
        """
        :param mlp: list of int, spec of the pointnet before the global max_pool
        :param npoint: int, number of features
        :param radius: float, radius of ball
        :param nsample: int, number of samples in the ball query
        :param bn: whether to use batchnorm
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        """
        super().__init__(
            mlps=[mlp], npoint=npoint, radii=[radius], nsamples=[nsample], bn=bn, use_xyz=use_xyz,
            pool_method=pool_method
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another"""

    def __init__(self, *, mlp: List[int], bn: bool = True):
        """
        :param mlp: list of int
        :param bn: whether to use batchnorm
        """
        super().__init__()

        shared_mlps = []
        for k in range(len(mlp) - 1):
            shared_mlps.extend([
                nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
                nn.BatchNorm2d(mlp[k + 1]),
                nn.ReLU()
            ])
        self.mlp = nn.Sequential(*shared_mlps)

    def forward(
            self, unknown: torch.Tensor, known: torch.Tensor, unknow_feats: torch.Tensor, known_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        :param unknown: (B, n, 3) tensor of the xyz positions of the unknown features
        :param known: (B, m, 3) tensor of the xyz positions of the known features
        :param unknow_feats: (B, C1, n) tensor of the features to be propigated to
        :param known_feats: (B, C2, m) tensor of features to be propigated
        :return:
            new_features: (B, mlp[-1], n) tensor of the features of the unknown features
        """
        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))

        if unknow_feats is not None:
            new_features = torch.cat([interpolated_feats, unknow_feats], dim=1)  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


class AttentiveSAModule(nn.Module):
    def __init__(self, *,
                 npoint_list: List[int],
                 radii: List[float],
                 nsamples: List[int],
                 out_channel = -1, 
                 mlps: List[List[int]],                 
                 use_xyz: bool = True,
                 dilated_group=False,
                 pool_method='max_pool',
                 attention_type='PCT',
                 pos_encoding=False,
                 ):
        """
        :param npoint_list: list of int, number of samples for every sampling type
        :param radii: list of float, list of radii to group with
        :param nsamples: list of int, number of samples in each ball query
        :param mlps: list of list of int, spec of the pointnet before the global pooling for each scale
        :param use_xyz:
        :param pool_method: max_pool / avg_pool
        :param attention_type: point cloud transformer layer
        :param pos_encoding: whether to use position encoding
        """
        super().__init__()
       
        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint_list = npoint_list
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.dilated_group = dilated_group
        # out_channels = 0

        self.groupers = self.get_groupers(radii, nsamples, use_xyz, npoint_list)
        self.xyz_groupers = self.get_groupers(radii, nsamples, False, npoint_list)
        out_channels = self.get_mlps(radii, mlps, use_xyz) # change Conv2d -> Conv1d
        self.pool_method = pool_method
        if attention_type == 'PCT':
            self.attention_layer = PCT_Layer(out_channels, use_pose_encoding=pos_encoding)
        else:
            raise NotImplementedError
        if out_channel != -1 and len(self.mlps) > 0:
            in_channel = 0
            for mlp_tmp in mlps:
                in_channel += mlp_tmp[-1]
            shared_mlps = []
            shared_mlps.extend([
                nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channel),
                nn.ReLU()
            ])
            self.out_aggregation = nn.Sequential(*shared_mlps)

    def get_groupers(self, radii, nsamples, use_xyz, npoint_list):
        use_xyz = False # disable use_xyz
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            groupers = nn.ModuleList()
            if self.dilated_group:
                if i == 0:
                    min_radius = 0.
                else:
                    min_radius = radii[i-1]
                groupers.append(
                    pointnet2_utils.QueryDilatedAndGroup(
                        radius, min_radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            else:
                groupers.append(
                    pointnet2_utils.QueryAndGroup(
                        radius, nsample, use_xyz=use_xyz)
                    if npoint_list is not None else pointnet2_utils.GroupAll(use_xyz)
                )
            return groupers
    def get_mlps(self, radii, mlps, use_xyz):
        out_channels = 0
        for i in range(len(radii)):
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            shared_mlps = []
            for k in range(len(mlp_spec) - 1):
                shared_mlps.extend([
                    nn.Conv1d(mlp_spec[k], mlp_spec[k + 1],
                              kernel_size=1, bias=False),
                    nn.BatchNorm1d(mlp_spec[k + 1]),
                    nn.ReLU()
                ])
            self.mlps.append(nn.Sequential(*shared_mlps))
            out_channels += mlp_spec[-1]
        return out_channels
        
    def forward(self, xyz: torch.Tensor, features: torch.Tensor, ctr_xyz: torch.Tensor):
        """
        :param xyz: (B, N, 3) tensor of the xyz coordinates of the features
        :param features: (B, C, N) tensor of the descriptors of the the features
        :param ctr_xyz: (B, M, 3) tensor of the xyz coordinates of the sampled points
        :return:
            new_xyz: (B, npoint, 3) tensor of the new features' xyz
            new_features: (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
            cls_features: (B, npoint, num_class) tensor of confidence (classification) features
        """
        xyz_transpose = xyz.transpose(1, 2).contiguous()
        # new_features_list = []
        new_xyz = ctr_xyz
        ctr_xyz_transpose = ctr_xyz.transpose(1, 2).contiguous() # (B, 3, M)
        features_input = torch.cat((xyz_transpose, features), dim=1)
        if len(self.groupers) > 0:
            
                # new_features = self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)
                # new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
                # pass through Conv1d mlp first would save computation time
            new_features = self.mlps[0](features_input) # features: (B, C, N) -> (B, mlp[i], N)
            group_features = self.groupers[0](xyz, new_xyz, new_features)  # (B, mlp[i], N, nsample)
            group_xyz = self.xyz_groupers[0](xyz, new_xyz, xyz_transpose)
            group_ctr = self.xyz_groupers[0](new_xyz, new_xyz, ctr_xyz_transpose)
            att_features = self.attention_layer.forward(group_xyz, group_features, group_ctr)
            # new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)
            if self.pool_method == 'max_pool':
                att_features = F.max_pool2d(
                    att_features, kernel_size=[1, att_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            elif self.pool_method == 'avg_pool':
                att_features = F.avg_pool2d(
                    att_features, kernel_size=[1, att_features.size(3)]
                )  # (B, mlp[-1], npoint, 1)
            else:
                raise NotImplementedError
            att_features = att_features.squeeze(-1) # (B, C, n_ctr)
        else:
            raise RuntimeError('empty groupers!')
        
        att_features = self.out_aggregation(att_features)

        return new_xyz, att_features

class transformer_aggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz: torch.tensor, features: torch.tensor, ctr_xyz: torch.tensor):
        # grouping
        
        # calculate attention value

        # calculate 
        pass

class PCT_Layer(nn.Module):
    def __init__(self, in_ch, inter_ch=8, use_pose_encoding=False):
        super().__init__()
        
        if use_pose_encoding:
            self.pose_encoding = nn.Conv1d(3, in_ch)
            in_feat = in_ch
        else:
            self.pose_encoding = None
            in_feat = in_ch + 3
        
        self.conv_q = nn.Conv2d(in_feat, inter_ch, 1, 1, bias=False)
        self.conv_k = nn.Conv2d(in_feat, inter_ch, 1, 1, bias=False)
        self.conv_v = nn.Conv2d(in_feat, in_ch, 1, 1, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.final_conv = nn.Conv2d(in_ch, in_ch, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()
        

    def forward(self, group_xyz: torch.tensor, group_features: torch.tensor, group_ctr_xyz: torch.tensor):
        """
        :param xyz: (B, 3, N, nsample) tensor of the xyz coordinates of the features
        :param features: (B, C, N, nsample) tensor of the descriptors of the the features
        "param ctr_xyz: (B, 3, N, nsample) tensor of the xyz coordinates of the centers 
        :return:
            new_features: (B, C, N, nsample) tensor of the new_features descriptors
        """

        relative_pos = group_ctr_xyz - group_xyz
        if self.pose_encoding is None:
            ip_feat = torch.cat((group_features, relative_pos), dim=1) # (B, C+3, N, nsample)
        else:
            ip_feat = group_features + self.pose_encoding(relative_pos)
        
        q = self.conv_q(ip_feat) # (B, C_d, N, nsample)
        k = self.conv_k(ip_feat)
        v = self.conv_v(ip_feat) # (B, C, N, nsample)
        k = torch.transpose(k, 1, 2) # (B, N, C_d, nsample)
        att_map = torch.einsum('bncm,bcjm->bnjm', k, q) # (B, N, N, nsample)

        att_map = self.sm(att_map)

        # L1 normalize
        att_map_sum = torch.sum(att_map, dim=1, keepdim=True)
        att_map = att_map / (att_map_sum + 1e-9) 

        # apply attention
        att_feat = torch.einsum('bnkm,bcnm->bckm', att_map, v) # (B, C, N, nsample)
        
        offset_feat = att_feat - group_features
        lbr_feat = self.relu(self.bn(self.final_conv(offset_feat))) # LBR
        result = lbr_feat + group_features
        return result

if __name__ == "__main__":
    pass
