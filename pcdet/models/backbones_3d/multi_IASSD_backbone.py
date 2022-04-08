from sre_constants import BRANCH
from turtle import forward
import torch
import torch.nn as nn
import os
from .IASSD_backbone import IASSD_Backbone
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules

class multi_IASSD_Backbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        '''
        use multi backbone and fuse the point feature with a SA module
        '''
        self.model_cfg = model_cfg
        self.branch_num = model_cfg.get('BRANCH_NUM', None)
        self.num_class = model_cfg.num_class

        if self.branch_num is None:
            raise RuntimeError
        self.module_list = nn.ModuleList()
        for id in range(self.branch_num):
            self.module_list.append(IASSD_Backbone_branch(
                self.model_cfg,
                input_channels,
                id
            ))
        self.SA_modules = nn.ModuleList()
        fusion_cfg = self.model_cfg.FUSION_CONFIG
        self.ctr_idx_list = fusion_cfg.CTR_INDEX
        self.layer_inputs = fusion_cfg.LAYER_INPUT
        self.aggregation_mlps = fusion_cfg.get('AGGREGATION_MLPS', None)
        assert len(self.layer_inputs) == 1
        mlps = fusion_cfg.MLPS[-1].copy()
        channel_in = self.module_list[-1].num_point_features
        channel_out = 0
        for idx in range(mlps.__len__()):
            mlps[idx] = [channel_in] + mlps[idx]
            channel_out += mlps[idx][-1]

        if self.aggregation_mlps and self.aggregation_mlps[-1]:
            aggregation_mlp = self.aggregation_mlps[-1].copy()
            if aggregation_mlp.__len__() == 0:
                aggregation_mlp = None
            else:
                channel_out = aggregation_mlp[-1]
        else:
            aggregation_mlp = None

        self.SA_modules.append(
            pointnet2_modules.PointnetSAModuleMSG_WithSampling(
                npoint_list=fusion_cfg.NPOINT_LIST[-1],
                sample_range_list=fusion_cfg.SAMPLE_RANGE_LIST[-1],
                sample_type_list=[],
                radii=fusion_cfg.RADIUS_LIST[-1],
                nsamples=fusion_cfg.NSAMPLE_LIST[-1],
                mlps=mlps,
                use_xyz=True,                                                
                dilated_group=False,
                aggregation_mlp=aggregation_mlp,
                confidence_mlp=None,
                num_class = self.num_class
            )
        )

        self.output_list = [
            'ctr_offsets', 
            'centers', 
            'centers_origin', 
            'ctr_batch_idx',
            'encoder_xyz',
            'encoder_coords',
            'sa_ins_preds',
            'encoder_features'] 

        # compatible to framework
        self.num_point_features = channel_out

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        for m in self.module_list:
            batch_dict = m(batch_dict)
        
        # combine all the output
        result_dict = {}
        for id in range(self.branch_num):
            id_string = str(id) + '_'
            for key in self.output_list:
                branch_result = batch_dict[id_string + key]
                if key == 'encoder_features':
                    cat_dim = 2
                elif key in ['ctr_batch_idx', 'centers', 'centers_origin', 'ctr_offsets']:
                    cat_dim = 0
                else:
                    cat_dim = 1
                if result_dict.get(key, None) is None:
                    result_dict[key] = branch_result
                else:
                    temp_result = result_dict[key]
                    if type(temp_result) == list:
                        for result_idx in range(len(temp_result)):
                            try:
                                temp_result[result_idx] = \
                                    torch.cat((temp_result[result_idx], branch_result[result_idx]), dim=cat_dim)
                            except:
                                temp_result[result_idx] = []
                            
                    else:
                        temp_result = torch.cat((temp_result, batch_dict[id_string + key]), dim=cat_dim)
                    result_dict[key] = temp_result
        # add the combined result to batch_dict
        for key in result_dict.keys():
            batch_dict[key] = result_dict[key]

        # remove branch result
        for key in self.output_list:
            for id in range(self.branch_num):
                temp_key = str(id) + '_' + key
                batch_dict.pop(temp_key)

        # Final SA module for center_features
        encoder_xyz = batch_dict['encoder_xyz']
        encoder_coords = batch_dict['encoder_coords']
        sa_ins_preds = batch_dict['sa_ins_preds']
        encoder_features = batch_dict['encoder_features']

        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_input = encoder_xyz[self.layer_inputs[-1]]
        feature_input = encoder_features[self.layer_inputs[-1]]
        ctr_xyz = encoder_xyz[self.ctr_idx_list[-1]]
        li_xyz, li_features, li_cls_pred = self.SA_modules[0](xyz_input, feature_input, None, ctr_xyz=ctr_xyz)

        li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
        encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
        encoder_features.append(li_features)
        sa_ins_preds.append([])
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['encoder_xyz'] = encoder_xyz
        batch_dict['encoder_coords'] = encoder_coords
        batch_dict['sa_ins_preds'] = sa_ins_preds
        batch_dict['encoder_features'] = encoder_features
        return batch_dict

class IASSD_Backbone_branch(IASSD_Backbone):
    def __init__(self, model_cfg, input_channels, id, **kwargs):
        super().__init__(model_cfg, input_channels, **kwargs)
        self.id = id
    
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        li_cls_pred = None
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]

            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = encoder_xyz[self.ctr_idx_list[i]] if self.ctr_idx_list[i] != -1 else None
                li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)

            elif self.layer_types[i] == 'Vote_Layer': #i=4
                li_xyz, li_features, xyz_select, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_select
                center_origin_batch_idx = batch_idx.view(batch_size, -1)[:, :centers_origin.shape[1]]
                encoder_coords.append(torch.cat([center_origin_batch_idx[..., None].float(),centers_origin.view(batch_size, -1, 3)],dim =-1))
                    
            encoder_xyz.append(li_xyz)
            li_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
            encoder_coords.append(torch.cat([li_batch_idx[..., None].float(),li_xyz.view(batch_size, -1, 3)],dim =-1))
            encoder_features.append(li_features)            
            if li_cls_pred is not None:
                li_cls_batch_idx = batch_idx.view(batch_size, -1)[:, :li_cls_pred.shape[1]]
                sa_ins_preds.append(torch.cat([li_cls_batch_idx[..., None].float(),li_cls_pred.view(batch_size, -1, li_cls_pred.shape[-1])],dim =-1)) 
            else:
                sa_ins_preds.append([])
           
        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :li_xyz.shape[1]]
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)

        # save results
        id_string = str(self.id) + '_'
        batch_dict[id_string + 'ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)

        batch_dict[id_string + 'centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict[id_string + 'centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)

        
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict[id_string + 'centers_features'] = center_features
        batch_dict[id_string + 'ctr_batch_idx'] = ctr_batch_idx
        batch_dict[id_string + 'encoder_xyz'] = encoder_xyz
        batch_dict[id_string + 'encoder_coords'] = encoder_coords
        batch_dict[id_string + 'sa_ins_preds'] = sa_ins_preds
        batch_dict[id_string + 'encoder_features'] = encoder_features # not used later?
        return batch_dict