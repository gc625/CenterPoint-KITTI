import torch
import torch.nn as nn
from easydict import EasyDict
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
import os



class BERTSSD_Backbone(nn.Module):
    '''
    BERT-SSD backbone
    '''

    # def build_both_sa_layers(
    #     self, 
    #     lidar_sa_cfg: EasyDict, 
    #     radar_sa_cfg: EasyDict) -> tuple(nn.ModuleList(), nn.ModuleList()):

    #     lidar_sa_modules = self._build_sa_layers(lidar_sa_cfg)
    #     radar_sa_modules = self._build_sa_layers(radar_sa_cfg)

    #     return lidar_sa_modules,radar_sa_modules
    
    # def _build_sa_layers(self, sa_cfg: EasyDict) -> nn.ModuleList():

    #     SA_modules = nn.ModuleList()

    #     channel_in = sa_cfg.POINT_INPUT_FEATURES - 3 
    #     channel_out_list = [channel_in]
        
        
    #     layer_types = sa_cfg.LAYER_TYPE
    #     ctr_idx_list = sa_cfg.CTR_INDEX
    #     layer_inputs = sa_cfg.LAYER_INPUT
    #     aggregation_mlps = sa_cfg.get('AGGREGATION_MLPS', None)
    #     confidence_mlps = sa_cfg.get('CONFIDENCE_MLPS', None)lidar
    #                     channel_out = aggregation_mlp[-1]
    #             else: 
    #                 aggregation_mlp = None

    #             # * Confidence MLP For center-aware sampling
    #             if confidence_mlps and confidence_mlps[k]:
    #                 confidence_mlp = confidence_mlps[k].copy()
    #                 if confidence_mlp.__len__() == 0:
    #                     confidence_mlp = None
    #             else:
    #                 confidence_mlp = None
    #             SA_modules.append(
    #                 pointnet2_modules.POint

    #             )


    #     return SA_modules
        

#         [B,512,5]          [B,16k,4]
#           Radar              Lidar 
#             |                  |
#@        (sampling)         (sampling)
#             |                  |
#      [xyz,feats,idx]    [xyz,feats,idx]
#             |                  |
#             |----Cross-attn--->|
#             |<---Cross-attn----|
#             |                  |
#@        (grouping)         (grouping)
#             |                  |
#@        (aggregate)        (aggregate)
#             |                  |
#!        (sampling)         (sampling)
#             |                  |
#      [xyz,feats,idx]    [xyz,feats,idx]
#             |                  |
#             |----Cross-attn--->|
#             |<---Cross-attn----|
#             |                  |
#!        (grouping)         (grouping)
#             |                  |
#!        (aggregate)        (aggregate)              
#             |                  |
#?        (sampling)         (sampling)
#             |                  |
#      [xyz,feats,idx]    [xyz,feats,idx]
#             |                  |
#             |----Cross-attn--->|
#             |<---Cross-attn----|
#             |                  |
#?        (grouping)         (grouping)
#             |                  |
#?        (aggregate)        (aggregate)   
#             |                  |
#@        (sampling)         (sampling)
#             |                  |
#      [xyz,feats,idx]    [xyz,feats,idx]
#             |                  |
#             |----Cross-attn--->|
#             |<---Cross-attn----|
#             |                  |
#@        (grouping)         (grouping)
#             |                  |
#@        (aggregate)        (aggregate)
#             |                  |
#!       (VOTE_LAYER)       (VOTE_LAYER)
#             |                  |
#      [xyz,feats,idx]    [xyz,feats,idx]
#             |---------|--------|
#                       |
#                (match &  MLP)
#                       |
#                 [B,256,512?]
#                       |
#                 (IASSD_HEAD)

    def build_MMSA_layers(self,lidar_sa_cfg,radar_sa_cfg,disable_cross_attn,concat_attn_ft_dim,nheads):

        
        radar_npoint_list = radar_sa_cfg.NPOINT_LIST
        radar_sample_range_list = radar_sa_cfg.SAMPLE_RANGE_LIST
        radar_sample_type_list = radar_sa_cfg.SAMPLE_METHOD_LIST
        radar_radii = radar_sa_cfg.RADIUS_LIST
        radar_nsamples = radar_sa_cfg.NSAMPLE_LIST
        radar_mlps = radar_sa_cfg.MLPS
        radar_use_xyz = True
        radar_dilated_group = radar_sa_cfg.DILATED_GROUP
        radar_layer_types = radar_sa_cfg.LAYER_TYPE
        radar_ctr_idx_list = radar_sa_cfg.CTR_INDEX
        radar_layer_inputs = radar_sa_cfg.LAYER_INPUT
        radar_aggregation_mlps = radar_sa_cfg.get('AGGREGATION_MLPS', None)
        radar_confidence_mlps = radar_sa_cfg.get('CONFIDENCE_MLPS', None)
        radar_max_translate_range = radar_sa_cfg.get('MAX_TRANSLATE_RANGE', None)

        lidar_npoint_list = lidar_sa_cfg.NPOINT_LIST
        lidar_sample_range_list = lidar_sa_cfg.SAMPLE_RANGE_LIST
        lidar_sample_type_list = lidar_sa_cfg.SAMPLE_METHOD_LIST
        lidar_radii = lidar_sa_cfg.RADIUS_LIST
        lidar_nsamples = lidar_sa_cfg.NSAMPLE_LIST
        lidar_mlps = lidar_sa_cfg.MLPS
        lidar_use_xyz = True
        lidar_dilated_group = lidar_sa_cfg.DILATED_GROUP
        lidar_layer_types = lidar_sa_cfg.LAYER_TYPE
        lidar_ctr_idx_list = lidar_sa_cfg.CTR_INDEX
        lidar_layer_inputs = lidar_sa_cfg.LAYER_INPUT
        lidar_aggregation_mlps = lidar_sa_cfg.get('AGGREGATION_MLPS', None)
        lidar_confidence_mlps = lidar_sa_cfg.get('CONFIDENCE_MLPS', None)
        lidar_max_translate_range = lidar_sa_cfg.get('MAX_TRANSLATE_RANGE', None)


        radar_channel_out_list = [radar_sa_cfg.POINT_INPUT_FEATURES - 3] #TODO: CHECK
        lidar_channel_out_list = [lidar_sa_cfg.POINT_INPUT_FEATURES - 3] #TODO: CHECK


        for k in range(radar_nsamples.__len__()):


            #@ Setting channel input  
            # Radar: 5-3 -> 64 -> 128 -> 256 -> 256 -> 256  
            # LiDAR: 4-3 -> 64 -> 128 -> 256 -> 256 -> 256
            
            if isinstance(radar_layer_inputs[k], list): ###
                radar_channel_in = radar_channel_out_list[radar_layer_inputs[k][-1]]
                lidar_channel_in = lidar_channel_out_list[radar_layer_inputs[k][-1]]

            else:
                radar_channel_in = radar_channel_out_list[radar_layer_inputs[k]]
                lidar_channel_in = lidar_channel_out_list[radar_layer_inputs[k]]

            if radar_layer_types[k] == 'SA_Layer':
                radar_mlp = radar_mlps[k].copy()
                lidar_mlp = lidar_mlps[k].copy()

                channel_out = 0
                for idx in range(radar_mlp.__len__()):
                    radar_mlp[idx] = [radar_channel_in] + radar_mlp[idx]
                    lidar_mlp[idx] = [lidar_channel_in] + lidar_mlp[idx] 
                    
                    channel_out += radar_mlp[idx][-1]

                if radar_aggregation_mlps and radar_aggregation_mlps[k]:
                    radar_aggregation_mlp = radar_aggregation_mlps[k].copy()
                    lidar_aggregation_mlp = lidar_aggregation_mlps[k].copy()

                    if radar_aggregation_mlp.__len__() == 0:
                        radar_aggregation_mlp = None
                        lidar_aggregation_mlp = None
                    else:
                        channel_out = radar_aggregation_mlp[-1]
                else:
                    radar_aggregation_mlp = None
                    lidar_aggregation_mlp = None
                

                if radar_confidence_mlps and radar_confidence_mlps[k]:
                    radar_confidence_mlp = radar_confidence_mlps[k].copy()
                    lidar_confidence_mlp = lidar_confidence_mlps[k].copy()
                    
                    if radar_confidence_mlp.__len__() == 0:
                        radar_confidence_mlp = None
                        lidar_confidence_mlp = None
                else:
                    radar_confidence_mlp = None
                    lidar_confidence_mlp = None
                

                radar_settings = [
                    radar_npoint_list[k],
                    radar_sample_range_list[k],
                    radar_sample_type_list[k],
                    radar_radii[k],
                    radar_nsamples[k],
                    radar_mlp,
                    radar_use_xyz,
                    radar_dilated_group[k],
                    False, # REVERSE,
                    'max_pool', #pool method
                    False, # sample_idx
                    False, # use pooling weight
                    radar_aggregation_mlp,
                    radar_confidence_mlp,
                    self.num_class
                ]

                lidar_settings = [
                    lidar_npoint_list[k],
                    lidar_sample_range_list[k],
                    lidar_sample_type_list[k],
                    lidar_radii[k],
                    lidar_nsamples[k],
                    lidar_mlp,
                    lidar_use_xyz,
                    lidar_dilated_group[k],
                    False, # REVERSE,
                    'max_pool', #pool method
                    False, # sample_idx
                    False, # use pooling weight
                    lidar_aggregation_mlp,
                    lidar_confidence_mlp,
                    self.num_class,
                ]
                
                
                self.MMSA_modules.append(
                    pointnet2_modules.MMSAModuleMSG_WithSampling(
                        radar_settings=radar_settings,
                        lidar_settings=lidar_settings,
                        disable_cross_attn = disable_cross_attn,
                        concat_attn_ft_dim = concat_attn_ft_dim,
                        n_heads= nheads[k]
                    )
                )

            elif radar_layer_types[k] == 'Vote_Layer':
                
                self.MMSA_modules.append(
                    pointnet2_modules.Double_Vote_layer(
                        lidar_mlp_list=lidar_mlps[k],
                        lidar_prechannel=lidar_channel_out_list[lidar_layer_inputs[k]],
                        lidar_max_translate_range=lidar_max_translate_range,
                        radar_mlp_list=radar_mlps[k],
                        radar_prechannel=radar_channel_out_list[radar_layer_inputs[k]],
                        radar_max_translate_range=radar_max_translate_range,    
                    )
                )
            radar_channel_out_list += [channel_out]
            lidar_channel_out_list += [channel_out]

        self.radar_num_point_features = channel_out
        self.lidar_num_point_features = channel_out
        self.radar_layer_inputs = radar_layer_inputs
        self.lidar_layer_inputs = lidar_layer_inputs
        self.lidar_layer_types = lidar_layer_types
        self.radar_layer_types = radar_layer_types
        self.lidar_ctr_idx_list = lidar_ctr_idx_list  
        self.radar_ctr_idx_list = radar_ctr_idx_list


    def __init__(self,model_cfg: EasyDict, input_channels: int,**kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = model_cfg.num_class

        # self.SA_modules = nn.ModuleList()
        
        self.MMSA_modules = nn.ModuleList()


        self.num_point_features = 4  
        # Subtract 3 for xyz
        channel_in = input_channels - 3 
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        lidar_sa_cfg,radar_sa_cfg = self.model_cfg.LIDAR_SA_CONFIG,self.model_cfg.RADAR_SA_CONFIG
        
        self.disable_cross_attn = self.model_cfg.DISABLE_CROSS_ATTENTION
        self.concat_attn_ft_dim = self.model_cfg.CONCAT_ATTN_FT_DIM
        self.nheads = self.model_cfg.NHEADS
        self.build_MMSA_layers(lidar_sa_cfg,radar_sa_cfg,self.disable_cross_attn,self.concat_attn_ft_dim,self.nheads)
        # lidar_SA_modules,radar_SA_modules = self.build_both_sa_layers(lidar_sa_cfg,radar_sa_cfg)
    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features
        
    def split_into_batch(self,points,batch_size):
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###

        return batch_idx,xyz,features
    
    def forward(self,batch_dict):

        batch_size = batch_dict['batch_size']
        lidar_points = batch_dict['points']
        radar_points = batch_dict['attach']

        lidar_batch_idx,lidar_xyz,lidar_features = self.split_into_batch(lidar_points,batch_size)
        radar_batch_idx,radar_xyz,radar_features = self.split_into_batch(radar_points,batch_size)


        lidar_encoder_xyz, lidar_encoder_features, lidar_sa_ins_preds = [lidar_xyz], [lidar_features], []
        radar_encoder_xyz, radar_encoder_features, radar_sa_ins_preds = [radar_xyz], [radar_features], []

        lidar_encoder_coords = [torch.cat([lidar_batch_idx.view(batch_size,-1,1),lidar_xyz],dim=-1)]
        radar_encoder_coords = [torch.cat([radar_batch_idx.view(batch_size,-1,1),radar_xyz],dim=-1)]

        lidar_li_cls_pred = None
        radar_li_cls_pred = None


        for i in range(len(self.MMSA_modules)):
            assert self.radar_layer_inputs[i] == self.lidar_layer_inputs[i], "layer inputs are different"
            assert self.radar_layer_types[i] == self.lidar_layer_types[i], "Layer type doesnt match"  
            assert self.lidar_ctr_idx_list[i] == self.radar_ctr_idx_list[i], "ctr idx is diff"

            lidar_xyz_input = lidar_encoder_xyz[self.lidar_layer_inputs[i]]
            radar_xyz_input = radar_encoder_xyz[self.radar_layer_inputs[i]]
                
            lidar_feature_input = lidar_encoder_features[self.lidar_layer_inputs[i]]
            radar_feature_input = radar_encoder_features[self.radar_layer_inputs[i]]

            if self.lidar_layer_types[i] == 'SA_Layer':
                lidar_ctr_xyz = lidar_encoder_xyz[self.lidar_ctr_idx_list[i]] if self.lidar_ctr_idx_list[i] != -1 else None
                radar_ctr_xyz = radar_encoder_xyz[self.lidar_ctr_idx_list[i]] if self.lidar_ctr_idx_list[i] != -1 else None


                ret = self.MMSA_modules[i](
                        lidar_xyz_input,
                        radar_xyz_input,
                        lidar_feature_input,
                        lidar_li_cls_pred,
                        None,
                        lidar_ctr_xyz,
                        radar_feature_input,
                        radar_li_cls_pred,
                        None,
                        radar_ctr_xyz)

                lidar_li_xyz, lidar_li_features, lidar_li_cls_pred = ret['lidar']
                radar_li_xyz, radar_li_features, radar_li_cls_pred = ret['radar']


            elif self.lidar_layer_types[i] == 'Vote_Layer': #i=4
            
                ret = self.MMSA_modules[i](lidar_xyz_input, radar_xyz_input,lidar_feature_input,radar_feature_input)
                lidar_li_xyz, lidar_li_features, lidar_xyz_select, lidar_ctr_offsets = ret['lidar']
                radar_li_xyz, radar_li_features, radar_xyz_select, radar_ctr_offsets = ret['radar']

                lidar_centers,lidar_centers_origin = lidar_li_xyz,lidar_xyz_select
                radar_centers,radar_centers_origin = radar_li_xyz,radar_xyz_select

                lidar_center_origin_batch_idx = lidar_batch_idx.view(batch_size, -1)[:, :lidar_centers_origin.shape[1]]
                radar_center_origin_batch_idx = radar_batch_idx.view(batch_size, -1)[:, :radar_centers_origin.shape[1]]

                lidar_encoder_coords.append(torch.cat([lidar_center_origin_batch_idx[..., None].float(),lidar_centers_origin.view(batch_size, -1, 3)],dim =-1))
                radar_encoder_coords.append(torch.cat([radar_center_origin_batch_idx[..., None].float(),radar_centers_origin.view(batch_size, -1, 3)],dim =-1))
                # centers = li_xyz
                # centers_origin = xyz_select
                # center_origin_batch_idx = batch_idx.view(batch_size, -1)[:, :centers_origin.shape[1]]
                # encoder_coords.append(torch.cat([center_origin_batch_idx[..., None].float(),centers_origin.view(batch_size, -1, 3)],dim =-1))

            lidar_encoder_xyz.append(lidar_li_xyz)
            radar_encoder_xyz.append(radar_li_xyz)

            lidar_li_batch_idx = lidar_batch_idx.view(batch_size, -1)[:, :lidar_li_xyz.shape[1]]
            radar_li_batch_idx = radar_batch_idx.view(batch_size, -1)[:, :radar_li_xyz.shape[1]]
            lidar_encoder_coords.append(torch.cat([lidar_li_batch_idx[..., None].float(),lidar_li_xyz.view(batch_size, -1, 3)],dim =-1))
            radar_encoder_coords.append(torch.cat([radar_li_batch_idx[..., None].float(),radar_li_xyz.view(batch_size, -1, 3)],dim =-1))
            lidar_encoder_features.append(lidar_li_features)     
            radar_encoder_features.append(radar_li_features)     

            if lidar_li_cls_pred is not None:
                lidar_li_cls_batch_idx = lidar_batch_idx.view(batch_size, -1)[:, :lidar_li_cls_pred.shape[1]]
                radar_li_cls_batch_idx = radar_batch_idx.view(batch_size, -1)[:, :radar_li_cls_pred.shape[1]]
                lidar_sa_ins_preds.append(torch.cat([lidar_li_cls_batch_idx[..., None].float(),lidar_li_cls_pred.view(batch_size, -1, lidar_li_cls_pred.shape[-1])],dim =-1))
                radar_sa_ins_preds.append(torch.cat([radar_li_cls_batch_idx[..., None].float(),radar_li_cls_pred.view(batch_size, -1, radar_li_cls_pred.shape[-1])],dim =-1)) 
                pass
            else:
                lidar_sa_ins_preds.append([])
                radar_sa_ins_preds.append([])


        lidar_ctr_batch_idx = lidar_batch_idx.view(batch_size, -1)[:, :lidar_li_xyz.shape[1]]
        lidar_ctr_batch_idx = lidar_ctr_batch_idx.contiguous().view(-1)

        radar_ctr_batch_idx = radar_batch_idx.view(batch_size, -1)[:, :radar_li_xyz.shape[1]]
        radar_ctr_batch_idx = radar_ctr_batch_idx.contiguous().view(-1)

        batch_dict['lidar_ctr_offsets'] = torch.cat((lidar_ctr_batch_idx[:, None].float(), lidar_ctr_offsets.contiguous().view(-1, 3)), dim=1)
        batch_dict['lidar_centers'] = torch.cat((lidar_ctr_batch_idx[:, None].float(), lidar_centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['lidar_centers_origin'] = torch.cat((lidar_ctr_batch_idx[:, None].float(), lidar_centers_origin.contiguous().view(-1, 3)), dim=1)
        batch_dict['lidar_ctr_batch_idx'] = lidar_ctr_batch_idx

        batch_dict['radar_ctr_offsets'] = torch.cat((radar_ctr_batch_idx[:, None].float(), radar_ctr_offsets.contiguous().view(-1, 3)), dim=1)
        batch_dict['radar_centers'] = torch.cat((radar_ctr_batch_idx[:, None].float(), radar_centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['radar_centers_origin'] = torch.cat((radar_ctr_batch_idx[:, None].float(), radar_centers_origin.contiguous().view(-1, 3)), dim=1)
        batch_dict['radar_ctr_batch_idx'] = radar_ctr_batch_idx


        lidar_center_features = lidar_encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, lidar_encoder_features[-1].shape[1]) # shape?
        batch_dict['lidar_centers_features'] = lidar_center_features

        radar_center_features = radar_encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, radar_encoder_features[-1].shape[1]) # shape?
        batch_dict['radar_centers_features'] = radar_center_features            

        batch_dict['lidar_encoder_xyz'] = lidar_encoder_xyz
        batch_dict['lidar_encoder_coords'] = lidar_encoder_coords
        batch_dict['lidar_sa_ins_preds'] = lidar_sa_ins_preds
        batch_dict['lidar_encoder_features'] = lidar_encoder_features 

        batch_dict['radar_encoder_xyz'] = radar_encoder_xyz
        batch_dict['radar_encoder_coords'] = radar_encoder_coords
        batch_dict['radar_sa_ins_preds'] = radar_sa_ins_preds
        batch_dict['radar_encoder_features'] = radar_encoder_features 


        # if self.disable_cross_attn:

        batch_dict['ctr_offsets'] = batch_dict['lidar_ctr_offsets'] 
        batch_dict['centers'] = batch_dict['lidar_centers']
        batch_dict['centers_origin'] = batch_dict['lidar_centers_origin']
        batch_dict['ctr_batch_idx'] = batch_dict['lidar_ctr_batch_idx']
        
        batch_dict['centers_features'] = lidar_center_features

        batch_dict['encoder_xyz'] = batch_dict['lidar_encoder_xyz']
        batch_dict['encoder_coords'] = batch_dict['lidar_encoder_coords']
        batch_dict['sa_ins_preds'] = batch_dict['lidar_sa_ins_preds']
        batch_dict['encoder_features'] = batch_dict['lidar_encoder_features'] # not used later?
        
            








        return batch_dict