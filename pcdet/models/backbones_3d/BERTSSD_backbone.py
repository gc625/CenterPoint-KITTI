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
    
    def _build_sa_layers(self, sa_cfg: EasyDict) -> nn.ModuleList():

        SA_modules = nn.ModuleList()

        channel_in = sa_cfg.POINT_INPUT_FEATURES - 3 
        channel_out_list = [channel_in]
        
        
        layer_types = sa_cfg.LAYER_TYPE
        ctr_idx_list = sa_cfg.CTR_INDEX
        layer_inputs = sa_cfg.LAYER_INPUT
        aggregation_mlps = sa_cfg.get('AGGREGATION_MLPS', None)
        confidence_mlps = sa_cfg.get('CONFIDENCE_MLPS', None)
        max_translate_range = sa_cfg.get('MAX_TRANSLATE_RANGE', None)

        for k in range(sa_cfg.NSAMPLE_LIST.__len__()):
            if isinstance(layer_inputs[k], list):
                channel_in = channel_out_list[layer_inputs[k][-1]]
            else:
                channel_in = channel_out_list[layer_inputs[k]]

            if layer_types[k] == 'SA_Layer':
                mlps = sa_cfg.MLPS[k].copy()
                channel_out = 0 

                for idx in range(mlps.__len__()):
                    mlps[idx] = [channel_in] + mlps[idx]
                    channel_out += mlps[idx][-1]
                

                # * MLP to aggregate grouped points after a SA layer
                if aggregation_mlps and aggregation_mlps[k]:
                    aggregation_mlp = aggregation_mlps[k].copy()
                    if aggregation_mlp.__len__() == 0:
                        aggregation_mlp = None
                    else:
                        channel_out = aggregation_mlp[-1]
                else: 
                    aggregation_mlp = None

                # * Confidence MLP For center-aware sampling
                if confidence_mlps and confidence_mlps[k]:
                    confidence_mlp = confidence_mlps[k].copy()
                    if confidence_mlp.__len__() == 0:
                        confidence_mlp = None
                else:
                    confidence_mlp = None
                SA_modules.append(
                    pointnet2_modules.POint

                )


        return SA_modules
        

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

    def build_MMSA_layers(self,lidar_sa_cfg,radar_sa_cfg):

        
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
                        lidar_settings=lidar_settings
                    )
                )

            elif radar_layer_types[k] == 'Vote_Layer':
                print('double vote not implemented yet')
                # self.MMSA_modules.append(
                    ## DOUBLE_VOTE_LAYER
                # )

            radar_channel_out_list += [channel_out]
            lidar_channel_out_list += [channel_out]

        self.radar_num_point_features = channel_out
        self.lidar_num_point_features = channel_out

            
    def __init__(self,model_cfg: EasyDict, input_channels: int,**kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = model_cfg.num_class

        # self.SA_modules = nn.ModuleList()
        
        self.MMSA_modules = nn.ModuleList()



        # Subtract 3 for xyz
        channel_in = input_channels - 3 
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        lidar_sa_cfg,radar_sa_cfg = self.model_cfg.LIDAR_SA_CONFIG,self.model_cfg.RADAR_SA_CONFIG
        self.build_MMSA_layers(lidar_sa_cfg,radar_sa_cfg)
        # lidar_SA_modules,radar_SA_modules = self.build_both_sa_layers(lidar_sa_cfg,radar_sa_cfg)

        
        