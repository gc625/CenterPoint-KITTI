import torch
import torch.nn as nn
from easydict import EasyDict
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
import os



class BERTSSD_Backbone(nn.Module):
    '''
    BERT-SSD backbone
    '''

    def build_both_sa_layers(
        self, 
        lidar_sa_cfg: EasyDict, 
        radar_sa_cfg: EasyDict) -> tuple(nn.ModuleList(), nn.ModuleList()):

        lidar_sa_modules = self._build_sa_layers(lidar_sa_cfg)
        radar_sa_modules = self._build_sa_layers(radar_sa_cfg)

        return lidar_sa_modules,radar_sa_modules
    
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
#
#
#                       
#
#
#
#
#
#                        
#
#
#
#
#
#



    def __init__(self,model_cfg: EasyDict, input_channels: int):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = model_cfg.num_class

        # self.SA_modules = nn.ModuleList()
        
        # Subtract 3 for xyz
        channel_in = input_channels - 3 
        channel_out_list = [channel_in]

        self.num_points_each_layer = []

        lidar_sa_cfg,radar_sa_cfg = self.model_cfg.LIDAR_SA_CONFIG,self.model_cfg.RADAR_SA_CONFIG
        
        lidar_SA_modules,radar_SA_modules = self.build_both_sa_layers(lidar_sa_cfg,radar_sa_cfg)

        
        