import torch
import torch.nn as nn
import os # for debug

class ClusterBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = model_cfg.num_class
        self.channel_in = input_channels
        self.cluster_num = model_cfg.CLUSTER_NUM

    def forward(self, batch_dict):

        # pointnet based classification for cluster
        
        # point clustering based on classification

        # cluster merging

        # fix num_points in cluster

        # fix num_cls


        pass

        
