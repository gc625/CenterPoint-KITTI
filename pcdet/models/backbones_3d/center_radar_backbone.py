from turtle import forward
import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules

class CenterRadarBackbone(nn.Module):
    def __init__(self, model_cfg, num_class, input_channels, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_class = num_class

        # SA(256, (0.5, 1), (8, 16), ((16, 16, 32), (32, 32, 64))) -> MLP (96, 64) 'ins_aware', split backgroud foreground
        self.SA_modules_shared = nn.ModuleList() 
        

        # SA(128, (1.5, 3), (16, 32), ((64, 64, 128), (128, 128, 256))) -> MLP(384, 256) 'ins_aware'
        # Vote_layer
        # SA(128, )
        self.SA_modules_fg = nn.ModuleList()

        self.SA_modules_bg = nn.ModuleList()

        channel_in = input_channels - 3

        sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = sa_config.LAYER_TYPE
        self.ctr_idx_list = sa_config.CTR_INDEX
        self.layer_inputs = sa_config.LAYER_INPUT
        

        pass

    def forward(self, batch_dict):
        pass