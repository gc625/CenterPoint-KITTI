from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_batch import domain_fusion as df
import os
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
import ipdb
from ...vis_tools.vis_tools import *
import numpy as np



class BERTSSD(Detector3DTemplate):
    #TODO self.transfer seems useless? 
    def __init__(self, model_cfg, num_class, dataset, tb_log=None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        
        print('Start building IA-SSD-GAN') 
        # Tensorboard + Debug 
        self.tb_log = tb_log
        self.debug = self.model_cfg.get('DEBUG', False)

        # Network Modules: 
        # Used modules: bb_3d, feature_aug, point_head
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'feature_aug', 'dense_head',  'point_head', 'roi_head'
        ]

        self.module_list = self.build_networks()
        
        # Attach Network
        self.attach_module_topology = ['backbone_3d']
        self.attach_model_cfg = model_cfg.get('ATTACH_NETWORK')
        self.attach_model_cfg.BACKBONE_3D['num_class'] = num_class        
        self.attach_model = None if model_cfg.get('DISABLE_ATTACH') else self.build_attach_network()[0] # idx becos it reutrns bb in list 
        

        self.radar_



        # Shared Head
        self.shared_module_topology = ['point_head']
        shared_head = self.build_shared_head()
        self.shared_head = None if len(shared_head) == 0 else shared_head[0] 
        
        # ? Not sure if used
        self.cross_over_cfg = self.model_cfg.CROSS_OVER 
        
        # Feature augmentation for feature detection
        self.use_feature_aug = model_cfg.get('USE_FEAT_AUG', False)
        
        print('Done building IA-SSD-GAN')   
        
        # Visualization settings
        self.vis_cnt = 0 
        self.vis_interval = 100 # in unit batch
        self.class_names = model_cfg.get('CLASS_NAMES', None)
        

    def load_radar_network(self):
        num_feats = self.attach_model_cfg.get('NUM_POINT_FEATURES',4) 
        # print(f"ATTACH NUM FEATS {num_feats}")
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': num_feats,
            'num_point_features': num_feats,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': True
        }
        for module_name in self.attach_module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            full_module_name = 'attach_' + module_name
            self.add_module(full_module_name, module)
        return model_info_dict['module_list']




# ===========================================================================
    def load_ckpt_to_attach(self, filename, logger, to_cpu=False):
        """
        not used..?
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        
        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            attach_key = 'attach_' + key
            if attach_key in self.state_dict() and self.state_dict()[attach_key].shape == model_state_disk[key].shape:
                update_model_state[attach_key] = val
                logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        # for key in state_dict:
        #     if key not in update_model_state:
        #         logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def freeze_attach(self, logger):
        for name, param in self.named_parameters():
            if 'attach' in name:
                param.requires_grad = False
                logger.info('Freeze param in ' + name)

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': False
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            # print("XX"*150)
            # print(module)
            self.add_module(module_name, module)
        return model_info_dict['module_list']


    def build_attach_network(self):
        num_feats = self.attach_model_cfg.get('NUM_POINT_FEATURES',4) 
        # print(f"ATTACH NUM FEATS {num_feats}")
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': num_feats,
            'num_point_features': num_feats,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': True
        }
        for module_name in self.attach_module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            full_module_name = 'attach_' + module_name
            self.add_module(full_module_name, module)
        return model_info_dict['module_list']
    
    def build_shared_head(self):
        num_feats = self.attach_model_cfg.get('NUM_POINT_FEATURES',4) 
        # print(f"SHARED HEAD NUM FEATS {num_feats}")
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': num_feats,
            'num_point_features': self.model_cfg.SHARED_HEAD.NUM_POINT_FEATURES,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': False
        }

        for module_name in self.shared_module_topology:
            self.model_cfg.SHARED_HEAD['DEBUG'] = self.debug
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict,
                custom_cfg=self.model_cfg.SHARED_HEAD
            )
            full_module_name = 'shared_' + module_name
            self.add_module(full_module_name, module)
        return model_info_dict['module_list']

# ===========================================================================

    def print_shapes(self, batch_dict):
        keys = batch_dict.keys()
        print('='*80)
        for k in batch_dict.keys():
            if isinstance(batch_dict[k],int):
                print(f'{k} (int): {batch_dict[k]}')
            elif isinstance(batch_dict[k],dict):
                dict2 = batch_dict[k]
                print('-'*30+f'inner dict: {k}'+'-'*30)
                for K in dict2.keys():
                    if isinstance(dict2[K],int):
                        print(f'{K}: {dict2[K]}')
                    elif isinstance(dict2[K],list):
                        print(f'{K} (len): {len(dict2[K])} , {[len(tensor) for tensor in dict2[K]]}')
                    elif dict2[K] is None:
                        print(f'{K}: IS NONE')
                    else:
                        print(f'{K}: {dict2[K].shape}')
                print('-'*60)
            elif isinstance(batch_dict[k],list):
                print(f'{k} (len): {len(batch_dict[k])}, {[len(tensor) for tensor in batch_dict[k]]}')
            elif batch_dict[k] is None:
                print(f'{k}: is NONE')    
            else:
                print(f'{k}: {batch_dict[k].shape}')
            
    def get_transfer_feature(self, batch_dict):
        attach_dict = {
            'points': torch.clone(batch_dict['attach']),
            'batch_size': batch_dict['batch_size'],
            'frame_id': batch_dict['frame_id']
        }

        attach_dict = self.attach_model(attach_dict)

        return attach_dict

    def forward(self, batch_dict):
        """
        batch_dict: dict = ['points', 'frame_id', 'attach', 'gt_boxes', 'use_lead_xyz', 'image_shape', 'batch_size']
        """
        if self.use_feature_aug & self.training:
            if self.attach_model is not None:
                transfer_dict = self.get_transfer_feature(batch_dict)
                # self.print_shapes(batch_dict)
                # print('TRANSFER DICT')
                # self.print_shapes(transfer_dict)
                batch_dict['att'] = transfer_dict
        for cur_module in self.module_list:
            
            batch_dict = cur_module(batch_dict)
            # self.print_shapes(batch_dict)
            # print('='*150)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            # get feat transfer loss
            transfer_loss, shared_tb_dict, transfer_disp_dict = self.get_transfer_loss(batch_dict)
            disp_dict['det_loss'] = loss.item()
            disp_dict['matching_loss'] = tb_dict['matching_loss']
            loss = (transfer_loss + loss) / 2
            tb_keys = ['center_loss_cls', 'center_loss_box', 'corner_loss_reg']

            ret_dict = {
                'loss': loss,
                'gan_loss': transfer_loss
            }
            disp_dict['gan_loss'] = transfer_loss.item()
            disp_dict['tatal_loss'] = loss.item()



            shared_det_list = []
            det_list = []
            for k in tb_keys:
                shared_det_list += [shared_tb_dict[k]]
                det_list += [tb_dict[k]]
            disp_dict['shared_box_loss'] = sum(shared_det_list)
            disp_dict['box_loss'] = sum(det_list)
            tb_dict['shared_box_loss'] = sum(shared_det_list)
            tb_dict['box_loss'] = sum(det_list)

            return ret_dict, tb_dict, disp_dict
        else:
            
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.debug:
                loss, tb_dict, disp_dict = self.get_training_loss()
                transfer_loss = self.get_transfer_loss(batch_dict)
                batch_dict['sa_ins_labels'] = tb_dict['sa_ins_labels']
                # selected lidar points
                # selected lidar points labels
                # radar points labels
                # radar points classification
                pass
            recall_dicts['batch_dict'] = batch_dict
            return pred_dicts, recall_dicts

    def get_transfer_loss(self, batch_dict):

        attach_dict = self.get_transfer_feature(batch_dict)
        transfer_dict = {
            'att': attach_dict,
            'batch': batch_dict
        }
        radar_shared_feat = batch_dict['radar_shared']
        share_head_dict = {}
        # print(f'RAD SHARE FEAT{radar_shared_feat.shape}')
        for key in attach_dict.keys():
            if key in batch_dict:
                share_head_dict[key] = batch_dict[key]
        share_head_dict.pop('centers_features')
        share_head_dict['gt_boxes'] = batch_dict['gt_boxes']
        _, c, _ = radar_shared_feat.shape
        # print(f'RAD SHARE FEAT{radar_shared_feat.shape}')
        share_head_dict['centers_features'] = radar_shared_feat.permute(0,2,1).contiguous().view(-1, c)
        share_head_dict = self.shared_head(share_head_dict)
        share_head_loss, shared_tb_dict = self.shared_head.get_loss(share_head_dict)
        disp_dict = {
            'share_det_loss': share_head_loss.item()
        }
        return share_head_loss, shared_tb_dict, disp_dict
        
    
    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_match, new_tb_dict = self.feature_aug.get_loss()

        # get loss from shared_det_head
        tb_dict.update(new_tb_dict)
        loss = (2/3)*loss_point + (1/3)*loss_match

        return loss, tb_dict, disp_dict
