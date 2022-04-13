from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_batch import domain_fusion as df
import os
import ipdb

class IASSD_GAN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.attach_module_topology = ['backbone_3d']
        self.attach_model_cfg = model_cfg.get('ATTACH_NETWORK')
        self.attach_model_cfg.BACKBONE_3D['num_class'] = num_class
        attach_model = self.build_attach_network()
        self.attach_model = attach_model[0]
        # self.module_list += attach_model
        self.GAN = feat_gan(None, 1, [[-1,-1]]) # use center feature only
        print('building IA-SSD-GAN')

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # for i in range(self.full_len):
        #     batch_dict = self.module_list[i](batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            gan_loss = self.get_GAN_loss(batch_dict)
            loss += gan_loss

            ret_dict = {
                'loss': loss,
                'gan_loss': gan_loss
            }
            disp_dict['gan_loss'] = gan_loss.item()
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def freeze_attach(self):
        pass

    def get_GAN_loss(self, batch_dict):
        attach_dict = {
            'points': torch.clone(batch_dict['attach']),
            'batch_size': batch_dict['batch_size']
        }
        # torch.clone(batch_dict)
        # attach_dict['points'] = attach_dict['attach']
        
        attach_dict = self.attach_model(attach_dict)
        gan_dict = {
            'att': attach_dict,
            'batch': batch_dict
        }
        gan_loss = self.GAN(gan_dict)
        # if gan_loss.item

        return gan_loss

    def build_attach_network(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': 4,
            'num_point_features': 4,
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

    def load_ckpt_to_attach(self, filename, logger, to_cpu=False):
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
            


    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        
        loss = loss_point
        return loss, tb_dict, disp_dict

class feat_gan(nn.Module):
    def __init__(self, mlps, nsample, transfer_layer, contrastive=True):
        super().__init__()
        self.nsample = nsample

        self.mlps = mlps # use to build mlp
        self.contrastive = contrastive
        self.transfer_layer = transfer_layer
        if not contrastive:
            self.mlp_module = nn.ModuleList()


        pass

    def forward(self, x):
        attach_dict = x['att']
        batch_dict = x['batch']

        att_xyzs = attach_dict['encoder_xyz']
        bat_xyzs = batch_dict['encoder_xyz']
        att_feats = attach_dict['encoder_features']
        bat_feats = batch_dict['encoder_features']
        gan_loss = []
        # get required layers
        for tl in self.transfer_layer:
            att_xyz = att_xyzs[tl[0]]
            bat_xyz = bat_xyzs[tl[1]]
            att_feat = att_feats[tl[0]].permute(0, 2, 1) # B, N, C
            bat_feat = bat_feats[tl[1]].permute(0, 2, 1)
            
            # print(att_feat.shape)
            # print(bat_feat.shape)

            # find correspondence
            bat_idx, _ = df.ball_point(1, bat_xyz, bat_xyz, 3)
            att_idx, mask = df.ball_point(1, att_xyz, bat_xyz, 3)
            print(mask.sum())
            # test = index_points(att_xyz, att_idx)
            group_att_feat = df.index_points_group(att_feat, att_idx) # [B, N, k, C]
            group_bat_feat = df.index_points_group(bat_feat, bat_idx) 
            group_att_xyz = df.index_points_group(att_xyz, att_idx) # [B, N, k, 3]
            group_bat_xyz = df.index_points_group(bat_xyz, bat_idx) # [B, N, k, 3]
            # if torch.isnan(group_att_feat).sum() + torch.isnan(group_bat_feat).sum() > 0:
            #     ipdb.set_trace()
            # if torch.isnan(group_att_xyz).sum() + torch.isnan(group_bat_xyz).sum() > 0:
            #     ipdb.set_trace()
            # group_att_xyz = df.index_points_group()
            
            group_att_points = torch.cat((group_att_xyz, group_att_feat), dim=-1) # [B, N, k, C+3]
            group_bat_points = torch.cat((group_bat_xyz, group_bat_feat), dim=-1) # [B, N, k, C+3]
            
            B, N = mask.shape
            mask = mask.reshape([B, N, 1, 1])
            _, _, _, C = group_att_points.shape
            mask = mask.repeat([1, 1, 1, C])
            group_att_points = group_att_points * mask
            group_bat_points = group_bat_points * mask
            # ipdb.set_trace()
            # print(nn.functional.mse_loss(group_att_feat, group_bat_feat, reduction='mean'))
            # break
            if self.contrastive:
                gan_loss += [nn.functional.mse_loss(group_att_points, group_bat_points, reduction='mean')]
            else:
                raise NotImplementedError
        
        loss = sum(gan_loss) / len(gan_loss)
        if torch.isnan(loss):
            loss = gan_loss[-1]

        # loss = loss / len(gan_loss)
        # print(loss)
        return loss
                    
                




        

    