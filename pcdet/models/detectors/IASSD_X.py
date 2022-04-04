from .detector3d_template import Detector3DTemplate

class IASSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # define build_attach_networks() function

        self.attach_module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d'
        ] # backbone only
        self.train_main = True
        # used for training attach network
        self.head_idx = None
        if self.attach_model_cfg is not None:
            self.attach_module_list = self.build_attach_network()

            if self.entire_cfg.get('TRAIN_ATTACH', False):
                # freeze main network params
                self.train_main = False
                
                for single_module, idx in enumerate(self.module_list):
                    for param in single_module.parameters:
                        param.requires_grad = False
                    if 'heads' in str(type(single_module)):
                        self.head_idx = idx

            else:
                # freeze attach network backbone params
                for single_module in self.attach_module_list:
                    for param in single_module.parameters:
                        param.requires_grad = False

    def forward(self, batch_dict):
        if self.train_main:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
        else:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
            batch_dict = self.module_list[self.head_idx]

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        
        loss = loss_point
        return loss, tb_dict, disp_dict