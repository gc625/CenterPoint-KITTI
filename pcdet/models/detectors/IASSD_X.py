from .detectorX_template import DetectorX_template

class IASSD_X(DetectorX_template):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        print('building IA-SSD cross modal')

    def forward(self, batch_dict):
        # if self.train_main:
        #     for cur_module in self.module_list:
        #         batch_dict = cur_module(batch_dict)
        # else:
        #     for cur_module in self.module_list:
        #         batch_dict = cur_module(batch_dict)
        #     batch_dict = self.module_list[self.head_idx]
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

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