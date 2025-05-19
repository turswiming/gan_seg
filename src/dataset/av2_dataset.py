

import torch
from torch import nn
from torch.nn import functional as F
from .gen_point_traj_flow import process_one_sample
from .datasetutil.av2 import read_av2_scene
class AV2Dataset(nn.Module):
    def __init__(self):
        super(AV2Dataset, self).__init__()
        self.point_cloud_first = None
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        start = 0
        end  = 1
        if self.point_cloud_first is None:

            av2_scene_path = "/home/lzq/workspace/gan_seg/demo_data/demo/train/8de6abb6-6589-3da7-8e21-6ecc80004a36.h5"
            av2_test_scene_path = "/home/lzq/workspace/gan_seg/demo_data/demo/val/25e5c600-36fe-3245-9cc0-40ef91620c22.h5"
            av2_dataset = read_av2_scene(av2_scene_path)
            #get all keys
            keys = list(av2_dataset.keys())
            #get the first key
            first_key = keys[0]
            #get the first value
            first_value = av2_dataset[first_key]
            valid_mask = first_value["flow_is_valid"]
            ground_mask = first_value["ground_mask"]
            dynamic_mask = first_value["flow_category"] != 0
            valid_mask = valid_mask & dynamic_mask
            self.point_cloud_first = first_value["point_cloud_first"][valid_mask]

            #get the second value
            second_key = keys[1]
            second_value = av2_dataset[second_key]
            valid_mask_second = second_value["flow_is_valid"]
            dynamic_mask_second = second_value["flow_category"] != 0
            valid_mask_second = valid_mask_second & dynamic_mask_second
            self.point_cloud_second = second_value["point_cloud_first"][valid_mask_second]
            flow = first_value["flow"]
            self.flow = flow[valid_mask]


        sample = {}
        sample["point_cloud_first"] = self.point_cloud_first
        print("datatype",self.point_cloud_first.dtype)
        sample["point_cloud_second"] = self.point_cloud_second
        sample['flow'] = self.flow

        return sample