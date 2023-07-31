# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import numpy as np
import pickle
from typing import List, Union

from mmengine.fileio import get_local_path

from mmdet.registry import DATASETS
# from .registry import DATASETS
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class DogSkinDataset(BaseDetDataset):

    METAINFO = {
        'classes':
        ("A1_구진_플라크",
        "A2_비듬_각질_상피성잔고리",
        "A3_태선화_과다색소침착",
        "A4_농포_여드름",
        "A5_미란_궤양",
        "A6_결절_종괴"),
        
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100),]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.dogSkin = self.COCOAPI(local_path)

        self.cat_ids = self.dogSkin.get_cat_ids(
            cat_names=self.metainfo['classes'])

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.dogSkin.cat_img_map)

        img_ids = self.dogSkin.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.dogSkin.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.dogSkin.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.dogSkin.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.dogSkin

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])

        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None

        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids: # [0, 1, 2, 3, 4, 5]: 
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])

        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue

            if width * height >= min_size:
                valid_data_infos.append(data_info)
                
        return valid_data_infos
