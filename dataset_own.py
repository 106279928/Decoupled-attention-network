import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import os
from torchvision import transforms

class OwnDataset(Dataset):
    def __init__(
        self,
        base_dir,
        gt_file_dir,
        img_size
    ):
        self.base_dir = base_dir
        self.gt_file_dir = gt_file_dir
        self.img_size = img_size
        # It will be map of {img_name: [list of gt]}
        self.img_gt_map = self.extract_gt(self.gt_file_dir)

    def extract_gt(self, gt_file_dir):
        gt_map = {}
        with open(gt_file_dir) as gt_file:
            for line in gt_file:
                line = line.split('\t')
                assert len(line) == 2
                img_name = line[0]
                gt = line[1]
                if len(gt) > 24:
                    continue
                if img_name in gt_map and gt_map[img_name] == gt:
                    pass
                elif img_name in gt_map:
                    assert img_name not in gt_map, f"{img_name} already exists, new_val: {gt}, old_val: {gt_map[img_name]}"
                gt_map[img_name] = gt
        return gt_map

    def __len__(self):
        return len(self.img_gt_map)

    def __getitem__(self, idx):
        img_name = list(self.img_gt_map)[idx]
        gt = self.img_gt_map[img_name]
        img = Image.open(self.base_dir + img_name)
        img = img.resize((self.img_size, self.img_size))
        img = transforms.Grayscale()(img)
        img = transforms.ToTensor()(img)
        return {
            'image': img,
            'label': gt
        }