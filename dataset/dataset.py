import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random
from torch.nn.modules.utils import _pair
import pandas as pd


class SingleImageDataset(Dataset):
    def __init__(self, image_path, ann_file, aug=True, crop_ratio=(0.6, 1.0), p=0.5, brightness=0.2, contrast=0.0, saturation=0.2, hue=0.0):
        self.dir_lq = os.path.join(image_path, 'input/JPG/480p')
        self.ann_file = ann_file
        self.data_infos = self.load_annotations(self.ann_file)
        
        self.aug = aug
        self.crop_ratio = crop_ratio
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        lq = Image.open(self.data_infos[idx]['lq_path']).convert('RGB')
        if self.aug:
            lq = self.custom_transformation(lq)
        
        transform = transforms.ToTensor()
        lq = transform(lq)
        file_name = self.data_infos[idx]['lq_path'].split('/')[-1]
        return lq, file_name
    
    def load_annotations(self, ann_file):
        """Load annoations for enhancement dataset.

        It loads the LQ and GT image path from the annotation file.
        Each line in the annotation file contains the image name.
        Returns:
            dict: Returned dict for LQ and GT pairs.
        """
        data_infos = []
        with open(ann_file, 'r') as fin:
            for line in fin:
                basename = line.split('\n')[0]
                lq_name = basename + '.jpg'
                data_infos.append(
                    dict(
                        lq_path=os.path.join(self.dir_lq, lq_name)))
        return data_infos
    
    def custom_transformation(self, img):
        
        ratio_h = random.uniform(*self.crop_ratio)
        ratio_w = random.uniform(*self.crop_ratio)
        crop_size = (int(img.size[1] * ratio_h), int(img.size[0] * ratio_w))
        
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=crop_size)
        img = transforms.functional.crop(img, i, j, h, w)
        
        if self.p > 0 and random.random() < self.p:
            img = transforms.functional.hflip(img)
        
        b_min = max(0, 1 - self.brightness)
        b_max = 1 + self.brightness
        c_min = max(0, 1 - self.contrast)
        c_max = 1 + self.contrast
        s_min = max(0, 1 - self.saturation)
        s_max = 1 + self.saturation
        h_min = -self.hue
        h_max = self.hue
        
        _, brightness_factor, contrast_factor, saturation_factor, hue_factor, = transforms.ColorJitter.get_params([b_min, b_max], [c_min,c_max], [s_min,s_max], [h_min,h_max])
        img = transforms.functional.adjust_brightness(img, brightness_factor)
        img = transforms.functional.adjust_contrast(img, contrast_factor)
        img = transforms.functional.adjust_saturation(img, saturation_factor)
        img = transforms.functional.adjust_hue(img, hue_factor)
        
        
        return img


class DirectionDataset(Dataset):
    def __init__(self, tensor_path):
        self.data_infos = torch.load(tensor_path)
        
    def __len__(self):
        return len(self.data_infos)
    
    def __getitem__(self, idx):
        return self.data_infos[idx]
