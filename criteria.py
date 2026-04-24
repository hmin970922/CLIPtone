import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

import math
import open_clip
from PIL import Image

import time


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)


class CLIPLoss(torch.nn.Module):
    def __init__(self, device, direction_loss_type='cosine', clip_model='MobileCLIP2-S0', clip_pretrained='dfndr2b'):
        super(CLIPLoss, self).__init__()

        self.device = device
        self.model, _, clip_preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=clip_pretrained
        )
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        self.clip_preprocess = clip_preprocess

        # 텐서 입력용 전처리: Resize/CenterCrop/Normalize만 유지, PIL 변환 계열 스킵
        tensor_transforms = [
            t for t in clip_preprocess.transforms
            if isinstance(t, (transforms.Resize, transforms.CenterCrop, transforms.Normalize))
        ]
        self.preprocess = transforms.Compose(tensor_transforms)

        self.cos = torch.nn.CosineSimilarity()
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def text_dim(self) -> int:
        return self.model.text.output_dim

    def tokenize(self, strings: list):
        return self.tokenizer(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, class_str: str, norm: bool = True) -> torch.Tensor:
        tokens = self.tokenizer([class_str]).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def clip_directional_loss(self, src_img: torch.Tensor, target_img: torch.Tensor, target_direction: torch.Tensor, src_encoding: torch.Tensor = None) -> torch.Tensor:
        if src_encoding is None:
            src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum() == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True).clamp(min=1e-4))

        return self.direction_loss(edit_direction, target_direction).mean()

    def forward(self, src_img: torch.Tensor, target_img: torch.Tensor, target_direction: torch.Tensor, src_encoding: torch.Tensor = None):
        clip_loss = self.clip_directional_loss(src_img, target_img, target_direction, src_encoding=src_encoding)
        return clip_loss
