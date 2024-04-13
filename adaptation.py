import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class AdaptationModule(nn.Module):
    r"""The Text Adaptor.
    It consists of two FC layers and single activation function.

    Args:
        feature_dim (int): Dimension of the input direction features. 
        n_feats (int): Dimension of the input image represedntation vector.
    """

    def __init__(self, args, feature_dim, n_feats) -> None:
        super().__init__()
        
        self.lut_weight_dim = (args.n_ranks, n_feats)
        self.adaint_weight_dim = ((args.n_vertices - 1) * args.n_colors, n_feats)
        self.block1 = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.LeakyReLU()
        self.lut_delta_generator = nn.Linear(feature_dim, self.lut_weight_dim[0] * self.lut_weight_dim[1])
        self.adaint_delta_generator = nn.Linear(feature_dim, self.adaint_weight_dim[0] * self.adaint_weight_dim[1])
        self.init_weights()
        
    def init_weights(self):
        r"""Init weights for models.

        We use uniform initializations for weight, and all-zero initializations for bias.
        """
        nn.init.uniform_(self.block1.weight, 0.0, 0.01)
        nn.init.zeros_(self.block1.bias)
        nn.init.uniform_(self.lut_delta_generator.weight, 0.0, 0.01)
        nn.init.uniform_(self.adaint_delta_generator.weight, 0.0, 0.01)
        nn.init.zeros_(self.lut_delta_generator.bias)
        nn.init.zeros_(self.adaint_delta_generator.bias)
        
    def forward(self, x, intensity=1):
        x = x.to(torch.float32)
        x = self.block1(x)
        x = self.activation(x)
        lut_weights_delta = self.lut_delta_generator(x)
        adaint_weights_delta = self.adaint_delta_generator(x)
        lut_weights_delta = lut_weights_delta.view(self.lut_weight_dim[0], self.lut_weight_dim[1])
        adaint_weights_delta = adaint_weights_delta.view(self.adaint_weight_dim[0], self.adaint_weight_dim[1])
        return (intensity * lut_weights_delta, intensity * adaint_weights_delta)