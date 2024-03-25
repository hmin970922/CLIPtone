import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class AdaptationModule(nn.Module):
    r"""The Adaptive Interval Learning (AdaInt) module (mapping g).

    It consists of a single fully-connected layer and some post-process operations.

    Args:
        n_colors (int): Number of input color channels.
        n_vertices (int): Number of sampling points along each lattice dimension.
        n_feats (int): Dimension of the input image represedntation vector.
        adaint_share (bool, optional): Whether to enable Share-AdaInt. Default: False.
    """

    def __init__(self, args, feature_dim, n_feats) -> None:
        super().__init__()
        
        self.lut_weight_dim = (args.n_ranks, n_feats)
        self.adaint_weight_dim = ((args.n_vertices - 1) * args.n_colors, n_feats)
        self.block1 = nn.Linear(feature_dim, feature_dim)
        # self.block2 = nn.Linear(feature_dim, feature_dim)
        # self.block3 = nn.Linear(feature_dim, feature_dim)
        # self.block4 = nn.Linear(feature_dim, feature_dim)
        # self.block5 = nn.Linear(feature_dim, feature_dim)
        self.activation = nn.LeakyReLU()
        self.lut_delta_generator = nn.Linear(feature_dim, self.lut_weight_dim[0] * self.lut_weight_dim[1])
        self.adaint_delta_generator = nn.Linear(feature_dim, self.adaint_weight_dim[0] * self.adaint_weight_dim[1])
        self.init_weights()
        
    def init_weights(self):
        r"""Init weights for models.

        We use all-zero and all-one initializations for its weights and bias, respectively.
        """
        nn.init.uniform_(self.block1.weight, 0.0, 0.01)
        nn.init.zeros_(self.block1.bias)
        # nn.init.uniform_(self.block2.weight, 0.0, 0.1)
        # nn.init.zeros_(self.block2.bias)
        # nn.init.uniform_(self.block3.weight, 0.0, 0.1)
        # nn.init.zeros_(self.block3.bias)
        # nn.init.uniform_(self.block4.weight, 0.0, 0.1)
        # nn.init.zeros_(self.block4.bias)
        # nn.init.uniform_(self.block5.weight, 0.0, 0.1)
        # nn.init.zeros_(self.block5.bias)
        nn.init.uniform_(self.lut_delta_generator.weight, 0.0, 0.01)
        nn.init.uniform_(self.adaint_delta_generator.weight, 0.0, 0.01)
        nn.init.zeros_(self.lut_delta_generator.bias)
        nn.init.zeros_(self.adaint_delta_generator.bias)
        
    def forward(self, x, intensity=1):
        r"""Forward function for AdaInt module.

        Args:
            x (tensor): Input image representation, shape (b, f).
        Returns:
            Tensor: Sampling coordinates along each lattice dimension, shape (b, c, d).
        """
        # import pdb; pdb.set_trace()
        x = x.to(torch.float32)
        x = self.block1(x)
        x = self.activation(x)
        # x = self.block2(x)
        # x = self.activation(x)
        # x = self.block3(x)
        # x = self.activation(x)
        # x = self.block4(x)
        # x = self.activation(x)
        # x = self.block5(x)
        # x = self.activation(x)
        lut_weights_delta = self.lut_delta_generator(x)
        adaint_weights_delta = self.adaint_delta_generator(x)
        lut_weights_delta = lut_weights_delta.view(self.lut_weight_dim[0], self.lut_weight_dim[1])
        adaint_weights_delta = adaint_weights_delta.view(self.adaint_weight_dim[0], self.adaint_weight_dim[1])
        return (intensity * lut_weights_delta, intensity * adaint_weights_delta)