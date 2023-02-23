import cv2
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.config.yamlize import yamlize
from src.constants import DEVICE
from src.encoders.base import BaseEncoder
from src.encoders.transforms.preprocessing import crop_resize_center


@yamlize
class RandomEncoder(BaseEncoder, torch.nn.Module):
    """Input should be (bsz, C, H, W) where C=3, H=42, W=144"""

    def __init__(
            self,
            image_channels: int = 3,
            image_height: int = 42,
            image_width: int = 144,
            z_dim: int = 32,
            load_checkpoint_from: str = "",
    ):
        super().__init__()

        self.im_c = image_channels
        self.im_h = image_height
        self.im_w = image_width
        self.z_dim = z_dim

    def encode(self, x: np.ndarray, device=DEVICE) -> torch.Tensor:
        # assume x is RGB image with shape (H, W, 3)
        v = torch.randn(1, self.z_dim)
        return v

    def decode(self, z):
        z = self.fc3(z)
        return self.decoder(z)
