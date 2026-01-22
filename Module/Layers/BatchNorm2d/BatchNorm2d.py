##
## EPITECH PROJECT, 2025
## library
## File description:
## BatchNorm2d
##
from __future__ import annotations
import numpy as np
from library.Module.Module import Module
from library.Parameter.Parameter import Parameter
from library.Tensor.Tensor import Tensor


class BatchNorm2d(Module):
    ##
    # @brief Initialize a 2D batch normalization layer.
    # @details Allocates learnable scale and bias tensors alongside running
    # statistics for inference-time normalization.
    ##
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.weight = Parameter(np.ones(num_features, dtype=np.float32))
        self.bias = Parameter(np.zeros(num_features, dtype=np.float32))
        self.running_mean: np.ndarray = np.zeros(num_features, dtype=np.float32)
        self.running_var: np.ndarray = np.ones(num_features, dtype=np.float32)
        self.eps = eps
        self.momentum = momentum

    ##
    # @brief Normalize a 4D tensor across batch and spatial dimensions.
    # @details Uses batch statistics during training and the cached running
    # statistics during evaluation, mirroring PyTorch semantics.
    ##
    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 4:
            raise ValueError("BatchNorm2d expects 4D inputs in NCHW format.")
        axes = (0, 2, 3)
        if self.training:
            mean = x.mean(axis=axes, keepdims=True)
            centered = x - mean
            variance = (centered * centered).mean(axis=axes, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data.reshape(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance.data.reshape(-1)
        else:
            mean = Tensor(self.running_mean.reshape(1, -1, 1, 1))
            centered = x - mean
            variance = Tensor(self.running_var.reshape(1, -1, 1, 1))
        std = (variance + self.eps).pow(0.5)
        normalized = centered / std
        weight = self.weight.reshape((1, -1, 1, 1))
        bias = self.bias.reshape((1, -1, 1, 1))
        return normalized * weight + bias
