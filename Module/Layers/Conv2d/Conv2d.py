##
## EPITECH PROJECT, 2025
## library
## File description:
## Conv2d
##
from __future__ import annotations
from library.Module.Module import Module
from library.Parameter.Parameter import Parameter
from library.Tensor.Tensor import Tensor
from library.Helper.Initializers import xavier_uniform, he_uniform, zeros_initializer


class Conv2d(Module):
    ##
    # @brief Initialize a 2D convolutional layer.
    # @details Creates learnable kernels and biases with configurable stride,
    # padding, and initializer, mirroring the NCHW convention used across the
    # framework.
    ##
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0, init_method: str = "xavier",) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _ensure_pair(kernel_size, "kernel_size")
        self.stride = _ensure_pair(stride, "stride")
        self.padding = _ensure_pair(padding, "padding")
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = out_channels * self.kernel_size[0] * self.kernel_size[1]
        weight_shape = (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        if init_method == "xavier":
            weight_data = xavier_uniform(weight_shape, fan_in=fan_in, fan_out=fan_out)
        elif init_method == "he":
            weight_data = he_uniform(weight_shape, fan_in=fan_in)
        elif init_method == "zeros":
            weight_data = zeros_initializer(weight_shape)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        self.weight = Parameter(weight_data)
        self.bias = Parameter(zeros_initializer((out_channels,)))

    ##
    # @brief Apply the convolution to an input tensor.
    # @details Delegates the heavy lifting to the Tensor conv2d operator and
    # adds the broadcasted bias term to match PyTorch-style semantics.
    ##
    def forward(self, x: Tensor) -> Tensor:
        out = x.conv2d(self.weight, stride=self.stride, padding=self.padding)
        return out + self.bias.reshape((1, self.out_channels, 1, 1))


def _ensure_pair(value: int | tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    ##
    # @brief Normalize Conv2d hyperparameters to a tuple.
    # @details Converts scalar, tuple, or list inputs into a consistent
    # (height, width) pair for kernel, stride, and padding attributes.
    ##
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two integers.")
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return value, value
    raise TypeError(f"{name} must be either an int or a pair of ints.")
