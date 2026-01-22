##
## EPITECH PROJECT, 2025
## library
## File description:
## MaxPool2d
##
from __future__ import annotations
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor


class MaxPool2d(Module):
    ##
    # @brief Initialize a 2D max pooling layer.
    # @details Stores kernel, stride, and padding information with torch-like
    # semantics where stride defaults to kernel size.
    ##
    def __init__(self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None, padding: int | tuple[int, int] = 0,) -> None:
        super().__init__()
        self.kernel_size = _ensure_pair(kernel_size, "kernel_size")
        stride_value = stride if stride is not None else self.kernel_size
        self.stride = _ensure_pair(stride_value, "stride")
        self.padding = _ensure_pair(padding, "padding")

    ##
    # @brief Apply the pooling operation to the incoming tensor.
    # @details Delegates to the Tensor.max_pool2d operator which handles
    # autograd bookkeeping and window-wise argmax caching.
    ##
    def forward(self, x: Tensor) -> Tensor:
        return x.max_pool2d(
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

 ##
# @brief Normalize pooling hyperparameters to a tuple.
# @details Accepts scalar, tuple, or list inputs and emits a (height, width)
# pair usable by the underlying Tensor operator.
##
def _ensure_pair(value: int | tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two integers.")
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return value, value
    raise TypeError(f"{name} must be either an int or a pair of ints.")
