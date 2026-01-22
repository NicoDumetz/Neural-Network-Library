##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## MaxPool2d backward operator
##

# **************************************************************************** #
#                                                                              #
#                           MAXPOOL UNARY BACKWARD                             #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.UnaryBackwardOp.UnaryBackwardOp import UnaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor

##
# @brief Backward operator for 2D max pooling.
# @details Routes upstream gradients to the stored maximal positions for each
# pooling window, reversing the spatial down-sampling performed during the
# forward pass.
##
class MaxPool2dBackward(UnaryBackwardOp):
    def __init__(self, input_shape: Tuple[int, ...], kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int], indices: np.ndarray,) -> None:
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.indices = indices

    def _compute_grad(self, out_grad: np.ndarray, a: "Tensor") -> np.ndarray:
        return _max_pool2d_grad_input(out_grad, self.indices, self.input_shape, self.kernel_size, self.stride, self.padding,)

##
# @brief Distribute gradients back to the maximal elements of each window.
# @details Uses the cached argmax indices to accumulate the upstream
# gradient contributions at the original spatial resolution.
##
def _max_pool2d_grad_input(out_grad: np.ndarray, indices: np.ndarray, input_shape: Tuple[int, ...], kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int],) -> np.ndarray:
    batch, channels, height, width = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_h, out_w = out_grad.shape[2], out_grad.shape[3]
    padded_grad = np.zeros(
        (batch, channels, height + 2 * pad_h, width + 2 * pad_w),
        dtype=out_grad.dtype,
    )
    batch_ids = np.repeat(np.arange(batch), channels)
    channel_ids = np.tile(np.arange(channels), batch)
    for i in range(out_h):
        h_start = i * stride_h
        h_end = h_start + kernel_h
        for j in range(out_w):
            w_start = j * stride_w
            w_end = w_start + kernel_w
            idx_flat = indices[:, :, i, j].reshape(-1)
            grad_flat = out_grad[:, :, i, j].reshape(-1)
            h_offsets = idx_flat // kernel_w
            w_offsets = idx_flat % kernel_w
            padded_grad[
                batch_ids,
                channel_ids,
                h_start + h_offsets,
                w_start + w_offsets,
            ] += grad_flat
    h_slice = slice(pad_h, -pad_h) if pad_h > 0 else slice(None)
    w_slice = slice(pad_w, -pad_w) if pad_w > 0 else slice(None)
    return padded_grad[:, :, h_slice, w_slice]
