##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Conv2d backward operator
##

# **************************************************************************** #
#                                                                              #
#                           CONV2D BINARY BACKWARD                             #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import numpy as np
from library.Tensor.BackwardOP.BinaryBackwardOp.BinaryBackwardOp import BinaryBackwardOp
if TYPE_CHECKING:
    from library.Tensor.Tensor import Tensor


class Conv2dBackward(BinaryBackwardOp):
    ##
    # @brief Backward operator for 2D convolutions.
    # @details Stores convolution hyperparameters and reconstructs the gradients
    # for both the input activation map and the convolution kernels during
    # backpropagation.
    ##
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        weight_shape: Tuple[int, ...],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
    ) -> None:
        self.input_shape = input_shape
        self.weight_shape = weight_shape
        self.stride = stride
        self.padding = padding

    ##
    # @brief Compute gradient with respect to the input activation.
    # @details Slides the flipped kernels over the upstream gradient map while
    # accounting for stride and padding to reconstruct the influence on each
    # input pixel.
    ##
    def _compute_grad_a(self, out_grad: np.ndarray, a: "Tensor", b: "Tensor") -> np.ndarray:
        return _conv2d_grad_input(
            out_grad,
            b.data,
            self.input_shape,
            self.stride,
            self.padding,
        )

    ##
    # @brief Compute gradient with respect to the convolution kernels.
    # @details Reuses the original padded input windows multiplied by the
    # upstream gradient at every spatial position to accumulate kernel updates.
    ##
    def _compute_grad_b(self, out_grad: np.ndarray, a: "Tensor", b: "Tensor") -> np.ndarray:
        return _conv2d_grad_weight(
            out_grad,
            a.data,
            self.weight_shape,
            self.stride,
            self.padding,
        )


##
# @brief Gradient helper for the input activation map.
# @details Applies the rotated kernels over the upstream gradients to
# reconstruct per-pixel influence, trimming artificial padding afterwards.
##
def _conv2d_grad_input(out_grad: np.ndarray,weight: np.ndarray,input_shape: Tuple[int, ...],stride: Tuple[int, int],padding: Tuple[int, int],) -> np.ndarray:
    batch, in_channels, in_h, in_w = input_shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    _, _, kernel_h, kernel_w = weight.shape
    out_h, out_w = out_grad.shape[2], out_grad.shape[3]
    flipped = np.flip(weight, axis=(2, 3))
    padded_grad = np.zeros(
        (batch, in_channels, in_h + 2 * pad_h, in_w + 2 * pad_w),
        dtype=out_grad.dtype,
    )
    for i in range(out_h):
        h_start = i * stride_h
        h_end = h_start + kernel_h
        for j in range(out_w):
            w_start = j * stride_w
            w_end = w_start + kernel_w
            grad_slice = np.tensordot(
                out_grad[:, :, i, j],
                flipped,
                axes=([1], [0]),
            )
            padded_grad[:, :, h_start:h_end, w_start:w_end] += grad_slice
    h_slice = slice(pad_h, -pad_h) if pad_h > 0 else slice(None)
    w_slice = slice(pad_w, -pad_w) if pad_w > 0 else slice(None)
    return padded_grad[:, :, h_slice, w_slice]

##
# @brief Gradient helper for convolution kernels.
# @details Accumulates contributions from every spatial location by
# multiplying the upstream gradient with the corresponding padded input
# patches.
##
def _conv2d_grad_weight(out_grad: np.ndarray, input_data: np.ndarray, weight_shape: Tuple[int, ...], stride: Tuple[int, int], padding: Tuple[int, int],) -> np.ndarray:
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    kernel_h, kernel_w = weight_shape[2], weight_shape[3]
    padded_input = np.pad(
        input_data,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    grad_weight = np.zeros(weight_shape, dtype=input_data.dtype)
    out_h, out_w = out_grad.shape[2], out_grad.shape[3]
    for i in range(out_h):
        h_start = i * stride_h
        h_end = h_start + kernel_h
        for j in range(out_w):
            w_start = j * stride_w
            w_end = w_start + kernel_w
            window = padded_input[:, :, h_start:h_end, w_start:w_end]
            grad_weight += np.tensordot(
                out_grad[:, :, i, j],
                window,
                axes=([0], [0]),
            )
    return grad_weight
