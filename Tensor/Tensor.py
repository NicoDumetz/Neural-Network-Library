##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## tensor
##

# **************************************************************************** #
#                                                                              #
#                               TENSOR CORE API                                #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
from typing import Tuple, List, Set
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from library.Tensor.BackwardOP import (
    BackwardOp,
    AddBackward,
    MulBackward,
    MatMulBackward,
    Conv2dBackward,
    MaxPool2dBackward,
    NegBackward,
    DivBackward,
    ReluBackward,
    ExpBackward,
    LogBackward,
    SigmoidBackward,
    TanhBackward,
    SoftmaxBackward,
    LogSoftmaxBackward,
    SumBackward,
    MeanBackward,
    ClampBackward,
    PowBackward,
    TransposeBackward,
    ReshapeBackward,
    DropoutBackward,
    AbsBackward,
)

class Tensor:
    ##
    # @brief Create a new tensor.
    # @details Converts input to NumPy, configures dtype and gradient flag.
    # Initializes gradient and autograd metadata to None.
    ##
    def __init__(self, data: ArrayLike, requires_grad: bool = False, dtype: DTypeLike = np.float32) -> None:
        array = np.array(data, dtype=dtype)
        self.data: np.ndarray = array
        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray | None = None
        self._parents: Tuple[Tensor, ...] = tuple()
        self._op: BackwardOp | None = None

    ##
    # @brief Build topological graph ordering.
    # @details Depth-first traversal ensures parents appear before children.
    # Used by backward() to compute gradients in reverse.
    ##
    def _build_graph(self, topo: List[Tensor], visited: Set[Tensor]) -> None:
        if self in visited:
            return
        visited.add(self)
        if self._parents:
            for parent in self._parents:
                parent._build_graph(topo, visited)
        topo.append(self)

    ##
    # @brief Launch reverse-mode autodiff.
    # @details Seeds gradient, builds topological order, applies backward ops.
    # Gradients populate self.grad and parents that require them.
    ##
    def backward(self, gradient: np.ndarray | None = None) -> None:
        self.grad = gradient if gradient is not None else np.ones_like(self.data)
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()
        self._build_graph(topo, visited)
        for tensor in reversed(topo):
            if tensor._op is not None:
                tensor._op.backward(tensor.grad, tensor._parents)
    ##
    # @brief Helper for binary operations.
    # @details Connects both parents and stores backward operator.
    ##
    def _op_tensor(self, other: Tensor, data: np.ndarray, op: BackwardOp) -> Tensor:
        out = Tensor(data, requires_grad=self.requires_grad or other.requires_grad)
        out._parents = (self, other)
        out._op = op
        return out

    ##
    # @brief Helper for unary operations.
    # @details Links this tensor as the single parent and stores the operator.
    ##
    def _op_unary(self, data: np.ndarray, op: BackwardOp) -> Tensor:
        out = Tensor(data, requires_grad=self.requires_grad)
        out._parents = (self,)
        out._op = op
        return out

    ##
    # @brief Element-wise addition.
    # @details Performs broadcasted x + y and forwards gradients unchanged.
    # Used heavily in neural nets (biases, residuals).
    # @example Tensor([1,2,3]) + Tensor([4,5,6])
    ##
    def __add__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self._op_tensor(other_t, self.data + other_t.data, AddBackward())

    ##
    # @brief Element-wise multiplication.
    # @details Computes x*y with broadcasting; gradients use product rule.
    # Common in gating and attention mechanisms.
    # @example Tensor([2,3]) * Tensor([4,5])
    ##
    def __mul__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self._op_tensor(other_t, self.data * other_t.data, MulBackward())

    ##
    # @brief Matrix multiplication (A@B).
    # @details Uses NumPy dot; gradients follow dA=grad@Bᵀ and dB=Aᵀ@grad.
    # Core component of neural network linear layers.
    # @example Tensor([[1,2]]) @ Tensor([[3],[4]])
    ##
    def __matmul__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        data_result = np.matmul(self.data, other_t.data)
        return self._op_tensor(other_t, data_result, MatMulBackward())

    ##
    # @brief Apply a 2D convolution between an activation map and kernels.
    # @details Performs a zero-padded sliding window convolution with optional
    # stride adjustment and stores an autograd operator to propagate gradients
    # back to both the input tensor and the filter weights.
    ##
    def conv2d(self, weight: "Tensor", stride: int | tuple[int, int] = 1, padding: int | tuple[int, int] = 0,) -> "Tensor":
        stride_pair = _ensure_pair(stride, "stride")
        padding_pair = _ensure_pair(padding, "padding")
        if self.data.ndim != 4:
            raise ValueError("Conv2d expects input tensors in NCHW format.")
        if weight.data.ndim != 4:
            raise ValueError("Conv2d weights must be 4D (out_channels, in_channels, kH, kW).")
        if self.data.shape[1] != weight.data.shape[1]:
            raise ValueError(
                f"Input channels ({self.data.shape[1]}) do not match weight channels ({weight.data.shape[1]})."
            )
        output = _conv2d_forward_numpy(self.data, weight.data, stride_pair, padding_pair)
        backward = Conv2dBackward(self.data.shape, weight.data.shape, stride_pair, padding_pair)
        return self._op_tensor(weight, output, backward)

    ##
    # @brief Apply 2D max pooling over the input activations.
    # @details Down-samples the input tensor by taking the maximal value inside
    # each receptive field while storing argmax indices for gradient routing.
    ##
    def max_pool2d(self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None, padding: int | tuple[int, int] = 0,) -> "Tensor":
        if self.data.ndim != 4:
            raise ValueError("MaxPool2d expects input tensors in NCHW format.")
        kernel = _ensure_pair(kernel_size, "kernel_size")
        stride_value = stride if stride is not None else kernel
        stride_pair = _ensure_pair(stride_value, "stride")
        padding_pair = _ensure_pair(padding, "padding")
        output, indices = _max_pool2d_forward_numpy(self.data, kernel, stride_pair, padding_pair)
        backward = MaxPool2dBackward(self.data.shape, kernel, stride_pair, padding_pair, indices)
        return self._op_unary(output, backward)
    ##
    # @brief Unary negation.
    # @details Computes -x and multiplies gradient by -1.
    # @example -Tensor([1,-2])
    ##
    def __neg__(self) -> Tensor:
        return self._op_unary(-self.data, NegBackward())

    ##
    # @brief Element-wise subtraction.
    # @details Implemented as x + (-y), reusing addition logic.
    # @example Tensor([5,6]) - Tensor([1,2])
    ##
    def __sub__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self.__add__(-other_t)

    ##
    # @brief Reverse subtraction (y - x).
    # @details Implemented as y + (-x); same gradients as normal subtraction.
    # @example 10 - Tensor([3])
    ##
    def __rsub__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return other_t.__add__(-self)

    ##
    # @brief Element-wise division.
    # @details Computes x/y; gradients follow quotient rule with broadcasting.
    # @example Tensor([10,20]) / Tensor([2,5])
    ##
    def __truediv__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self._op_tensor(other_t, self.data / other_t.data, DivBackward())

    ##
    # @brief Reverse division (y / x).
    # @details Same division logic but operands swapped.
    # @example 20 / Tensor([4])
    ##
    def __rtruediv__(self, other: Tensor | ArrayLike) -> Tensor:
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self._op_tensor(self, other_t.data / self.data, DivBackward())

    ##
    # @brief ReLU activation.
    # @details Applies max(0,x); zeros gradient for negative inputs.
    # @example Tensor([-1,2]).relu()
    ##
    def relu(self) -> Tensor:
        return self._op_unary(np.maximum(0, self.data), ReluBackward())

    ##
    # @brief Sum reduction.
    # @details Sums elements and stores shape info for gradient broadcast.
    ##
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        data = self.data.sum(axis=axis, keepdims=keepdims)
        return self._op_unary(data, SumBackward(self.data.shape, axis, keepdims))

    ##
    # @brief Mean reduction.
    # @details Averages elements and rescales gradients by element count.
    ##
    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        data = self.data.mean(axis=axis, keepdims=keepdims)
        return self._op_unary(data, MeanBackward(self.data.shape, axis, keepdims))

    ##
    # @brief Element-wise exponential.
    # @details Computes exp(x); gradient also uses exp(x).
    # @example Tensor([0]).exp()
    ##
    def exp(self) -> Tensor:
        return self._op_unary(np.exp(self.data), ExpBackward())

    ##
    # @brief Element-wise natural log.
    # @details Computes ln(x); gradient is 1/x.
    # @example Tensor([1,2]).log()
    ##
    def log(self) -> Tensor:
        safe_data = np.maximum(self.data, 1e-8)
        return self._op_unary(np.log(safe_data), LogBackward())
    ##
    # @brief Sigmoid activation.
    # @details Computes 1/(1+exp(-x)); gradient uses σ(x)(1-σ(x)).
    # @example Tensor([0]).sigmoid()
    ##
    def sigmoid(self) -> Tensor:
        out = 1 / (1 + np.exp(-self.data))
        return self._op_unary(out, SigmoidBackward(out))

    ##
    # @brief Hyperbolic tangent.
    # @details Uses tanh(x); gradient = 1 - tanh(x)².
    # @example Tensor([1]).tanh()
    ##
    def tanh(self) -> Tensor:
        out = np.tanh(self.data)
        return self._op_unary(out, TanhBackward(out))

    ##
    # @brief Softmax activation.
    # @details Normalizes values into probabilities; stable via shifting.
    # @example Tensor([[1,2,3]]).softmax()
    ##
    def softmax(self) -> Tensor:
        exp_x = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self._op_unary(out, SoftmaxBackward(out))

    ##
    # @brief Log-Softmax activation.
    # @details Computes log probabilities in a numerically stable way.
    # @example Tensor([[1,2]]).log_softmax()
    ##
    def log_softmax(self) -> Tensor:
        shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        log_soft = shifted - np.log(np.sum(exp_x, axis=-1, keepdims=True))
        return self._op_unary(log_soft, LogSoftmaxBackward(softmax))

    ##
    # @brief Reshape tensor.
    # @details Stores original shape for backward broadcasting.
    ##
    def reshape(self, shape: tuple[int, ...]) -> Tensor:
        return self._op_unary(self.data.reshape(shape), ReshapeBackward(self.data.shape))

    ##
    # @brief Transpose tensor axes.
    # @details Permutes axes; backward restores original permutation.
    ##
    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        return self._op_unary(self.data.transpose(axes), TransposeBackward(axes))

    ##
    # @brief Element-wise exponentiation.
    # @details Computes x**p; gradient = p*x^(p-1).
    # @example Tensor([2]).pow(3)
    ##
    def pow(self, power: float) -> Tensor:
        return self._op_unary(np.power(self.data, power), PowBackward(power))

    ##
    # @brief Clamp tensor values.
    # @details Values outside [min,max] are clipped and gradients blocked.
    # @example Tensor([-5,3]).clamp(0,2)
    ##
    def clamp(self, min: float | None = None, max: float | None = None) -> Tensor:
        return self._op_unary(np.clip(self.data, min, max), ClampBackward(min, max))

    ##
    # @brief Remove singleton dimensions.
    # @details Stores original shape for reshaping gradient.
    ##
    def squeeze(self, axis: int | None = None) -> Tensor:
        return self._op_unary(np.squeeze(self.data, axis=axis), ReshapeBackward(self.data.shape))

    ##
    # @brief Insert singleton dimension.
    # @details Expands shape and records original for backward.
    ##
    def unsqueeze(self, axis: int) -> Tensor:
        return self._op_unary(np.expand_dims(self.data, axis=axis), ReshapeBackward(self.data.shape))

    ##
    # @brief Apply dropout to the tensor using a binary mask.
    # @details Multiplies the tensor’s data by the dropout mask and scaling factor.
    # Records a backward operator that blocks gradients on dropped elements.
    ##
    def dropout(self, mask: np.ndarray, scale: float) -> "Tensor":
        return self._op_unary(self.data * mask * scale, DropoutBackward(mask, scale))
    ##
    # @brief Compute the element-wise absolute value.
    # @details Applies |x| element-wise and registers the corresponding backward
    # operator so gradients flow according to the sign of the original values.
    ##
    def abs(self) -> Tensor:
        return self._op_unary(np.abs(self.data), AbsBackward())

    ##
    # @brief Reverse element-wise multiplication.
    # @details Allows expressions where the tensor appears on the right side of '*'.
    # Delegates to __mul__ for consistent gradient tracking and broadcasting.
    ##
    def __rmul__(self, other: Tensor | ArrayLike) -> Tensor:
        return self.__mul__(other)

    ##
    # @brief Reverse element-wise addition.
    # @details Enables expressions where the tensor appears on the right side of '+'.
    # Redirects to __add__ to preserve autograd behavior and operand symmetry.
    ##
    def __radd__(self, other: Tensor | ArrayLike) -> Tensor:
        return self.__add__(other)


def _ensure_pair(value: int | tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    ##
    # @brief Normalize convolution hyperparameters to a tuple.
    # @details Accepts either a single integer or a tuple of two integers and
    # returns a (height, width) pair to keep downstream logic uniform.
    ##
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two integers.")
        return int(value[0]), int(value[1])
    if isinstance(value, int):
        return value, value
    raise TypeError(f"{name} must be either an int or a tuple of two ints.")

##
# @brief Pure NumPy implementation of a 2D convolution.
# @details Pads the input tensor, extracts each receptive field, and applies
# a tensor dot-product with the kernels to generate the activation map.
##
def _conv2d_forward_numpy(input_data: np.ndarray, weight_data: np.ndarray, stride: tuple[int, int], padding: tuple[int, int]) -> np.ndarray:
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("Stride values must be positive.")
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Padding must be non-negative.")
    batch, _, height, width = input_data.shape
    out_channels, _, kernel_h, kernel_w = weight_data.shape
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    padded = np.pad(
        input_data,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    out_h = ((height + 2 * pad_h - kernel_h) // stride_h) + 1
    out_w = ((width + 2 * pad_w - kernel_w) // stride_w) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Convolution parameters yield an empty output tensor.")
    output = np.zeros((batch, out_channels, out_h, out_w), dtype=input_data.dtype)
    for i in range(out_h):
        h_start = i * stride_h
        h_end = h_start + kernel_h
        for j in range(out_w):
            w_start = j * stride_w
            w_end = w_start + kernel_w
            region = padded[:, :, h_start:h_end, w_start:w_end]
            output[:, :, i, j] = np.tensordot(
                region,
                weight_data,
                axes=([1, 2, 3], [1, 2, 3]),
            )
    return output

##
# @brief Pure NumPy implementation of 2D max pooling.
# @details Pads the input with -inf, extracts each receptive field, and
# caches the argmax index per window for use during backpropagation.
##
def _max_pool2d_forward_numpy(input_data: np.ndarray, kernel_size: tuple[int, int], stride: tuple[int, int], padding: tuple[int, int],) -> tuple[np.ndarray, np.ndarray]:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    if kernel_h <= 0 or kernel_w <= 0:
        raise ValueError("Kernel dimensions must be positive for MaxPool2d.")
    if stride_h <= 0 or stride_w <= 0:
        raise ValueError("Stride values must be positive for MaxPool2d.")
    if pad_h < 0 or pad_w < 0:
        raise ValueError("Padding must be non-negative for MaxPool2d.")
    batch, channels, height, width = input_data.shape
    padded = np.pad(
        input_data,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    out_h = ((height + 2 * pad_h - kernel_h) // stride_h) + 1
    out_w = ((width + 2 * pad_w - kernel_w) // stride_w) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("MaxPool2d parameters yield an empty output tensor.")
    output = np.zeros((batch, channels, out_h, out_w), dtype=input_data.dtype)
    indices = np.zeros((batch, channels, out_h, out_w), dtype=np.int32)
    for i in range(out_h):
        h_start = i * stride_h
        h_end = h_start + kernel_h
        for j in range(out_w):
            w_start = j * stride_w
            w_end = w_start + kernel_w
            window = padded[:, :, h_start:h_end, w_start:w_end]
            flat_window = window.reshape(batch, channels, -1)
            max_idx = np.argmax(flat_window, axis=2)
            max_vals = np.take_along_axis(flat_window, max_idx[..., None], axis=2).squeeze(-1)
            output[:, :, i, j] = max_vals
            indices[:, :, i, j] = max_idx
    return output, indices
