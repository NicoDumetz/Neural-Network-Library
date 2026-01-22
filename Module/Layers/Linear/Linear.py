##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Linear
##
from library.Module.Module import Module
from library.Parameter.Parameter import Parameter
import numpy as np
from library.Tensor.Tensor import Tensor
from library.Helper.Initializers import xavier_uniform, he_uniform, zeros_initializer

# **************************************************************************** #
#                                                                              #
#                              LINEAR LAYER  API                               #
#                                                                              #
# **************************************************************************** #


class Linear(Module):
    ##
    # @brief Initialize a fully-connected linear layer.
    # @details Creates a weight matrix and bias vector using Xavier-like uniform
    # initialization based on the input and output dimensions. Allocates parameter.
    # with gradient tracking enabled and prepares masks for pruning-aware training.
    ##
    def __init__(self, in_features: int, out_features: int, init_method: str = 'xavier') -> None:
        super().__init__()
        if init_method == 'xavier':
            weight_initializer = xavier_uniform
            init_kwargs = {'fan_in': in_features, 'fan_out': out_features}
        elif init_method == 'he':
            weight_initializer = he_uniform
            init_kwargs = {'fan_in': in_features} 
        elif init_method == 'zeros':
            weight_initializer = zeros_initializer
            init_kwargs = {}
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        weight_data = weight_initializer(shape=(in_features, out_features), **init_kwargs)
        bias_data = zeros_initializer((out_features,))
        self.weight = Parameter(weight_data)
        self.bias = Parameter(bias_data)
        self.in_features = in_features
        self.out_features = out_features

    ##
    # @brief Apply the linear transformation to the input tensor.
    # @details Performs the forward pass by applying a masked matrix multiplication
    # between the input and the prunable weight matrix, then adds the bias vector.
    # The masking mechanism allows structured pruning without modifying parameter
    # shapes.
    ##
    def forward(self, x: Tensor) -> Tensor:
        return x.__matmul__(self.weight) + self.bias