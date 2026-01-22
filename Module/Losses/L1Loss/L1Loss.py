##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## L1Loss
##

from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                                 LOSS CORE API                                #
#                                                                              #
# **************************************************************************** #


class L1Loss(Module):
    ##
    # @brief Initialize the L1 loss module.
    # @details Sets up a parameter-free regression loss that measures the mean
    # absolute error between predictions and targets. Useful for robust training
    # when outliers should have reduced influence compared to L2 loss.
    ##
    def __init__(self) -> None:
        super().__init__()

    ##
    # @brief Compute the mean absolute error between predictions and targets.
    # @details Subtracts the target from the prediction, applies an element-wise
    # absolute value, and averages all elements to produce the final loss. Supports
    # autograd and broadcasting for multi-dimensional tensors.
    ##
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        out = diff.abs().mean()
        return out

    ##
    # @brief Return a string representation of the L1 loss module.
    # @details Useful when printing model structures or logging the training setup.
    ##
    def __repr__(self) -> str:
        return "L1Loss()"