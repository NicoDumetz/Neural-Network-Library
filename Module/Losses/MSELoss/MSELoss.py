##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## MSELoss
##
from __future__ import annotations
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                                 LOSS CORE API                                #
#                                                                              #
# **************************************************************************** #


class MSELoss(Module):
    ##
    # @brief Initialize the mean squared error loss module.
    # @details Configures a parameter-free regression loss that penalizes the
    # squared distance between predictions and targets. Widely used when smooth
    # gradients and strong penalization of large errors are desired.
    ##
    def __init__(self) -> None:
        super().__init__()

    ##
    # @brief Compute the mean squared error between predictions and targets.
    # @details Computes the element-wise difference, squares it, and averages all
    # values to produce the final loss. Produces smooth gradients that facilitate
    # stable optimization in regression tasks.
    ##
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        out = (diff * diff).mean()
        return out

    ##
    # @brief Return a string representation of the MSE loss module.
    # @details Useful when visualizing model architectures or printing debugging
    # information during training.
    ##
    def __repr__(self) -> str:
        return "MSELoss()"
