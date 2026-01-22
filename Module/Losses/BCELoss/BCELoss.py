##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## BCELoss
##

from library.Module.Module import Module
from library.Tensor.Tensor import Tensor
import numpy as np

# **************************************************************************** #
#                                                                              #
#                              LINEAR LAYER  API                               #
#                                                                              #
# **************************************************************************** #

class BCELoss(Module):
    ##
    # @brief Initialize the binary cross-entropy loss module.
    # @details Sets up numerical-stability parameters such as epsilon to prevent
    # undefined behavior when taking logarithms of extreme prediction values.
    # Prepares the loss function for binary classification tasks.
    ##
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-8

    ##
    # @brief Compute the binary cross-entropy loss between predictions and targets.
    # @details Clamps predictions to avoid log(0), computes y*log(p) and
    # (1-y)*log(1-p), sums them, applies a negative sign, and returns the mean loss.
    # Works with Tensor autograd and supports broadcasting.
    ##
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        safe_pred = y_pred.clamp(min=self.epsilon, max=1.0 - self.epsilon)
        term_1 = y_true * safe_pred.log()
        one_minus_y_true = Tensor(1.0, dtype=y_true.data.dtype) - y_true
        one_minus_safe_pred = Tensor(1.0, dtype=y_pred.data.dtype) - safe_pred
        term_2 = one_minus_y_true * one_minus_safe_pred.log()
        loss = -(term_1 + term_2).mean()
        return loss

    ##
    # @brief Return a string representation of the BCE loss module.
    # @details Provides a clear identifier for debugging and logging within model
    # architectures that use multiple loss components.
    ##
    def __repr__(self) -> str:
        return "BCELoss()"