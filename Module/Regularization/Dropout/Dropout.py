##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Dropout
##
from __future__ import annotations
import numpy as np
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor

# **************************************************************************** #
#                                                                              #
#                                DROPOUT CORE API                              ""
#                                                                              #
# **************************************************************************** #


class Dropout(Module):
    ##
    # @brief Initialize a dropout module.
    # @details Stores the dropout probability used to randomly deactivate a
    # fraction of the input units during training. Helps reduce overfitting by
    # preventing co-adaptation between neurons.
    ##
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    ##
    # @brief Apply dropout regularization to the input tensor.
    # @details During training, generates a binary mask using the dropout
    # probability, scales the surviving activations to maintain expected output
    # magnitude, and applies the mask using tensor-level dropout. When not in
    # training mode, returns the input unchanged.
    ##
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        prob = self.p
        scale = 1.0 / (1.0 - prob)
        mask = (np.random.rand(*x.data.shape) > prob).astype(x.data.dtype)
        return x.dropout(mask, scale)

    ##
    # @brief Return a readable representation of the dropout module.
    # @details Exposes the configured dropout probability for debugging and
    # architecture inspection.
    ##
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"