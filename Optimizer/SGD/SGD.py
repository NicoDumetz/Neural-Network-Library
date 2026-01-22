##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## SGD
##
from __future__ import annotations
from typing import Iterable, Dict
import numpy as np
from library.Parameter.Parameter import Parameter

# **************************************************************************** #
#                                                                              #
#                                   SGD API                                    #
#                                                                              #
# **************************************************************************** #


class SGD:
    ##
    # @brief Initialize the stochastic gradient descent optimizer.
    # @details Stores learning rate, momentum coefficient, and both L1 and L2
    # regularization factors. Allocates velocity buffers used for momentum-based
    # updates and prepares optimization state for iterative training.
    ##
    def __init__(self, params: Iterable[Parameter], lr: float = 0.01, momentum: float = 0.0,weight_decay: float = 0.0,l1_decay: float = 0.0,) -> None:
        self.params = list(params)
        self.lr: float = lr
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.l1_decay: float = l1_decay
        self.velocities: Dict[Parameter, np.ndarray] = {}
        for p in self.params:
            self.velocities[p] = np.zeros_like(p.data)

    ##
    # @brief Apply L1 and L2 regularization to a gradient.
    # @details Adds weight decay and sparsity-inducing L1 terms to the gradient,
    # then applies the parameter’s pruning mask to disable updates on pruned
    # weights.
    ##
    def _apply_regularization(self, grad: np.ndarray, param: Parameter) -> np.ndarray:
        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param.data
        if self.l1_decay > 0.0:
            grad = grad + self.l1_decay * np.sign(param.data)
        mask = getattr(param, "mask", None)
        return grad if mask is None else grad * mask

    ##
    # @brief Update a parameter using momentum-based SGD.
    # @details Combines the previous velocity with the current gradient to compute
    # a momentum-driven update, applies the learning rate scaling, subtracts the
    # velocity from the parameter data, and re-applies the pruning mask.
    ##
    def _update_parameter(self, param: Parameter, grad_eff: np.ndarray) -> None:
        mu = self.momentum
        lr = self.lr
        v_prev = self.velocities[param]
        v = mu * v_prev + lr * grad_eff
        self.velocities[param] = v
        param.data = param.data - v
        mask = getattr(param, "mask", None)
        param.data = param.data if mask is None else param.data * mask

    ##
    # @brief Perform one optimization step over all parameters.
    # @details Filters out parameters without gradients, applies regularization,
    # computes the effective gradient, and updates corresponding parameters using
    # the momentum-enhanced SGD update rule.
    ##
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            grad_effective = self._apply_regularization(p.grad, p)
            self._update_parameter(p, grad_effective)

    ##
    # @brief Reset gradients for all tracked parameters.
    # @details Clears gradients by setting them to None, preventing accumulation
    # across training iterations.
    ##
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None