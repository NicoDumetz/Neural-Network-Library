from __future__ import annotations
from typing import Iterable, Dict
import numpy as np
from library.Parameter.Parameter import Parameter

# **************************************************************************** #
#                                                                              #
#                                   ADAM API                                   #
#                                                                              #
# **************************************************************************** #

class Adam:
    ##
    # @brief Initialize the Adam optimizer.
    # @details Registers all trainable parameters, creates state buffers for first
    # and second moment estimates, and sets hyperparameters such as learning rate,
    # beta coefficients, epsilon, and both L1/L2 regularization factors. Each
    # parameter receives its own moment vectors m, v and timestep t.
    ##
    def __init__(self, params: Iterable[Parameter], lr: float = 0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, l1_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.l1_decay = l1_decay
        self.state: Dict[Parameter, Dict[str, np.ndarray]] = {}
        for p in self.params:
            self.state[p] = {
                "m": np.zeros_like(p.data),
                "v": np.zeros_like(p.data),
                "t": 0
            }

    ##
    # @brief Apply L1/L2 regularization and pruning mask.
    # @details Modifies the raw gradient by adding L2 weight decay (λ · W), L1 decay
    # (λ · sign(W)), and masks pruned parameters so they receive no gradient.
    # Functions exactly like PyTorch’s regularization logic, but integrated into
    # the gradient pipeline.
    ##
    def _apply_regularization(self, grad: np.ndarray, param: Parameter) -> np.ndarray:
        if self.weight_decay > 0.0:
            grad = grad + self.weight_decay * param.data
        if self.l1_decay > 0.0:
            grad = grad + self.l1_decay * np.sign(param.data)
        mask = getattr(param, "mask", None)
        return grad if mask is None else grad * mask

    ##
    # @brief Update a single parameter using Adam rules.
    # @details Maintains exponential moving averages of gradients (m) and squared
    # gradients (v), applies bias correction m̂ and v̂, and computes the adaptive
    # update:
    # @formula m = β₁·m + (1-β₁)·g
    # @formula v = β₂·v + (1-β₂)·g²
    # @formula m̂ = m / (1-β₁ᵗ)
    # @formula v̂ = v / (1-β₂ᵗ)
    # @formula update = lr · m̂ / (sqrt(v̂) + eps)
    # Applies pruning masks when present.
    ##
    def _update_parameter(self, param: Parameter, grad_eff: np.ndarray) -> None:
        state = self.state[param]
        m = state["m"]
        v = state["v"]
        t = state["t"] + 1
        state["t"] = t
        b1, b2 = self.betas
        lr = self.lr
        m = b1 * m + (1 - b1) * grad_eff
        v = b2 * v + (1 - b2) * (grad_eff * grad_eff)
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + self.eps)
        param.data = param.data - update
        mask = getattr(param, "mask", None)
        if mask is not None:
            param.data = param.data * mask
        state["m"] = m
        state["v"] = v

    ##
    # @brief Perform one optimization step.
    # @details Iterates through all parameters, skips those without gradients,
    # applies regularization, and computes the Adam update through
    # _update_parameter(). Equivalent to PyTorch's optimizer.step().
    ##
    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            grad_eff = self._apply_regularization(p.grad, p)
            self._update_parameter(p, grad_eff)

    ##
    # @brief Reset gradients of all parameters.
    # @details Sets parameter.grad to None for each tracked parameter to avoid
    # gradient accumulation across backward passes.
    ##
    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
