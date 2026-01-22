##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## CrossEntropyLoss
##
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from library.Module.Module import Module
from library.Tensor.Tensor import Tensor
from library.Tensor.BackwardOP.UnaryBackwardOp.NLLLossBackward.NLLLossBackward import NLLLossBackward
from library.Tensor.BackwardOP.UnaryBackwardOp.NLLLossBackward.SmoothedNLLLossBackward import SmoothedNLLLossBackward

# **************************************************************************** #
#                                                                              #
#                                 LOSS CORE API                                #
#                                                                              #
# **************************************************************************** #


class CrossEntropyLoss(Module):
    ##
    # @brief Initialize the cross-entropy loss module.
    # @details Prepares the module for multi-class classification. This loss
    # combines a log-softmax computation with a negative log-likelihood (NLL)
    # reduction, ensuring numerical stability and compatibility with autograd.
    ##
    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0.0, 1.0).")
        self.label_smoothing = label_smoothing

    ##
    # @brief Compute the cross-entropy loss between logits and class targets.
    # @details Applies log-softmax to obtain log-probabilities, selects the
    # log-probability corresponding to each target class, averages the negative
    # values to compute the final loss, and registers an NLL backward operator to
    # propagate gradients correctly through the log-softmax operation.
    ##
    def forward(self, logits: Tensor, target: ArrayLike) -> Tensor:
        log_probs = logits.log_softmax()
        target_np = np.array(target, dtype=np.int64)
        batch_indices = np.arange(log_probs.data.shape[0])
        N, num_classes = log_probs.data.shape
        if self.label_smoothing == 0.0:
            selected_log_probs = log_probs.data[batch_indices, target_np]
            final_loss_data = -selected_log_probs.mean()
            out = Tensor(final_loss_data)
            factor = 1.0 / N
            out._parents = (log_probs,)
            out._op = NLLLossBackward(target=target_np, factor=factor)
            return out

        smooth_targets = np.full_like(log_probs.data, self.label_smoothing / num_classes)
        smooth_targets[batch_indices, target_np] += 1.0 - self.label_smoothing
        loss_per_sample = -(smooth_targets * log_probs.data).sum(axis=1)
        final_loss_data = loss_per_sample.mean()
        out = Tensor(final_loss_data)
        factor = 1.0 / N
        out._parents = (log_probs,)
        out._op = SmoothedNLLLossBackward(smooth_targets=smooth_targets, factor=factor)
        return out

    ##
    # @brief Return a readable identifier for the cross-entropy module.
    # @details Useful when inspecting model structures or printing composed
    # architectures containing multiple loss functions.
    ##
    def __repr__(self) -> str:
        return "CrossEntropyLoss()"
