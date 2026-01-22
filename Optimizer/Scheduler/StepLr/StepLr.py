##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## StepLr
##
from __future__ import annotations
from library.Optimizer.Scheduler.Scheduler import Scheduler

# **************************************************************************** #
#                                                                              #
#                                 STEPLR API                                   #
#                                                                              #
# **************************************************************************** #


class StepLR(Scheduler):
    ##
    # @brief Initialize a step-based learning-rate scheduler.
    # @details Stores the update interval and decay factor. The learning rate is
    # reduced by multiplying with gamma every step_size epochs, following a
    # staircase decay policy.
    ##
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    ##
    # @brief Compute the scheduled learning rate for the current epoch.
    # @details Applies a discrete decay rule where the learning rate is multiplied
    # by gamma each time the number of elapsed epochs reaches a multiple of
    # step_size. Independent of any provided metric.
    ##
    def get_lr(self, metrics: float | None = None) -> float:
        power = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma ** power)

    ##
    # @brief Return a string representation of the StepLR scheduler.
    # @details Displays the decay interval and gamma factor, useful for debugging
    # and logging in training loops.
    ##
    def __repr__(self) -> str:
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"