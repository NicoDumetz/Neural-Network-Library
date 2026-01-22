##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Sheduler
##
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from library.Optimizer.SGD.SGD import SGD
    from library.Optimizer.Adam.Adam import Adam

# **************************************************************************** #
#                                                                              #
#                                SCHEDULER CORE API                            #
#                                                                              #
# **************************************************************************** #


class Scheduler:
    ##
    # @brief Initialize a learning-rate scheduler.
    # @details Stores a reference to the optimizer, records the base learning rate,
    # and initializes the epoch counter. Serves as an abstract interface for all
    # scheduling strategies.
    ##
    def __init__(self, optimizer: 'SGD' | 'Adam') -> None:
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    ##
    # @brief Advance the scheduler by one step.
    # @details Increments the internal epoch counter, computes a new learning rate
    # using the scheduler’s policy, and updates the optimizer accordingly. Optionally
    # accepts metrics for adaptive or plateau-based scheduling.
    ##
    def step(self, metrics: float | None = None) -> None:
        self.last_epoch += 1
        new_lr = self.get_lr(metrics)
        self.optimizer.lr = new_lr

    ##
    # @brief Compute the next learning rate according to the scheduling rule.
    # @details Must be implemented by subclasses. Can use epoch count, metrics, or
    # other signals to generate dynamic learning-rate adjustments.
    ##
    def get_lr(self, metrics: float | None = None) -> float:
        raise NotImplementedError