##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Initiailizer
##
import numpy as np

# **************************************************************************** #
#                                                                              #
#                          PARAMETER INITIALIZERS API                          #
#                                                                              #
# **************************************************************************** #

##
# @brief Xavier uniform initializer.
# @details Initializes weights using a uniform distribution in [-limit, limit],
# where limit = sqrt(6 / (fan_in + fan_out)). Designed to preserve variance
# across layers for activation functions like tanh or sigmoid.
# @formula limit = sqrt(6 / (fan_in + fan_out))
##
def xavier_uniform(shape: tuple[int, ...], fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)

##
# @brief He uniform initializer.
# @details Initializes weights using a uniform distribution in [-limit, limit],
# where limit = sqrt(6 / fan_in). Optimized for ReLU-based activations by
# increasing variance to counteract neuron dropout in negative domain.
# @formula limit = sqrt(6 / fan_in)
##
def he_uniform(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape).astype(np.float32)

##
# @brief Zero-filled initializer.
# @details Creates a tensor filled with zeros. Typically used for bias vectors
# or explicit parameter resets. Produces no initial variance.
# @formula W = 0
##
def zeros_initializer(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)

##
# @brief One-filled initializer.
# @details Creates a tensor filled with ones. Rarely used for weights, but
# sometimes employed for scale parameters (BatchNorm, LayerNorm).
# @formula W = 1
##
def ones_initializer(shape: tuple[int, ...]) -> np.ndarray:
    return np.ones(shape, dtype=np.float32)
