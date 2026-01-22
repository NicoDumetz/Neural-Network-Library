##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## quantize_weight
##

import numpy as np

# **************************************************************************** #
#                                                                              #
#                               TENSOR CORE API                                #
#                                                                              #
# **************************************************************************** #

##
# @brief Quantize floating-point weights to int8.
# @details Converts weights into 8-bit integers using asymmetric affine
# quantisation. Computes data_min, data_max, a scaling factor, and a zero-point
# that maps the real value range [data_min, data_max] to the quantized range
# [q_min, q_max]. Handles near-constant tensors by falling back to scale=1 and
# minimal offset.
# @formula q_min=-127,\ q_max=127
# @formula scale = (data_max - data_min) / (q_max - q_min)
# @formula zero_point = round(q_min - data_min / scale)
# @formula q_data = clip(round(data / scale + zero_point), q_min, q_max)
##
def quantize_weights(data: np.ndarray, num_bits: int = 8) -> tuple[np.ndarray, float, float]:
    if num_bits != 8:
        raise NotImplementedError("Only 8-bit quantisation is supported.")
    q_min = -127
    q_max = 127
    data_fp64 = data.astype(np.float64)
    data_min = data_fp64.min()
    data_max = data_fp64.max()
    data_range = data_max - data_min
    if data_range < 1e-12:
        scale = 1.0
        zero_point = q_min
    else:
        scale = data_range / (float(q_max) - float(q_min))
        zero_point_ideal = float(q_min) - (data_min / scale)
        zero_point = np.round(zero_point_ideal)
    q_data = np.round(data_fp64 / scale + zero_point)
    q_data = np.clip(q_data, q_min, q_max).astype(np.int8)
    return q_data, float(scale), float(zero_point)

##
# @brief Dequantize int8 weights back to float32.
# @details Restores float32 weights using the affine inverse transform defined
# by scale and zero-point. Reconstructs an approximation of the original real
# values prior to quantisation.
# @formula data ≈ scale * (q_data - zero_point)
##
def dequantize_weights(q_data: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    return scale * (q_data.astype(np.float32) - zero_point)
