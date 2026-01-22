##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Prunning
##
import numpy as np
from library.Parameter.Parameter import Parameter

# **************************************************************************** #
#                                                                              #
#                               TENSOR CORE API                                #
#                                                                              #
# **************************************************************************** #

##
# @brief Apply magnitude-based pruning on a module.
# @details Iterates through all parameters inside the module and its submodules.
# Creates a pruning mask by keeping values whose absolute magnitude is above
# the provided threshold. Recursively descends into nested modules to ensure
# full-structure pruning.
##
def prune_by_magnitude(module, threshold: float):
    from library.Module.Module import Module
    for name, param in module.__dict__.items():
        if isinstance(param, Parameter):
            param.mask = np.abs(param.data) >= threshold
        if isinstance(param, Module):
            prune_by_magnitude(param, threshold)
