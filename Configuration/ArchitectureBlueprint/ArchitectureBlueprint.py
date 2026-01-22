##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## ArchitectureBlueprint
##

# **************************************************************************** #
#                                                                              #
#                           CONFIGURATION MODELE                               #
#                                                                              #
# **************************************************************************** #

from __future__ import annotations
from typing import List, Dict, Any


class ArchitectureBlueprint:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        layers: List[Dict[str, Any]],
        output_activation: str,
        initialization_method: str,
        weight_decay: float,
    ):
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.layers: List[Dict[str, Any]] = layers
        self.output_activation: str = output_activation
        self.initialization_method: str = initialization_method
        self.weight_decay: float = weight_decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "layers": self.layers,
            "output_activation": self.output_activation,
            "initialization_method": self.initialization_method,
            "weight_decay": self.weight_decay,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureBlueprint":
        return cls(
            input_size=int(data.get("input_size", 0)),
            output_size=int(data.get("output_size", 0)),
            layers=list(data.get("layers", [])),
            output_activation=str(data.get("output_activation", "")),
            initialization_method=str(data.get("initialization_method", "")),
            weight_decay=float(data.get("weight_decay", 0.0)),
        )
