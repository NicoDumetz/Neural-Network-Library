##
## EPITECH PROJECT, 2025
## G-CNA-500-LIL-5-1-neuralnetwork-1
## File description:
## Module
##
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
from library.Parameter.Parameter import Parameter
from library.Helper.Quantize import quantize_weights, dequantize_weights
import numpy as np
import pickle
import copy
import json
from library.Configuration.ArchitectureBlueprint import ArchitectureBlueprint

# **************************************************************************** #
#                                                                              #
#                                  MODULE API                                  #
#                                                                              #
# **************************************************************************** #


class Module:
    @staticmethod
    def _extract_serialized_blob(container):
        if isinstance(container, np.ndarray):
            if container.dtype == object:
                return container.flat[0]
            if container.shape == ():
                return container.item()
            return container.tobytes()
        return container
    ##
    # @brief Initialize a base neural network module.
    # @details Sets up parameter and submodule registries, along with a training
    # mode flag. Acts as the foundational class for all layers and models in the
    # framework.
    ##
    def __init__(self) -> None:
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, Module] = {}
        self.training = True

    ##
    # @brief Override attribute assignment to register parameters and submodules.
    # @details Ensures that assigned Parameter or Module objects are tracked in
    # dedicated dictionaries for recursive optimization and state management.
    ##
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value) 
        if isinstance(value, Parameter):
            self._parameters[name] = value
        if isinstance(value, Module):
            self._modules[name] = value

    ##
    # @brief Invoke the forward pass of the module.
    # @details Redirects calls to the forward() method, enabling modules to be
    # used like functions during model execution.
    ##
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    ##
    # @brief Switch the module to training mode.
    # @details Sets the training flag and recursively updates all submodules,
    # enabling layers such as dropout to change behavior accordingly.
    ##
    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()

    ##
    # @brief Switch the module to evaluation mode.
    # @details Disables training-specific behaviors and recursively updates all
    # child modules to inference mode.
    ##
    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()

    ##
    # @brief Return all parameters contained in the module hierarchy.
    # @details Collects parameters from the current module and all nested
    # submodules, enabling optimizers to handle entire model trees.
    ##
    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for p in self._parameters.values():
            params.append(p)
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    ##
    # @brief Register a new parameter under a given name.
    # @details Adds the parameter to the internal registry and attaches it to the
    # module to ensure proper gradient tracking and serialization.
    ##
    def add_parameter(self, name: str, value: Parameter) -> None:
        self._parameters[name] = value
        setattr(self, name, value)

    ##
    # @brief Register a new submodule under a given name.
    # @details Stores the submodule for hierarchical traversal and attaches it to
    # the module to support nested architectures.
    ##
    def add_module(self, name: str, module: Module) -> None:
        self._modules[name] = module
        setattr(self, name, module)

    ##
    # @brief Forward computation for the module.
    # @details Must be implemented by subclasses. Defines how the module processes
    # input tensors.
    ##
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    ##
    # @brief Export all parameters as a flat state dictionary.
    # @details Recursively traverses the module hierarchy, assigning hierarchical
    # keys like 'layer.weight' to parameters, enabling clean saving and loading.
    ##
    def state_dict(self) -> dict:
        state = {}
        for name, param in self._parameters.items():
            state[name] = param.data
        for name, module in self._modules.items():
            for sub_name, sub_param in module.state_dict().items():
                state[f"{name}.{sub_name}"] = sub_param
        return state

    ##
    # @brief Load parameters into the module from a state dictionary.
    # @details Resolves hierarchical keys, locates target parameters, checks types,
    # and restores each value into the correct module and parameter object.
    ##
    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        model_keys = self.state_dict().keys()
        missing_keys = set(model_keys) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_keys)
        if strict:
            if missing_keys:
                raise ValueError(
                    f"Missing keys detected in state_dict: {missing_keys}. "
                    "The state_dict is incomplete."
                )
            if unexpected_keys:
                raise ValueError(
                    f"Unexpected keys detected in state_dict: {unexpected_keys}. "
                    "The state_dict contains parameters not present in the model."
                )
        for name, data in state_dict.items():
            if name in model_keys or not strict: 
                parts = name.split('.')
                target = self
                for part in parts[:-1]:
                    if hasattr(target, part):
                        target = getattr(target, part)
                    else:
                        raise ValueError(f"Attribute {part} not found in the model structure.")
                param_name = parts[-1]
                if hasattr(target, param_name):
                    param = getattr(target, param_name)
                    if isinstance(param, Parameter):
                        param.data = data
                    else:
                        raise TypeError(f"Attribute {name} is not a Parameter.")
                else:
                    raise ValueError(f"Parameter {param_name} not found in module.")

    def save( self, filepath: str, blueprint: Optional[ArchitectureBlueprint] = None, quantized: bool = False) -> None:
        state_dict = self.state_dict()
        metadata = {}
        np_arrays_to_save = {}
        for name, param in state_dict.items():
            if not isinstance(param, np.ndarray):
                raise TypeError(f"Parameter {name} cannot be serialized (expected ndarray).")
            if quantized:
                q_data, scale, zero_point = quantize_weights(param.astype(np.float32))
                np_arrays_to_save[f"{name}_data"] = q_data
                metadata[name] = {
                    "quantized": True,
                    "scale": scale,
                    "zero_point": zero_point,
                    "dtype": str(param.dtype),
                }
            else:
                np_arrays_to_save[f"{name}_data"] = param
                metadata[name] = {"quantized": False, "dtype": str(param.dtype)}
        temp_module = copy.deepcopy(self)
        temp_module._parameters = {}
        try:
            module_serialized = pickle.dumps(temp_module)
        except TypeError as e:
            raise TypeError(
                f"Serialization error (pickle): {e}. Check if all components are serializable."
            )
        np_arrays_to_save["module_instance"] = np.array(module_serialized, dtype=object)
        if blueprint is not None:
            blueprint_json = json.dumps(blueprint.to_dict())
            np_arrays_to_save["blueprint_params"] = np.array(blueprint_json, dtype=object)
        np_arrays_to_save["metadata"] = np.array(metadata, dtype=object)
        np.savez_compressed(filepath, **np_arrays_to_save)


    ##
    # @brief Restores a saved model instance (architecture + parameters) from a file.
    # @details Deserializes the architecture from the 'module_instance' key.
    ##
    @classmethod
    def load_model(cls, filepath: str, module_instance: Optional['Module'] = None) -> 'Module':
        try:
            archive = np.load(filepath, allow_pickle=True)
            if module_instance is None:
                if 'module_instance' not in archive:
                    raise ValueError("Model file incomplete: missing key 'module_instance'.")
                module_serialized_container = archive['module_instance']
                module_serialized = cls._extract_serialized_blob(
                    module_serialized_container.item()
                    if hasattr(module_serialized_container, "item")
                    else module_serialized_container
                )
                module_instance = pickle.loads(module_serialized)
            metadata_container = archive.get('metadata')
            if metadata_container is None:
                raise ValueError("Model file incomplete: missing key 'metadata'.")
            state = cls._extract_serialized_blob(
                metadata_container.item()
                if hasattr(metadata_container, "item")
                else metadata_container
            )
            if not isinstance(state, dict):
                raise ValueError("Corrupted metadata structure in model file.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Failed to load NPZ file state: {e}")
        restored_state_dict = {}
        for name, saved_data in state.items():
            if f"{name}_data" not in archive:
                raise ValueError(f"Model file missing data array for parameter '{name}'.")
            saved_array = archive[f"{name}_data"]
            if saved_data.get('quantized'):
                scale = float(saved_data.get('scale', 1.0))
                zero_point = float(saved_data.get('zero_point', 0.0))
                restored_data = dequantize_weights(saved_array, scale, zero_point)
            else:
                restored_data = saved_array
            target_dtype = saved_data.get('dtype')
            if target_dtype:
                restored_data = restored_data.astype(np.dtype(target_dtype))
            restored_state_dict[name] = restored_data
        module_instance.load_state_dict(restored_state_dict)
        return module_instance

    @classmethod
    def load_blueprint_from_npz(cls, filepath: str) -> ArchitectureBlueprint:
        try:
            archive = np.load(filepath, allow_pickle=True)
            if 'blueprint_params' not in archive:
                raise ValueError("Model file missing training parameters ('blueprint_params').")
            blueprint_container = archive['blueprint_params']
            blueprint_blob = cls._extract_serialized_blob(
                blueprint_container.item()
                if hasattr(blueprint_container, "item")
                else blueprint_container
            )
            if isinstance(blueprint_blob, (bytes, bytearray)):
                try:
                    blueprint_blob = blueprint_blob.decode("utf-8")
                except (AttributeError, UnicodeDecodeError):
                    pass
            if isinstance(blueprint_blob, str):
                try:
                    blueprint_data = json.loads(blueprint_blob)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error parsing blueprint JSON: {e}") from e
                return ArchitectureBlueprint.from_dict(blueprint_data)
            try:
                return pickle.loads(blueprint_blob)
            except Exception:
                raise ValueError("Blueprint format not supported or corrupted.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error deserializing Blueprint parameters: {e}")
