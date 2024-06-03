#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from typing import Optional, List, Union, Any, Dict, Tuple, Iterable
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
import numpy as np
import intel_npu_acceleration_library
import ctypes
import os


def adapt_weight(w: np.ndarray) -> np.ndarray:
    """Adapt the weights to run on the NPU.

    Args:
        w (np.ndarray): weights array

    Returns:
        np.ndarray: The adapted array
    """
    if len(w.shape) == 1:
        w_adapted = w.reshape((1, -1))
        return w_adapted, w_adapted.shape
    elif len(w.shape) == 2:
        return w, w.shape
    else:
        w_adapted = w.flatten().reshape((1, -1))
        return w_adapted, w_adapted.shape


class BaseNPUBackend:
    """A base class that represent a abstract Matrix-Matrix operation on the NPU."""

    def __init__(self, profile: Optional[bool] = False) -> None:
        """Initialize class profiling.

        Args:
            profile (Optional[bool], optional): Enable/Disable NPU profiling. Defaults to False.
        """
        if profile:
            os.environ["NPU_PRINT_PROFILING"] = "JSON"
            os.environ["NPU_PROFILING_OUTPUT_FILE"] = "profiling.json"
            os.environ["NPU_PROFILING_VERBOSITY"] = "HIGH"
        self._mm: Any = None

    def __del__(self):
        """Deallocate and free the class from the library."""
        if (
            hasattr(self, "_mm")
            and intel_npu_acceleration_library
            and hasattr(backend_lib, "destroyNNFactory")
        ):
            backend_lib.destroyNNFactory(self._mm)

    def save(self, path: str):
        """Save the Openvino model.

        Args:
            path (str): the model save path
        """
        backend_lib.saveModel(self._mm, ctypes.c_char_p(path.encode()))

    def saveCompiledModel(self, path: str):
        """Save the compiled model.

        Args:
            path (str): the compiled model save path
        """
        backend_lib.saveCompiledModel(self._mm, ctypes.c_char_p(path.encode()))


class BaseNPUBackendWithPrefetch(BaseNPUBackend):
    """A base class that represent a abstract Matrix-Matrix operation on the NPU.

    Linear type classes employ an algorithm to optimize weights prefetching
    """

    def __init__(self, profile: bool):
        """Initialize class.

        Args:
            profile (bool): Enable/Disable NPU profiling.
        """
        super().__init__(profile)
        self.wt_order: List[str] = []
        self.wt_map: Dict[str, ctypes._Pointer] = {}
        self.loaded: Optional[str] = None

    def load_wt_fn(self, module, parameters):
        """Load asyncronously the parameter into the NPU.

        Args:
            module: the NPU backend module
            parameters: the weights parameter class
        """
        backend_lib.setNNFactoryWeights(module, parameters)

    def create_parameters(
        self, weights: Iterable[Union[np.ndarray, Tuple[np.ndarray, ...]]]
    ) -> ctypes._Pointer:
        """Create an operation parameter from a list of weights.

        Args:
            weights (Iterable[Union[np.ndarray, Tuple[np.ndarray, ...]]]): Operation parameters

        Raises:
            RuntimeError: Quantized weights needs to be in int8 format
            ValueError: Invalid dtype for scale

        Returns:
            ctypes._Pointer: an instance to the Parameters object
        """
        param = backend_lib.createParameters()
        if isinstance(weights, (list, tuple)):
            for weight in weights:
                if isinstance(weight, (list, tuple)):
                    # int8: data and scale
                    data, scale = weight
                    if data.dtype not in [np.int8, np.uint8]:
                        raise RuntimeError(
                            "Quantized weights needs to be in int8 or uint8 format"
                        )
                    adapted_weights, shape = adapt_weight(data)
                    if scale.dtype == np.float16:
                        # Mixed precision matmul
                        if data.dtype == np.int8:
                            backend_lib.addIntParameter(
                                param,
                                adapted_weights,
                                adapt_weight(scale)[0],
                                *shape,
                            )
                        else:
                            backend_lib.addInt4Parameter(
                                param,
                                adapted_weights,
                                adapt_weight(scale)[0],
                                *shape,
                            )
                    elif scale.dtype == np.float32:
                        # FP16 matmul with CPU conversion
                        backend_lib.addIntParameterConversion(
                            param,
                            adapted_weights,
                            adapt_weight(scale)[0],
                            *shape,
                        )
                    else:
                        raise ValueError(f"Invalid dtype for scale: {scale.dtype}")
                else:
                    adapted_weights, shape = adapt_weight(weight)
                    backend_lib.addFloatParameter(param, adapted_weights, *shape)
        elif isinstance(weights, np.ndarray):
            adapted_weights, shape = adapt_weight(weights)
            backend_lib.addFloatParameter(param, adapted_weights, *shape)
        return param

    def add_to_map(
        self, wt_hash: str, weights: Iterable[Union[np.ndarray, Tuple[np.ndarray, ...]]]
    ):
        """Add an operation parameters to the operation hash:parameter map.

        Args:
            wt_hash (str): operation hash
            weights (Iterable[Union[np.ndarray, Tuple[np.ndarray, ...]]]): Operation parameters
        """
        self.wt_map[wt_hash] = self.create_parameters(weights)

        self.wt_order.append(wt_hash)

    def setWeights(
        self, wt_hash: Optional[str], *args: Union[np.ndarray, Tuple[np.ndarray, ...]]
    ) -> bool:
        """Set the operation weights in the NPU.

        Args:
            wt_hash (str): operation hash. If set to None force the load of the weights
            args (Union[np.ndarray, Tuple[np.ndarray, ...]]): Variable length weights list. Can be a np array or a tuple of weight, scale in case of quantized tensors

        Returns:
            bool: Return True if the op parameters are already in the op map
        """
        if wt_hash is None:
            self.load_wt_fn(self._mm, self.create_parameters(args))
            return False
        in_wt_map = wt_hash in self.wt_map.keys()
        if not wt_hash == self.loaded:
            if not in_wt_map:
                self.add_to_map(wt_hash, args)
            self.load_wt_fn(self._mm, self.wt_map[wt_hash])
            self.loaded = wt_hash
            return in_wt_map
        return in_wt_map

    def prefetchWeights(self):
        """Prefetch next operation weights."""
        next_wt_idx = (self.wt_order.index(self.loaded) + 1) % len(self.wt_order)
        wt_hash = self.wt_order[next_wt_idx]
        if not wt_hash == self.loaded:
            self.load_wt_fn(self._mm, self.wt_map[wt_hash])
            self.loaded = wt_hash

    def __del__(self):
        """Deallocate and free the class from the library."""
        super(BaseNPUBackendWithPrefetch, self).__del__()
        for par in self.wt_map.values():
            if intel_npu_acceleration_library and hasattr(
                backend_lib, "destroyParameters"
            ):
                backend_lib.destroyParameters(par)
