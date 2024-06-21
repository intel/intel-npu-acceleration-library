#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from intel_npu_acceleration_library.backend import NNFactory, Tensor
from typing import MutableMapping, Mapping, Tuple, Sequence, Any
import numpy as np
import torch


def pt_to_np_dtype(torch_dtype: torch.dtype) -> np.dtype:
    """Convert a PyTorch dtype to a NumPy dtype.

    Args:
        torch_dtype (torch.dtype): The PyTorch dtype to convert.

    Raises:
        ValueError: If the PyTorch dtype is not supported.

    Returns:
        np.dtype: The NumPy dtype.
    """
    if torch_dtype == torch.float16:
        return np.float16
    elif torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    elif torch_dtype == torch.int8:
        return np.int8
    elif torch_dtype == torch.int16:
        return np.int16
    elif torch_dtype == torch.int32:
        return np.int32
    elif torch_dtype == torch.int64:
        return np.int64
    else:
        raise ValueError(f"Unsupported dtype {torch_dtype}")


def compute_input_signature(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str:
    """Compute the input signature of a function call.

    Args:
        args (Sequence[Any]): The positional arguments.
        kwargs (Mapping[str, Any]): The keyword arguments.

    Returns:
        str: The input signature.
    """
    signature = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            signature.append("_".join(str(dim) for dim in arg.shape))
            signature.append(str(arg.dtype))
        else:
            signature.append(str(arg))
    for k, arg in kwargs.items():
        if isinstance(arg, torch.Tensor):
            signature.append(str(k))
            signature.append("_".join(str(dim) for dim in arg.shape))
            signature.append(str(arg.dtype))
        else:
            signature.append(str(arg))
    return "_".join(signature)


def patch_parameters(module: torch.nn.Module, model: NNFactory, recurse: bool = False):
    """Patch the parameters of a PyTorch module with constants.

    Args:
        module (torch.nn.Module): The PyTorch module.
        model (NNFactory): The NNFactory instance.
        recurse (bool, optional): Recurse over all submodules. Defaults to False.
    """
    elements = list(module.named_parameters(recurse=recurse))
    for name, param in elements:
        del module._parameters[name]
        setattr(module, name, model.constant(param.data.detach().numpy()))

    for name, param in module.named_buffers(recurse=recurse):
        del module._buffers[name]
        setattr(module, name, model.constant(param.data.detach().numpy()))


def patch_modules(module: torch.nn.Module, model: NNFactory):
    """Patch the modules of a PyTorch module with constants.

    Args:
        module (torch.nn.Module): The PyTorch module.
        model (NNFactory): The NNFactory instance.
    """
    modules = list(module.named_children())
    for _, module in modules:
        if isinstance(module, NPUModule):
            module.npu_top_level_module = False
        patch_parameters(module, model)
        patch_modules(module, model)


class NPUModule(torch.nn.Module):
    """A PyTorch module that runs on the NPU."""

    def __init__(self) -> None:
        """Initialize the module."""
        super().__init__()
        self._nn_factory_cache: MutableMapping[str, Tuple[NNFactory, Tensor]] = {}
        self._npu_inference = False
        self.npu_top_level_module = True

    def factory_forward(self, *args: Any, **kwargs: Any):
        """Run the model using the factory.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """
        signature = compute_input_signature(args, kwargs)
        model, out = self._nn_factory_cache[signature]

        out_shape, out_dtype = out.shape, out.dtype

        tensor_args = [
            arg.detach().numpy() for arg in args if isinstance(arg, torch.Tensor)
        ]
        tensor_args += [
            arg.detach().numpy()
            for k, arg in kwargs.items()
            if isinstance(arg, torch.Tensor)
        ]
        out = model.run(*tensor_args, **kwargs)

        return torch.tensor(out, dtype=out_dtype.torch_dtype).reshape(out_shape)

    def create_model(
        self, args: Sequence[Any], kwargs: Mapping[str, Any]
    ) -> Tuple[NNFactory, Tensor]:
        """Create a model from the module.

        Args:
            args (Sequence[Any]): positional arguments
            kwargs (Mapping[str, Any]): keyword arguments

        Returns:
            Tuple[NNFactory, Tensor]: The model and the output tensor.
        """
        model = NNFactory()
        npu_args, npu_kwargs = [], {}
        for arg in args:
            if isinstance(arg, torch.Tensor):
                npu_args.append(model.parameter(arg.shape, pt_to_np_dtype(arg.dtype)))
            else:
                npu_args.append(arg)

        for k, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                npu_kwargs[k] = model.parameter(arg.shape, pt_to_np_dtype(arg.dtype))
            else:
                npu_kwargs[k] = arg

        patch_modules(self, model)
        patch_parameters(self, model)

        out = self.forward(*npu_args, **npu_kwargs)
        model.compile(out)
        return model, out

    def _call_impl(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            Any: The output of the module.
        """
        if self._npu_inference and self.npu_top_level_module:

            signature = compute_input_signature(args, kwargs)
            if signature not in self._nn_factory_cache:
                self._nn_factory_cache[signature] = self.create_model(args, kwargs)

            # Run the model by replacing the forward method with the factory_forward
            old_forward = self.forward
            self.forward = self.factory_forward  # type: ignore
            out = super()._call_impl(*args, **kwargs)

            # Restore the original forward method
            self.forward = old_forward  # type: ignore

            return out
        else:
            return super()._call_impl(*args, **kwargs)

    def to(self, *args, **kwargs):
        """Move the module to a device or to a different dtype.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Returns:
            torch.Tensor: The output tensor.
        """
        device = kwargs.get("device", None)
        args = list(args)
        if device is None:
            for idx, arg in enumerate(args):
                if isinstance(arg, str) and arg.lower() in ["npu"]:
                    device = "npu"
                    args[idx] = "cpu"
        else:
            kwargs["device"] = "cpu"

        if device.lower() == "npu":
            self._npu_inference = True

        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Run the forward pass of the module.

        Args:
            args (Any): The positional arguments.
            kwargs (Any): The keyword arguments.

        Raises:
            NotImplementedError: If the forward method is not implemented.

        Returns:
            torch.Tensor: The output tensor.
        """
        raise NotImplementedError
        return torch.empty(0)
