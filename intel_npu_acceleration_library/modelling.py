#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import intel_npu_acceleration_library as npu_lib
from functools import partialmethod
from typing import Type, Any, Tuple, Optional
import hashlib
import torch
import os


def get_cache_dir() -> str:
    """Get the model cache directory.

    Returns:
        str: path to the cache directory
    """
    return os.path.join("cache", "models")


def get_mangled_model_name(model_name: str, *args: Any, **kwargs: Any) -> str:
    """Mangle the model name with all the parameters.

    Args:
        model_name (str): model name or path
        args (Any): positional arguments
        kwargs (Any): keyword arguments

    Returns:
        str: mangled name
    """
    # append all input parameters and create a string
    arguments_str = f"{[str(arg) for arg in args] + [f'{str(key)}_{str(arg)}' for key, arg in kwargs.items()]}"
    arguments_str_hash = hashlib.sha256(arguments_str.encode("utf-8")).hexdigest()
    mangled_model_name = f"{model_name}_{arguments_str_hash}_{npu_lib.__version__}"
    return mangled_model_name.replace("\\", "_").replace("/", "_")


def get_model_path(model_name: str, *args: Any, **kwargs: Any) -> Tuple[str, str]:
    """Get the model path.

    Args:
        model_name (str): model name or path
        args (Any): positional arguments
        kwargs (Any): keyword arguments

    Returns:
        Tuple[str, str]: model directory and full path
    """
    cache_dir = get_cache_dir()
    mangled_model_name = get_mangled_model_name(model_name, *args, **kwargs)
    model_dir_path = os.path.join(cache_dir, mangled_model_name)
    model_path = os.path.join(model_dir_path, "model.pt")
    return model_dir_path, model_path


class NPUModel:
    """Base NPU model class."""

    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        dtype: torch.dtype = torch.float16,
        training: bool = False,
        transformers_class: Optional[Type] = None,
        export=True,
        *args: Any,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Template for the `from_pretrained` static method.

        Args:
            model_name_or_path (str): model name or path
            dtype (torch.dtype, optional): compilation dtype. Defaults to torch.float16.
            training (bool, optional): enable training. Defaults to False.
            transformers_class (Optional[Type], optional): base class to use. Must have a `from_pretrained` method. Defaults to None.
            export (bool, optional): enable the caching of the model. Defaults to True.
            args (Any): positional arguments
            kwargs (Any): keyword arguments

        Raises:
            RuntimeError: Invalid class

        Returns:
            torch.nn.Module: compiled mode
        """
        if transformers_class is None:
            raise RuntimeError(f"Invalid transformer class {type(transformers_class)}")
        # get the model cache dir and path from the name and arguments
        model_dir_path, model_path = get_model_path(
            model_name_or_path, dtype, training, *args, **kwargs
        )
        if os.path.isdir(model_dir_path) and os.path.isfile(model_path):
            # Model already exist so I can load it directly
            return torch.load(model_path)
        else:
            # Model does not exists, so I need to compile it first
            print(f"Compiling model {model_name_or_path} {dtype} for the NPU")
            model = transformers_class.from_pretrained(
                model_name_or_path, *args, **kwargs
            )
            model = npu_lib.compile(model, dtype, training)
            if export:
                print(f"Exporting model {model_name_or_path} to {model_dir_path}")
                os.makedirs(model_dir_path, exist_ok=True)
                torch.save(model, model_path)
            return model


class NPUAutoModel:
    """NPU wrapper for AutoModel.

    Attrs:
        from_pretrained: Load a pretrained model
    """

    from_pretrained = partialmethod(
        NPUModel.from_pretrained, transformers_class=AutoModel
    )


class NPUModelForCausalLM:
    """NPU wrapper for AutoModelForCausalLM.

    Attrs:
        from_pretrained: Load a pretrained model
    """

    from_pretrained = partialmethod(
        NPUModel.from_pretrained, transformers_class=AutoModelForCausalLM
    )


class NPUModelForSeq2SeqLM:
    """NPU wrapper for AutoModelForSeq2SeqLM.

    Attrs:
        from_pretrained: Load a pretrained model
    """

    from_pretrained = partialmethod(
        NPUModel.from_pretrained, transformers_class=AutoModelForSeq2SeqLM
    )
