#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
from typing import Dict, List, Any
import torch.nn as nn
import torch.fx as fx
import operator
import torch


def delattr_recursively(module: nn.Module, target: str):
    """Delete attribute recursively by name in a torch.nn.Module.

    Args:
        module (nn.Module): the nn.Module
        target (str): the attribute you want to delete
    """
    *root, name = target.rsplit(".", 1)
    if root:
        root = root[0].split(".")
        delattr_recursively(getattr(module, root[0]), ".".join(root[1:] + [name]))
    else:
        delattr(module, target)


def fuse_linear_layers(
    model: nn.Module,
    modules: Dict[str, nn.Linear],
    targets: List[str],
    fused_layer_name: str,
) -> None:
    """Fuse two linear layers and append them to the nn Module.

    Args:
        model (nn.Module): Origianl nn.Module object
        modules (Dict[nn.Linear]): a dictiorany of node name: linear layer
        targets (List[str]): list of layer node names
        fused_layer_name (str): fused layer name

    Raises:
        ValueError: All linear layers must be of type nn.Linear and must have the same input dimension

    """
    # Get the attributes
    layers = [modules[name] for name in targets]

    in_features = list({layer.in_features for layer in layers})

    # ensure both linear layers have the same input dimensions and are not already fused
    if not all(isinstance(layer, nn.Linear) for layer in layers):
        raise ValueError("All linear layers must be of type nn.Linear")
    if len(in_features) != 1:
        raise ValueError(
            f"All linear layers must have the same input dimensions. Instead found: {in_features}"
        )

    # Create the new fused linear layer
    new_out_features = sum([layer.out_features for layer in layers])
    has_bias = any(layer.bias is not None for layer in layers)
    fused_layer = nn.Linear(in_features[0], new_out_features, bias=has_bias)

    # Concatenate the weights and biases
    with torch.no_grad():
        start, stop = 0, 0
        for layer in layers:
            stop += layer.out_features
            fused_layer.weight[start:stop, :] = layer.weight

            if has_bias:
                if layer.bias is not None:
                    fused_layer.bias[start:stop] = layer.bias
                else:
                    fused_layer.bias[start:stop] = torch.zeros_like(
                        fused_layer.bias[start:stop]
                    )
            start = stop

    # Replace the two layers in the original model with the new fused layer
    setattr(model, fused_layer_name, fused_layer)
    for layer_name in targets:
        delattr_recursively(model, layer_name)


def horizontal_fusion_linear(model: torch.nn.Module) -> torch.nn.Module:
    """Fuze horizontally two or more linear layers that share the same origin. This will increase NPU hw utilization.

    Args:
        model (torch.nn.Module): The original nn.Module

    Returns:
        torch.nn.Module: optimize nn.Module where parallel linear operations has been fused into a single bigger one
    """
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    # new_graph = copy.deepcopy(fx_model.graph)

    def node_condition(node: Any) -> bool:
        """Return true if the node is a module and is nn.Linear.

        Args:
            node (Any): A torch fx node

        Returns:
            bool: return condition
        """
        return node.op == "call_module" and isinstance(modules[node.target], nn.Linear)

    # First, find all node with a linear layer
    linear_nodes = [node for node in fx_model.graph.nodes if node_condition(node)]

    # Group the linear layers by input node
    linear_nodes_parents: Dict[str, List[Any]] = {}
    for node in linear_nodes:
        linear_nodes_parents.setdefault(node.args[0], []).append(node)

    # Get the ones with size > 1
    fused_modules = [
        (source, modules)
        for source, modules in linear_nodes_parents.items()
        if len(modules) > 1
    ]

    for source, layers in fused_modules:
        fused_layer_name = "fused_" + "_".join(node.target for node in layers)
        fused_layer_name = fused_layer_name.replace(".", "_")
        fuse_linear_layers(
            fx_model, modules, [layer.target for layer in layers], fused_layer_name
        )
        with fx_model.graph.inserting_after(source):
            fused_node = fx_model.graph.call_module(fused_layer_name, (source,))

        with fx_model.graph.inserting_after(fused_node):

            start, stop = 0, 0
            for layer in layers:
                stop += modules[layer.target].out_features

                layer_slice = fx_model.graph.call_function(
                    operator.getitem,
                    args=(
                        fused_node,
                        (
                            Ellipsis,
                            slice(start, stop, None),
                        ),
                    ),
                    kwargs={},
                )
                layer.replace_all_uses_with(layer_slice)
                fx_model.graph.erase_node(layer)
                start = stop

    fx_model.graph.lint()
    fx_model.recompile()

    return fx_model
