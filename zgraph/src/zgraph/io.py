import dataclasses
import json
import zipfile
import importlib
import numpy as np
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import safetensors.numpy as stnp
from typing import Any, Dict, List, Tuple

from zgraph.core.base import ZGraphNode, Ensemble

def serialize_tree(root_node: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Converts a ZGraph tree into a Flat Graph JSON registry and a dictionary of tensor weights.
    """
    nodes_registry = {}
    tensor_registry = {}
    node_id_map = {}
    
    def process_object(obj: Any) -> Any:
        if isinstance(obj, eqx.Module):
            obj_id = id(obj)
            if obj_id in node_id_map:
                return {"__node_ref": node_id_map[obj_id]}
                
            node_ref_id = f"node_{len(node_id_map)}"
            node_id_map[obj_id] = node_ref_id
            
            state = {}
            for field in dataclasses.fields(obj):
                val = getattr(obj, field.name)
                state[field.name] = process_object(val)
                
            cls = obj.__class__
            class_path = f"{cls.__module__}.{cls.__name__}"
            
            nodes_registry[node_ref_id] = {
                "class_path": class_path,
                "state": state
            }
            return {"__node_ref": node_ref_id}
            
        elif isinstance(obj, (jax.Array, np.ndarray)):
            tensor_id = f"tensor_{len(tensor_registry)}"
            arr = np.asarray(obj)
            tensor_registry[tensor_id] = arr
            return {"__tensor_id": tensor_id, "shape": list(arr.shape), "dtype": str(arr.dtype)}
            
        elif isinstance(obj, (list, tuple)):
            processed = [process_object(item) for item in obj]
            if isinstance(obj, tuple):
                return {"__tuple": processed}
            return processed
            
        elif isinstance(obj, dict):
            return {k: process_object(v) for k, v in obj.items()}
            
        elif callable(obj):
            return {"__callable": f"{obj.__module__}.{obj.__name__}"}
            
        else:
            # Primitives: int, float, str, bool, None
            return obj
            
    root_ref = process_object(root_node)
    
    topology = {
        "root": root_ref["__node_ref"],
        "nodes": nodes_registry
    }
    
    return topology, tensor_registry


def _resolve_callable(path: str) -> Any:
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def deserialize_tree(topology: Dict[str, Any], tensor_registry: Dict[str, np.ndarray]) -> Any:
    """
    Rebuilds a ZGraph tree from a Flat Graph JSON registry and a dictionary of tensor weights.
    """
    nodes_registry = topology["nodes"]
    
    # Step 1: Instantiate all skeletons
    skeletons = {}
    for node_id, node_config in nodes_registry.items():
        class_path = node_config["class_path"]
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        
        # Bypass __init__
        skeleton = object.__new__(cls)
        skeletons[node_id] = skeleton
        
    # Step 2: Link state and build arrays
    def resolve_state(state: Any) -> Any:
        if isinstance(state, dict):
            if "__node_ref" in state:
                return skeletons[state["__node_ref"]]
            elif "__tensor_id" in state:
                arr = tensor_registry[state["__tensor_id"]]
                return jnp.array(arr)
            elif "__tuple" in state:
                return tuple([resolve_state(item) for item in state["__tuple"]])
            elif "__callable" in state:
                return _resolve_callable(state["__callable"])
            else:
                return {k: resolve_state(v) for k, v in state.items()}
        elif isinstance(state, list):
            return [resolve_state(item) for item in state]
        else:
            return state

    for node_id, node_config in nodes_registry.items():
        skeleton = skeletons[node_id]
        state = node_config["state"]
        resolved_state = resolve_state(state)
        for k, v in resolved_state.items():
            object.__setattr__(skeleton, k, v)
            
    return skeletons[topology["root"]]


def save(tree: ZGraphNode, filepath: str, library_dir: str = None):
    """
    Saves a ZGraph PyTree to a .zg hybrid archive.
    """
    if library_dir is not None and not os.path.isabs(filepath):
        filepath = os.path.join(library_dir, filepath)

    if not filepath.endswith(".zg"):
        filepath += ".zg"
        
    topology, tensor_registry = serialize_tree(tree)
    topology_json = json.dumps(topology, indent=2)
    tensor_bytes = stnp.save(tensor_registry)
    
    with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("topology.json", topology_json)
        zf.writestr("weights.safetensors", tensor_bytes)


def load(filepath: str, library_dir: str = None) -> ZGraphNode:
    """
    Loads a ZGraph PyTree from a .zg hybrid archive.
    """
    import os
    if library_dir is not None and not os.path.isabs(filepath):
        filepath = os.path.join(library_dir, filepath)

    # Pseudo-URI handling for reference models
    if ":" in filepath and not filepath.startswith("http"):
        app, model_name = filepath.split(":", 1)
        # Assuming the library is in the working directory under {app}/library/{model_name}.zg
        filepath = os.path.join(app, "library", f"{model_name}.zg")
        
    if not filepath.endswith(".zg"):
        filepath += ".zg"
        
    with zipfile.ZipFile(filepath, "r") as zf:
        topology_json = zf.read("topology.json").decode("utf-8")
        topology = json.loads(topology_json)
        
        tensor_bytes = zf.read("weights.safetensors")
        tensor_registry = stnp.load(tensor_bytes)
        
    return deserialize_tree(topology, tensor_registry)
