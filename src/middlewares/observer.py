from dataclasses import dataclass
from typing import Optional, Protocol, List
import numpy as np


class SystemObserver(Protocol):
    """Protocolo para observadores del sistema"""

    def on_tensor_product(self, context: dict) -> None: ...
    def on_partition(self, context: dict) -> None: ...
    def on_condition(self, context: dict) -> None: ...


@dataclass
class DebugObserver:
    """Observer que registra información detallada para debugging"""

    def on_tensor_product(self, context: dict) -> None:
        print("\nTensor Product Debug:")
        print(f"Cubes: {context['n_cubes']}")
        print(f"Active dims: {context['active_dims']}")
        for cube in context["cubes"]:
            print(f"\nCube {cube.indices}:")
            print(f"Dimensions: {cube.dims}")
            print(f"Data shape: {cube.data.shape}")

    def on_partition(self, context: dict) -> None:
        print("\nPartition Debug:")
        print(f"Future prim: {context['future_prim']}")
        print(f"Present prim: {context['present_prim']}")

    def on_condition(self, context: dict) -> None:
        print("\nCondition Debug:")
        print(f"Indices: {context['indices']}")
        print(f"Valid indices: {context['valid_indices']}")
