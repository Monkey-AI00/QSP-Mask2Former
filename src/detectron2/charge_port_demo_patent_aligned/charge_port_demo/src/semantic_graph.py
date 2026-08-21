from typing import Any

import numpy as np

from validation import validate_transform


class SemanticGraph:
    def __init__(self, graph_dict: dict[str, Any]):
        self.graph_dict = dict(graph_dict)
        self.relations = dict(graph_dict.get("relations", {}))
        if not self.graph_dict.get("root_node"):
            raise ValueError("semantic graph is missing a vehicle root node")

    def get_observable_nodes(self) -> list[str]:
        return list(self.graph_dict.get("observable_nodes", []))

    def get_port_relation(self, node_name: str) -> dict[str, Any]:
        return dict(self.relations.get(node_name, {}))

    def get_part_to_port_transform(self, part_name: str) -> np.ndarray:
        relation = self.relations.get(part_name)
        if relation is None or "part_to_port_transform" not in relation:
            raise KeyError(f"semantic graph has no part-to-port transform for {part_name}")
        return validate_transform(relation["part_to_port_transform"], f"{part_name} part_to_port_transform")

    def get_part_normal(self, part_name: str) -> np.ndarray:
        relation = self.relations.get(part_name)
        if relation is None or "part_normal" not in relation:
            raise KeyError(f"semantic graph has no part normal for {part_name}")
        normal = np.asarray(relation["part_normal"], dtype=float)
        length = float(np.linalg.norm(normal))
        if normal.shape != (3,) or not np.all(np.isfinite(normal)) or length <= 1e-12:
            raise ValueError(f"{part_name} part normal must be a finite non-zero xyz vector")
        return normal / length

    def to_dict(self) -> dict[str, Any]:
        return dict(self.graph_dict)
