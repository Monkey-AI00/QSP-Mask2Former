from typing import Any


class SemanticGraph:
    def __init__(self, graph_dict: dict[str, Any]):
        self.graph_dict = graph_dict
        self.relations = graph_dict.get("relations", {})

    def get_observable_nodes(self) -> list[str]:
        return list(self.graph_dict.get("observable_nodes", []))

    def get_port_relation(self, node_name: str) -> dict[str, Any]:
        return dict(self.relations.get(node_name, {}))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.graph_dict)
