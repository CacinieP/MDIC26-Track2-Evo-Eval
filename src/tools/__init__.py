# Tools package
from .chart_analyzer import ChartAnalyzer
from .crosspage_merger import CrossPageMerger
from .image_enhancer import ImageEnhancer
from .mineru_parser import MinerUParser
from .table_parser import TableParser


def create_tool_registry(config: dict | None = None) -> dict:
    """
    Instantiate all tools with configuration and return a name->tool mapping.

    The returned dict is suitable for passing to create_agent_graph(tool_registry=...).
    Includes built-in verifier and exporter tools used by the agent graph.
    """
    config = config or {}
    mineru_cfg = config.get("mineru", {})

    from src.agents.graph import _BuiltinVerifier, _BuiltinExporter

    return {
        "mineru_parser": MinerUParser(mineru_cfg),
        "table_parser": TableParser(),
        "chart_analyzer": ChartAnalyzer(),
        "image_enhancer": ImageEnhancer(),
        "cross_page_merger": CrossPageMerger(),
        "verifier": _BuiltinVerifier(),
        "exporter": _BuiltinExporter(),
    }


__all__ = [
    "ChartAnalyzer",
    "CrossPageMerger",
    "ImageEnhancer",
    "MinerUParser",
    "TableParser",
    "create_tool_registry",
]
