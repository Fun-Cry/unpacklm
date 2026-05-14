"""
unpack.viz - Visualization for circuit paths.

Usage:
    from unpack.viz import CircuitGraph
    
    graph = CircuitGraph.from_tracer(tracer)
    graph.tokens = result.tokens
    graph.add_paths(result.paths[:5])
    graph.save_svg("circuit.svg")
    
    # For web integration:
    data = graph.to_dict()  # JSON-safe dict
"""

from unpack.viz.graph import CircuitGraph, VisPath, Hop

__all__ = ["CircuitGraph", "VisPath", "Hop"]
