# -*- encoding: utf-8 -*-

"""
Abstract Supply Chain Network/Graph Base with Module Agnostic Design
--------------------------------------------------------------------

A supply chain graph at the scale of thousands of nodes and millions
of edges sits at the boundary between an OLAP analytics workload and
a graph-traversal workload. NetworkX alone collapses past ~1M edges
(pure-Python overhead), so the package is designed around a backend
abstraction: a uniform ``AbstractGraph`` API with concrete
implementations for :mod:`networkx` (developer ergonomics, small
subgraphs) and :mod:`igraph` (C-level performance for full network
compute). The base defines the fundamental components of the graph
which is agnostic and used in the module.
"""

from arcline.graph.base.nodes import AbstractNode
from arcline.graph.base.edges import AbstractEdge
from arcline.graph.base.graph import AbstractGraph

__all__ = [
    "AbstractNode", "AbstractEdge", "AbstractGraph"
]
