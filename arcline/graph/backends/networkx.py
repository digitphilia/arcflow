# -*- encoding: utf-8 -*-

"""
Concrete Implementation of :class:`AbstractGraph` using NetworkX API
--------------------------------------------------------------------

This module provides :class:`NetworkXGraph`, the :mod:`networkx`-backed
concrete implementation of :class:`arcline.graph.base.AbstractGraph`.
It is the recommended choice for development workflows, exploratory
notebooks, and small subgraphs where the pure-Python overhead of
:mod:`networkx` (typicall ≤ 1M edges) is tolerable.

The practical scale ceiling for this backend is on the order of one
million edges; past that point per-operation overhead from the
Python-level adjacency structure begins to dominate runtime. For the
full supply-chain network, prefer the :mod:`igraph` backend which
sits on a C-level graph engine and offers one to two orders of
magnitude better throughput on bulk operations.

The wrapped backend is always a :class:`networkx.MultiDiGraph` but any
other instances of :class:`networkx.Graph` is also supported, but may
not be valid when designing for a supply chain optimization problem.
The supply-chain framework treats lanes as directed (asymmetric lead
time and cost) and as multi-edges so that parallel lanes between the
same endpoint pair can be modeled independently (different carriers,
contracts, transportation modes). Vertices in the underlying graph
are keyed by :attr:`AbstractNode.hashKey` and parallel edges between
the same endpoint pair are disambiguated by :attr:`AbstractEdge.hashKey`
passed to NetworkX as the edge ``key``.
"""

from collections.abc import Iterator
from typing import Dict, List, Optional

import networkx

from arcline.graph.base import AbstractGraph, AbstractNode, AbstractEdge

class NetworkXGraph(AbstractGraph):
    """
    NetworkX-backed concrete implementation of :class:`AbstractGraph`.
    Suitable for development workflows and small subgraphs where the
    pure-Python overhead of :mod:`networkx` is tolerable. For the full
    network (thousands of nodes, millions of edges), use the
    :mod:`igraph` backend instead.

    The wrapped backend is always a :class:`networkx.MultiDiGraph`. On
    construction the node and edge object lists supplied to
    :meth:`__init__` are materialised into the underlying graph via
    :meth:`buildGraph`; vertices use :attr:`AbstractNode.hashKey` as
    their NetworkX identifier and parallel edges between the same
    endpoint pair are disambiguated by :attr:`AbstractEdge.hashKey`
    passed to NetworkX as the per-edge ``key``.

    Internally a hashKey → ``AbstractNode`` lookup is maintained so
    that :meth:`neighbors`, :meth:`predecessors`, and :meth:`successors`
    can return typed :class:`AbstractNode` objects rather than the raw
    string identifiers that NetworkX stores natively.
    """

    def __init__(
        self,
        nodes : List[AbstractNode],
        edges : List[AbstractEdge],
        G : Optional[networkx.Graph] = None,
        **kwargs
    ) -> None:
        """
        The node and edge object lists are stored on the base class as
        :attr:`nodes` / :attr:`edges` and become the canonical source
        of truth for the model. The underlying NetworkX graph is
        either supplied via ``G`` (useful when wrapping a graph loaded
        from disk) or created fresh as an empty :class:`networkx.Graph`.
        When ``autoBuild`` is ``True`` (the default) :meth:`buildGraph`
        is invoked immediately so the instance is ready for traversal
        queries on return.

        :type  nodes: List[AbstractNode]
        :param nodes: The node objects that compose the graph. Each
            node must be an instance of :class:`AbstractNode` class or
            any derived sub-nodes with pre-built configurations.

        :type  edges: List[AbstractEdge]
        :param edges: The edge objects that compose the graph. Each
            edge holds direct references to its source and destination
            nodes and a unique :attr:`AbstractEdge.hashKey` that
            disambiguates parallel edges in the multi-graph.

        :type  G: Optional[networkx.Graph]
        :param G: Optional pre-built :mod:`networkx` graph to wrap.
            Defaults to a fresh empty :class:`networkx.MultiDiGraph`.

        **Keyword Arguments**

            * **name** (*str*): Model name for logging and auditing,
                defaults to the concrete class name.

            * **autoBuild** (*bool*): When ``True`` (default), the
                :meth:`buildGraph` is called at the end of construction
                to populate ``G`` from the node and edge lists. Pass
                ``False`` to defer materialisation (e.g. when ``G``
                is already populated).
        """

        super().__init__(
            G = G or networkx.MultiDiGraph(),
            nodes = nodes, edges = edges, name = kwargs.get("name", None)
        )

        # ? auto build the graph; else pass developed graph
        if kwargs.get("autoBuild", True):
            self.buildGraph()


    def buildGraph(self) -> bool:
        """
        Build the graph consisting of :attr:`nodes` and :attr:`edges`
        with underlying properties using :mod:`pydantic` models data
        payload which are added as the vertex attributes. For an edge
        the ``hashKey`` is set as the key attribute for multi-graph
        while all other attributes are added as attributes.

        :rtype:   bool
        :returns: The boolean flag is positional, either returns
            ``True`` if the build is succesful.
        """

        self.G.add_nodes_from([
            (node.hashKey, node.model_dump(exclude = {"hashKey"}))
            for node in self.nodes
        ])
        
        for edge in self.edges:
            attrs = edge.model_dump(exclude = {
                "hashKey", "srcNode", "dstNode"
            })

            self.G.add_edge(
                edge.srcNode.hashKey, edge.dstNode.hashKey,
                key = edge.hashKey, **attrs
            )

        return True
