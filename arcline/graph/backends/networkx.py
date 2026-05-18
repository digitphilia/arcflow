# -*- encoding: utf-8 -*-

"""
NetworkX-Backed Concrete Implementation of :class:`AbstractGraph`
-----------------------------------------------------------------

This module provides :class:`NetworkXGraph`, the :mod:`networkx`-backed
concrete implementation of :class:`arcline.graph.base.AbstractGraph`.
It is the recommended choice for development workflows, exploratory
notebooks, and small subgraphs where the pure-Python overhead of
:mod:`networkx` is tolerable.

The practical scale ceiling for this backend is on the order of one
million edges; past that point per-operation overhead from the
Python-level adjacency structure begins to dominate runtime. For the
full supply-chain network, prefer the :mod:`igraph` backend which
sits on a C-level graph engine and offers one to two orders of
magnitude better throughput on bulk operations.

The wrapped backend is always a :class:`networkx.MultiDiGraph`: the
supply-chain framework treats lanes as directed (asymmetric lead time
and cost) and as multi-edges so that parallel lanes between the same
endpoint pair can be modeled independently (different carriers,
contracts, transportation modes).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import Any, Self, TypeVar

import networkx as nx
from networkx import MultiDiGraph
from pydantic import BaseModel

from arcline.graph.base import (
    AbstractGraph, NodeID, EdgeKey, NodeData, EdgeData,
)


N = TypeVar("N", bound=BaseModel)
E = TypeVar("E", bound=BaseModel)


class NetworkXGraph(AbstractGraph[N, E]):
    """
    NetworkX-backed implementation of :class:`AbstractGraph`.

    Suitable for development workflows and small subgraphs where the
    pure-Python overhead of :mod:`networkx` is tolerable. For the full
    network (thousands of nodes, millions of edges), use the igraph
    backend instead.

    The concrete node-payload type ``N`` and edge-payload type ``E``
    must be Pydantic ``BaseModel`` subclasses. The classes themselves
    are passed at construction time (``nodeModel``, ``edgeModel``)
    because Python erases generic parameters at runtime and we still
    need a handle to call :meth:`pydantic.BaseModel.model_validate`
    and :meth:`pydantic.BaseModel.model_dump` on every payload
    boundary.

    The wrapped backend is always a :class:`networkx.MultiDiGraph`,
    consistent with the framework's directed multi-graph stance.
    """

    def __init__(
        self,
        nodeModel : type[N],
        edgeModel : type[E],
        *,
        G    : MultiDiGraph | None = None,
        name : str | None = None,
    ) -> None:
        """
        Construct a new NetworkX-backed graph.

        ``nodeModel`` and ``edgeModel`` (the Pydantic classes used for
        node and edge payloads) must be passed explicitly because
        Python erases generic type parameters at runtime and the
        instance still needs a stable handle to call
        :meth:`pydantic.BaseModel.model_validate` and
        :meth:`pydantic.BaseModel.model_dump` on every payload
        boundary.

        ``G`` is an optional escape hatch to wrap an existing
        :class:`networkx.MultiDiGraph` (loaded from disk, produced by
        another routine, etc.); if omitted a fresh empty graph is
        created.

        :type  nodeModel: type[N]
        :param nodeModel: Pydantic ``BaseModel`` subclass used as the
            typed node payload.

        :type  edgeModel: type[E]
        :param edgeModel: Pydantic ``BaseModel`` subclass used as the
            typed edge payload.

        :type  G: networkx.MultiDiGraph | None
        :param G: Optional pre-built :mod:`networkx` graph to wrap.
            Defaults to a new empty :class:`networkx.MultiDiGraph`.

        :type  name: str | None
        :param name: Optional model name for logging / auditing.
            Defaults to the class name.
        """
        super().__init__(G=G if G is not None else MultiDiGraph(), name=name)
        self._nodeModel : type[N] = nodeModel
        self._edgeModel : type[E] = edgeModel
        self._indexes   : dict[str, dict[Hashable, set[NodeID]]] = {}


    def addNode(
        self,
        nodeId : NodeID,
        data   : N | NodeData | None = None,
    ) -> None:
        """
        Insert a single node into the graph.

        ``data`` may be a typed payload of type ``N`` (a Pydantic
        model instance), a plain mapping with the same fields, or
        ``None`` for a bare node with no attributes. The payload is
        validated via :meth:`pydantic.BaseModel.model_validate` before
        being attached to the underlying graph.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier for the node.

        :type  data: N | NodeData | None
        :param data: Optional payload describing the node.

        :raises ValueError: If ``data`` fails Pydantic validation.
        """
        payload = self._validateNode(data)
        self.G.add_node(nodeId, **payload)
        self._updateIndexesOnAdd(nodeId, payload)


    def addNodes(
        self,
        nodes : Iterable[NodeID | tuple[NodeID, N | NodeData]],
    ) -> None:
        """
        Insert a batch of nodes in a single backend call.

        Each element of ``nodes`` is either a bare node identifier or
        a ``(nodeId, data)`` tuple. The whole batch is validated
        up-front, then handed to :meth:`networkx.Graph.add_nodes_from`
        as a single call so that adjacency-dictionary growth happens
        once rather than per-node.

        :type  nodes: Iterable[NodeID | tuple[NodeID, N | NodeData]]
        :param nodes: Heterogeneous iterable of identifiers with
            optional inline payloads.

        :raises ValueError: If any payload fails Pydantic validation.
        """
        prepared : list[tuple[NodeID, dict[str, Any]]] = []
        for item in nodes:
            if isinstance(item, tuple):
                nid, raw = item
            else:
                nid, raw = item, None
            prepared.append((nid, self._validateNode(raw)))
        self.G.add_nodes_from(prepared)
        for nid, payload in prepared:
            self._updateIndexesOnAdd(nid, payload)


    def removeNode(self, nodeId : NodeID) -> None:
        """
        Remove a node and all of its incident edges from the graph.

        Removal is cascading: every edge that touches ``nodeId`` is
        also deleted. Secondary indexes that reference the node are
        updated transparently before the underlying removal happens.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node to remove.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        if nodeId not in self.G:
            raise KeyError(nodeId)
        self._updateIndexesOnRemove(nodeId)
        self.G.remove_node(nodeId)


    def hasNode(self, nodeId : NodeID) -> bool:
        """
        Test whether a node is present in the graph.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier to test for membership.

        :rtype: bool
        :returns: ``True`` if ``nodeId`` is a node of the graph,
            ``False`` otherwise.
        """
        return nodeId in self.G


    def getNode(self, nodeId : NodeID) -> N:
        """
        Retrieve the typed payload attached to a node.

        The payload is reconstructed via
        :meth:`pydantic.BaseModel.model_validate` from the underlying
        attribute dictionary on each call; callers should not mutate
        the returned model and expect the change to round-trip into
        the graph.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node to fetch.

        :rtype: N
        :returns: The typed payload attached to ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        if nodeId not in self.G:
            raise KeyError(nodeId)
        return self._nodeModel.model_validate(self.G.nodes[nodeId])


    def addEdge(
        self,
        u    : NodeID,
        v    : NodeID,
        data : E | EdgeData | None = None,
    ) -> None:
        """
        Insert a single edge from ``u`` to ``v``.

        ``data`` may be a typed payload of type ``E``, a plain
        mapping, or ``None``. Because the wrapped graph is a
        :class:`networkx.MultiDiGraph`, repeated insertion between the
        same endpoint pair creates parallel edges rather than
        overwriting an existing one.

        :type  u: NodeID
        :param u: Source node identifier.

        :type  v: NodeID
        :param v: Target node identifier.

        :type  data: E | EdgeData | None
        :param data: Optional payload describing the edge.

        :raises KeyError: If either ``u`` or ``v`` is not present in
            the graph.
        :raises ValueError: If ``data`` fails Pydantic validation.
        """
        if u not in self.G:
            raise KeyError(u)
        if v not in self.G:
            raise KeyError(v)
        payload = self._validateEdge(data)
        self.G.add_edge(u, v, **payload)


    def addEdges(
        self,
        edges : Iterable[tuple[NodeID, NodeID] | tuple[NodeID, NodeID, E | EdgeData]],
    ) -> None:
        """
        Insert a batch of edges in a single backend call.

        Each element of ``edges`` is either a ``(u, v)`` endpoint pair
        or a ``(u, v, data)`` triple. The whole batch is validated
        up-front and then handed to
        :meth:`networkx.Graph.add_edges_from` as a single call.

        :type  edges: Iterable[tuple[NodeID, NodeID]
            | tuple[NodeID, NodeID, E | EdgeData]]
        :param edges: Heterogeneous iterable of endpoint pairs with
            optional inline payloads.

        :raises KeyError: If any endpoint is not present in the graph.
            Endpoint existence is validated for the whole batch before
            any edge is inserted, so the operation is atomic with
            respect to this check.
        :raises ValueError: If any element has the wrong arity or if
            any payload fails Pydantic validation.
        """
        prepared : list[tuple[NodeID, NodeID, dict[str, Any]]] = []
        for item in edges:
            if len(item) == 2:
                u, v = item
                raw  = None
            elif len(item) == 3:
                u, v, raw = item
            else:
                raise ValueError(f"edge tuple of wrong arity: {item!r}")
            if u not in self.G:
                raise KeyError(u)
            if v not in self.G:
                raise KeyError(v)
            prepared.append((u, v, self._validateEdge(raw)))
        self.G.add_edges_from(prepared)


    def removeEdge(self, u : NodeID, v : NodeID) -> None:
        """
        Remove the edge between ``u`` and ``v`` from the graph.

        Because the wrapped graph is a multigraph, this removes all
        parallel edges between the two endpoints; finer-grained
        removal (by edge key) is not exposed here and must be done
        through :attr:`G`.

        :type  u: NodeID
        :param u: Source node identifier.

        :type  v: NodeID
        :param v: Target node identifier.

        :raises KeyError: If no edge exists between ``u`` and ``v``.
        """
        if not self.G.has_edge(u, v):
            raise KeyError((u, v))
        while self.G.has_edge(u, v):
            self.G.remove_edge(u, v)


    def hasEdge(self, u : NodeID, v : NodeID) -> bool:
        """
        Test whether an edge exists from ``u`` to ``v``.

        :type  u: NodeID
        :param u: Source node identifier.

        :type  v: NodeID
        :param v: Target node identifier.

        :rtype: bool
        :returns: ``True`` if at least one edge connects ``u`` to
            ``v``, ``False`` otherwise.
        """
        return self.G.has_edge(u, v)


    def getEdge(self, u : NodeID, v : NodeID) -> E:
        """
        Retrieve the typed payload attached to an edge.

        Because the wrapped graph is a multigraph, the payload of an
        unspecified parallel edge is returned when more than one edge
        connects ``u`` to ``v``; callers that need to disambiguate
        must use :attr:`G`.

        :type  u: NodeID
        :param u: Source node identifier.

        :type  v: NodeID
        :param v: Target node identifier.

        :rtype: E
        :returns: The typed payload attached to the edge.

        :raises KeyError: If no edge exists between ``u`` and ``v``.
        """
        if not self.G.has_edge(u, v):
            raise KeyError((u, v))
        attrs = self.G.get_edge_data(u, v)
        attrs = attrs[next(iter(attrs))]
        return self._edgeModel.model_validate(attrs)


    def nodes(
        self,
        data : bool = False,
    ) -> Iterator[NodeID] | Iterator[tuple[NodeID, N]]:
        """
        Iterate over the nodes of the graph.

        The ``data`` flag is a discriminator on the return shape:

        * ``data=False`` (default) yields bare ``NodeID`` values.
        * ``data=True`` yields ``(nodeId, payload)`` tuples where
          ``payload`` is the typed node payload of type ``N``.

        :type  data: bool
        :param data: Whether to include the node payload alongside
            each identifier.

        :rtype: Iterator[NodeID] | Iterator[tuple[NodeID, N]]
        :returns: An iterator over node identifiers, optionally paired
            with their typed payloads.
        """
        if not data:
            yield from self.G.nodes()
            return
        for nid, attrs in self.G.nodes(data=True):
            yield nid, self._nodeModel.model_validate(attrs)


    def edges(
        self,
        data : bool = False,
    ) -> Iterator[EdgeKey] | Iterator[tuple[NodeID, NodeID, E]]:
        """
        Iterate over the edges of the graph.

        The ``data`` flag is a discriminator on the return shape:

        * ``data=False`` (default) yields ``(u, v)`` endpoint pairs.
        * ``data=True`` yields ``(u, v, payload)`` triples where
          ``payload`` is the typed edge payload of type ``E``.

        On the underlying multigraph each parallel edge is yielded
        independently.

        :type  data: bool
        :param data: Whether to include the edge payload alongside
            each endpoint pair.

        :rtype: Iterator[EdgeKey] | Iterator[tuple[NodeID, NodeID, E]]
        :returns: An iterator over edge endpoint pairs, optionally
            paired with their typed payloads.
        """
        if not data:
            for u, v, _ in self.G.edges(keys=True):
                yield u, v
            return
        for u, v, _, attrs in self.G.edges(keys=True, data=True):
            yield u, v, self._edgeModel.model_validate(attrs)


    def neighbors(self, nodeId : NodeID) -> Iterator[NodeID]:
        """
        Iterate over the out-neighbors of a node.

        Equivalent to :meth:`successors` on this directed backend.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            neighbors are requested.

        :rtype: Iterator[NodeID]
        :returns: An iterator over neighboring node identifiers.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        return iter(self.G.neighbors(nodeId))


    def predecessors(self, nodeId : NodeID) -> Iterator[NodeID]:
        """
        Iterate over the in-neighbors of a node.

        Yields every node ``u`` such that an edge ``u -> nodeId``
        exists in the wrapped :class:`networkx.MultiDiGraph`.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            in-neighbors are requested.

        :rtype: Iterator[NodeID]
        :returns: An iterator over predecessor node identifiers.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        return iter(self.G.predecessors(nodeId))


    def successors(self, nodeId : NodeID) -> Iterator[NodeID]:
        """
        Iterate over the out-neighbors of a node.

        Yields every node ``v`` such that an edge ``nodeId -> v``
        exists in the wrapped :class:`networkx.MultiDiGraph`.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            out-neighbors are requested.

        :rtype: Iterator[NodeID]
        :returns: An iterator over successor node identifiers.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        return iter(self.G.successors(nodeId))


    def degree(self, nodeId : NodeID) -> int:
        """
        Return the total degree of a node.

        Equal to :meth:`inDegree` ``+`` :meth:`outDegree`. On the
        underlying multigraph each parallel edge contributes
        independently.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose degree
            is requested.

        :rtype: int
        :returns: The total degree of ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        return int(self.G.degree(nodeId))


    def inDegree(self, nodeId : NodeID) -> int:
        """
        Return the in-degree of a node.

        Equal to the count of incoming edges; parallel incoming edges
        on the underlying multigraph contribute independently.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            in-degree is requested.

        :rtype: int
        :returns: The in-degree of ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        return int(self.G.in_degree(nodeId))


    def outDegree(self, nodeId : NodeID) -> int:
        """
        Return the out-degree of a node.

        Equal to the count of outgoing edges; parallel outgoing edges
        on the underlying multigraph contribute independently.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            out-degree is requested.

        :rtype: int
        :returns: The out-degree of ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        return int(self.G.out_degree(nodeId))


    def buildIndex(self, group : str) -> None:
        """
        Build or rebuild a secondary index over a node-payload field.

        Indexes accelerate group-wise slicing (:meth:`nodesInGroup`,
        :meth:`sliceByGroup`) from O(numNodes) to O(k) where k is the
        size of the slice. Re-calling :meth:`buildIndex` for a group
        that is already indexed rebuilds it from scratch.

        :type  group: str
        :param group: Name of the node-payload field to index on.

        :raises KeyError: If no node carries the requested field.
        """
        index : dict[Hashable, set[NodeID]] = {}
        seen = False
        for nodeId, attrs in self.G.nodes(data=True):
            if group in attrs:
                seen = True
                index.setdefault(attrs[group], set()).add(nodeId)
        if not seen:
            raise KeyError(group)
        self._indexes[group] = index


    def dropIndex(self, group : str) -> None:
        """
        Drop a previously built secondary index.

        After dropping, subsequent calls to :meth:`groups` and
        :meth:`nodesInGroup` for the same group fall back to their
        O(numNodes) scan path.

        :type  group: str
        :param group: Name of the indexed group to drop.

        :raises KeyError: If ``group`` is not currently indexed.
        """
        if group not in self._indexes:
            raise KeyError(group)
        del self._indexes[group]


    def indexedGroups(self) -> frozenset[str]:
        """
        Return the set of group names currently indexed.

        The returned set is a snapshot; mutating the index catalog
        after the call does not affect previously returned values.

        :rtype: frozenset[str]
        :returns: The names of all currently maintained group
            indexes.
        """
        return frozenset(self._indexes)


    def groups(self, group : str) -> Iterator[Hashable]:
        """
        Iterate over the distinct values observed for a group.

        If ``group`` has been registered via :meth:`buildIndex`, the
        fast path is O(k) in the number of distinct values. Otherwise
        the slow path falls back to an O(numNodes) scan over the
        node-payload field of the same name.

        :type  group: str
        :param group: Name of the node-payload field to enumerate.

        :rtype: Iterator[Hashable]
        :returns: An iterator over the distinct values found in
            ``group``.

        :raises KeyError: If no node carries the requested field.
        """
        if group in self._indexes:
            return iter(self._indexes[group])
        return self._scanGroups(group)


    def _scanGroups(self, group : str) -> Iterator[Hashable]:
        """Slow-path scan for :meth:`groups` when no index exists."""
        seen  : set[Hashable] = set()
        found = False
        for _, attrs in self.G.nodes(data=True):
            if group in attrs:
                found = True
                if attrs[group] not in seen:
                    seen.add(attrs[group])
                    yield attrs[group]
        if not found:
            raise KeyError(group)


    def nodesInGroup(self, group : str, value : Hashable) -> Iterator[NodeID]:
        """
        Iterate over the nodes whose ``group`` field equals ``value``.

        If ``group`` is indexed the fast path is O(k) in the size of
        the slice; otherwise the slow path falls back to an
        O(numNodes) scan.

        :type  group: str
        :param group: Name of the node-payload field to filter on.

        :type  value: Hashable
        :param value: The value that the field must equal.

        :rtype: Iterator[NodeID]
        :returns: An iterator over matching node identifiers.

        :raises KeyError: If no node carries the requested field.
        """
        if group in self._indexes:
            return iter(self._indexes[group].get(value, set()))
        return self._scanNodesInGroup(group, value)


    def _scanNodesInGroup(
        self, group : str, value : Hashable,
    ) -> Iterator[NodeID]:
        """Slow-path scan for :meth:`nodesInGroup` when no index exists."""
        found = False
        for nodeId, attrs in self.G.nodes(data=True):
            if group in attrs:
                found = True
                if attrs[group] == value:
                    yield nodeId
        if not found:
            raise KeyError(group)


    def sliceByGroup(
        self,
        group : str,
        value : Hashable | Iterable[Hashable],
    ) -> Self:
        """
        Return a new graph containing only the nodes that match.

        ``value`` may be a single value or an iterable of values; in
        the iterable case the slice contains the union of all
        matching nodes. The returned graph is the induced subgraph
        on those nodes.

        Returns a copy, not a view. :mod:`igraph` has no view
        abstraction, and OLAP-style slices need snapshot semantics so
        that downstream mutations on the slice do not propagate back
        into the parent graph (and vice versa).

        :type  group: str
        :param group: Name of the node-payload field to slice on.

        :type  value: Hashable | Iterable[Hashable]
        :param value: Single value or iterable of values to match.

        :rtype: Self
        :returns: A new graph of the same concrete type holding the
            sliced subgraph.

        :raises KeyError: If no node carries the requested field.
        """
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            values : set[Hashable] = set(value)
        else:
            values = {value}
        keep = {nid for v in values for nid in self.nodesInGroup(group, v)}
        return self.subgraph(keep)


    def subgraph(self, nodes : Iterable[NodeID]) -> Self:
        """
        Return the induced subgraph on the given nodes.

        The result contains exactly the requested nodes (those that
        exist in the parent graph) and every edge whose endpoints
        both belong to the requested set.

        Returns a copy, not a view. :mod:`igraph` has no view
        abstraction, and OLAP-style slices need snapshot semantics so
        that downstream mutations on the subgraph do not propagate
        back into the parent graph (and vice versa).

        :type  nodes: Iterable[NodeID]
        :param nodes: Identifiers of the nodes to retain.

        :rtype: Self
        :returns: A new graph of the same concrete type holding the
            induced subgraph.
        """
        keep = {n for n in nodes if n in self.G}
        return type(self)(
            nodeModel = self._nodeModel,
            edgeModel = self._edgeModel,
            G         = self.G.subgraph(keep).copy(),
            name      = self.name,
        )


    def nodesToFrame(self, *, group : str | None = None) -> "pd.DataFrame":
        """
        Materialize the node table as a :class:`pandas.DataFrame`.

        The frame's index is the node identifier and its columns are
        the fields of the typed node payload. When ``group`` is
        provided, the frame is restricted to nodes whose payload
        carries that field.

        :type  group: str | None
        :param group: Optional node-payload field to focus on.

        :rtype: pandas.DataFrame
        :returns: A tabular materialization of the node set.

        :raises KeyError: If ``group`` is given and no node carries
            that field.
        """
        import pandas as pd
        records = [
            {"nodeId": nid, **attrs}
            for nid, attrs in self.G.nodes(data=True)
        ]
        df = pd.DataFrame.from_records(records).set_index("nodeId")
        if group is not None:
            if group not in df.columns:
                raise KeyError(group)
            df = df.dropna(subset=[group])
        return df


    def edgesToFrame(self, *, group : str | None = None) -> "pd.DataFrame":
        """
        Materialize the edge table as a :class:`pandas.DataFrame`.

        The frame carries one row per edge with ``u`` and ``v``
        columns for the endpoints and additional columns for the
        fields of the typed edge payload. When ``group`` is provided,
        the frame is restricted to edges whose payload carries that
        field.

        :type  group: str | None
        :param group: Optional edge-payload field to focus on.

        :rtype: pandas.DataFrame
        :returns: A tabular materialization of the edge set.

        :raises KeyError: If ``group`` is given and no edge carries
            that field.
        """
        import pandas as pd
        records = [
            {"u": u, "v": v, **attrs}
            for u, v, attrs in self.G.edges(data=True)
        ]
        df = pd.DataFrame.from_records(records)
        if group is not None:
            if group not in df.columns:
                raise KeyError(group)
            df = df.dropna(subset=[group])
        return df


    def __repr__(self) -> str:
        """Return a short developer-facing string representation."""
        return (
            f"<{type(self).__name__} nodes={self.numNodes} "
            f"edges={self.numEdges} directed=True>"
        )


    def _validateNode(self, raw : N | NodeData | None) -> dict[str, Any]:
        """Validate and normalize a node payload into a plain dict."""
        if raw is None:
            return {}
        if isinstance(raw, self._nodeModel):
            return raw.model_dump()
        return self._nodeModel.model_validate(raw).model_dump()


    def _validateEdge(self, raw : E | EdgeData | None) -> dict[str, Any]:
        """Validate and normalize an edge payload into a plain dict."""
        if raw is None:
            return {}
        if isinstance(raw, self._edgeModel):
            return raw.model_dump()
        return self._edgeModel.model_validate(raw).model_dump()


    def _updateIndexesOnAdd(
        self, nodeId : NodeID, payload : Mapping[str, Any],
    ) -> None:
        """Insert a freshly added node into every live secondary index."""
        for group, index in self._indexes.items():
            if group in payload:
                index.setdefault(payload[group], set()).add(nodeId)


    def _updateIndexesOnRemove(self, nodeId : NodeID) -> None:
        """Evict a node from every live secondary index before removal."""
        attrs = self.G.nodes[nodeId]
        for group, index in self._indexes.items():
            if group in attrs:
                bucket = index.get(attrs[group])
                if bucket is None:
                    continue
                bucket.discard(nodeId)
                if not bucket:
                    del index[attrs[group]]
