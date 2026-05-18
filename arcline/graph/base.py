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
compute).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import Any, Generic, Self, TypeAlias, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

NodeID   : TypeAlias = Hashable
EdgeKey  : TypeAlias = tuple[NodeID, NodeID]
NodeData : TypeAlias = Mapping[str, Any]
EdgeData : TypeAlias = Mapping[str, Any]

N = TypeVar("N")
E = TypeVar("E")


class AbstractGraph(ABC, Generic[N, E]):
    """
    Abstract definition of the generic graph backend which is used as
    an API to interface with different modules (:mod:`networkx` or
    :mod:`igraph`) for a full-network compute or slice the graph on
    defined groups for index level optimization.

    A supply chain graph are almost always directional (asymmetric
    lead time and cost from origin to destination) thus the graph
    should always be of type :class:`networkx.DiGraph` or should be
    defined like :class:`igraph.Graph(directed - True)` and passed for
    initialization. In rare cases, when both the direction is possible
    then both ``A → B`` and ``B → A`` edges must be defined.
    """

    @property
    @abstractmethod
    def numNodes(self) -> int: ...


    @property
    @abstractmethod
    def numEdges(self) -> int: ...


    @property
    @abstractmethod
    def isMultiGraph(self) -> bool:
        """
        Whether the graph allows multiple parallel edges between the
        same pair of endpoints.

        A multigraph is required when distinct lanes between the same
        origin and destination must be modeled independently (e.g.
        different carriers, contracts, or transportation modes). Like
        :attr:`directed`, this attribute is decided at construction
        time and cannot be flipped on a live instance.

        :rtype: bool
        :returns: ``True`` if parallel edges are permitted, ``False``
            otherwise.
        """
        ...


    @property
    @abstractmethod
    def backend(self) -> Any:
        """
        Return the underlying backend graph object.

        This is an escape hatch for advanced users who need to call
        backend-specific routines that the abstract surface does not
        expose (custom :mod:`igraph` community detection, NetworkX
        algorithm internals, etc.).

        .. warning::

            Code that touches ``backend`` is, by definition, no longer
            backend-agnostic. Treat such code paths as backend
            specific and gate them appropriately.

        :rtype: Any
        :returns: The wrapped backend graph (e.g. an
            :class:`igraph.Graph` or :class:`networkx.DiGraph`).
        """
        ...


    @abstractmethod
    def addNode(
        self,
        nodeId : NodeID,
        data   : N | NodeData | None = None,
    ) -> None:
        """
        Insert a single node into the graph.

        ``data`` may be a typed payload of type ``N`` (typically a
        Pydantic model) or a plain mapping with the same fields;
        concrete backends validate and normalize the input. Passing
        ``None`` creates a bare node with no associated attributes.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier for the node.

        :type  data: N | NodeData | None
        :param data: Optional payload describing the node.

        :raises ValueError: If ``nodeId`` already exists and the
            backend's collision policy is strict.
        """
        ...


    @abstractmethod
    def addNodes(
        self,
        nodes : Iterable[NodeID | tuple[NodeID, N | NodeData]],
    ) -> None:
        """
        Insert a batch of nodes in a single backend call.

        Each element of ``nodes`` is either a bare node identifier or
        a ``(nodeId, data)`` tuple. ``data`` may be a Pydantic payload
        of type ``N`` or a plain mapping with the same fields;
        concrete backends validate and normalize the input. Bulk
        insertion is a distinct method (not a default loop over
        :meth:`addNode`) because backends such as :mod:`igraph`
        provide vectorized vertex creation that is one to two orders
        of magnitude faster than per-node calls.

        :type  nodes: Iterable[NodeID | tuple[NodeID, N | NodeData]]
        :param nodes: Heterogeneous iterable of node identifiers,
            with optional inline payload.

        :raises ValueError: If any element is malformed (e.g. a tuple
            of the wrong arity) or if a node identifier already
            exists and the backend's collision policy is strict.
        """
        ...


    @abstractmethod
    def removeNode(self, nodeId : NodeID) -> None:
        """
        Remove a node and all of its incident edges from the graph.

        Removal is cascading: every edge that touches ``nodeId`` is
        also deleted, regardless of direction. Indexes that reference
        the node are updated transparently by the backend.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node to remove.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def hasNode(self, nodeId : NodeID) -> bool:
        """
        Test whether a node is present in the graph.

        This is the cheap membership check used by :meth:`__contains__`
        and by any caller that needs an existence test without
        materializing the node payload.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier to test for membership.

        :rtype: bool
        :returns: ``True`` if ``nodeId`` is a node of the graph,
            ``False`` otherwise.
        """
        ...


    @abstractmethod
    def getNode(self, nodeId : NodeID) -> N:
        """
        Retrieve the typed payload attached to a node.

        The returned object is of the generic node-payload type ``N``
        bound to the concrete subclass (typically a Pydantic model).
        Implementations are expected to reconstruct ``N`` from the
        backend's native attribute storage on each call; callers
        should not mutate the returned instance and expect the change
        to round-trip into the graph.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node to fetch.

        :rtype: N
        :returns: The typed payload attached to ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def addEdge(
        self,
        u    : NodeID,
        v    : NodeID,
        data : E | EdgeData | None = None,
    ) -> None:
        """
        Insert a single edge between two nodes.

        For directed graphs the edge runs from ``u`` to ``v``; for
        undirected graphs the ordering of ``u`` and ``v`` is
        immaterial. ``data`` may be a typed payload of type ``E`` or
        a plain mapping with the same fields.

        :type  u: NodeID
        :param u: Source node identifier (or one endpoint, if
            undirected).

        :type  v: NodeID
        :param v: Target node identifier (or the other endpoint, if
            undirected).

        :type  data: E | EdgeData | None
        :param data: Optional payload describing the edge.

        :raises KeyError: If either ``u`` or ``v`` is not present in
            the graph.
        :raises ValueError: If the edge already exists and the
            backend's collision policy is strict (non-multigraph
            case).
        """
        ...


    @abstractmethod
    def addEdges(
        self,
        edges : Iterable[tuple[NodeID, NodeID] | tuple[NodeID, NodeID, E | EdgeData]],
    ) -> None:
        """
        Insert a batch of edges in a single backend call.

        Each element of ``edges`` is either a ``(u, v)`` endpoint
        pair or a ``(u, v, data)`` triple. As with :meth:`addNodes`,
        this is a distinct method rather than a default loop over
        :meth:`addEdge` because vectorized edge insertion in
        :mod:`igraph` is dramatically faster than per-edge calls on
        large networks.

        :type  edges: Iterable[tuple[NodeID, NodeID]
            | tuple[NodeID, NodeID, E | EdgeData]]
        :param edges: Heterogeneous iterable of endpoint pairs, with
            optional inline payload.

        :raises KeyError: If any endpoint is not present in the
            graph.
        :raises ValueError: If any element has the wrong arity or if
            an edge already exists and the backend's collision
            policy is strict.
        """
        ...


    @abstractmethod
    def removeEdge(self, u : NodeID, v : NodeID) -> None:
        """
        Remove the edge between ``u`` and ``v`` from the graph.

        On a multigraph this removes all parallel edges between the
        two endpoints; backends that need finer-grained removal (by
        edge key) should expose that through their backend-specific
        API rather than through this method.

        :type  u: NodeID
        :param u: Source node identifier (or one endpoint, if
            undirected).

        :type  v: NodeID
        :param v: Target node identifier (or the other endpoint, if
            undirected).

        :raises KeyError: If the edge ``(u, v)`` does not exist.
        """
        ...


    @abstractmethod
    def hasEdge(self, u : NodeID, v : NodeID) -> bool:
        """
        Test whether an edge exists between two nodes.

        For directed graphs this checks specifically for an edge from
        ``u`` to ``v``; for undirected graphs the direction is
        ignored.

        :type  u: NodeID
        :param u: Source node identifier (or one endpoint, if
            undirected).

        :type  v: NodeID
        :param v: Target node identifier (or the other endpoint, if
            undirected).

        :rtype: bool
        :returns: ``True`` if an edge connects ``u`` and ``v``,
            ``False`` otherwise.
        """
        ...


    @abstractmethod
    def getEdge(self, u : NodeID, v : NodeID) -> E:
        """
        Retrieve the typed payload attached to an edge.

        The returned object is of the generic edge-payload type ``E``
        bound to the concrete subclass (typically a Pydantic model).
        On a multigraph this returns the payload of an unspecified
        parallel edge; callers that need to disambiguate must fall
        back to backend-specific access.

        :type  u: NodeID
        :param u: Source node identifier (or one endpoint, if
            undirected).

        :type  v: NodeID
        :param v: Target node identifier (or the other endpoint, if
            undirected).

        :rtype: E
        :returns: The typed payload attached to the edge.

        :raises KeyError: If the edge ``(u, v)`` does not exist.
        """
        ...


    @abstractmethod
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
        :returns: An iterator over node identifiers, optionally
            paired with their typed payloads.
        """
        ...


    @abstractmethod
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

        :type  data: bool
        :param data: Whether to include the edge payload alongside
            each endpoint pair.

        :rtype: Iterator[EdgeKey] | Iterator[tuple[NodeID, NodeID, E]]
        :returns: An iterator over edge endpoint pairs, optionally
            paired with their typed payloads.
        """
        ...


    @abstractmethod
    def neighbors(self, nodeId : NodeID) -> Iterator[NodeID]:
        """
        Iterate over the neighbors of a node.

        On a directed graph this is equivalent to
        :meth:`successors`: it yields the out-neighbors of ``nodeId``.
        On an undirected graph it yields every node incident to
        ``nodeId`` regardless of direction.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            neighbors are requested.

        :rtype: Iterator[NodeID]
        :returns: An iterator over neighboring node identifiers.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def predecessors(self, nodeId : NodeID) -> Iterator[NodeID]:
        """
        Iterate over the in-neighbors of a node.

        On a directed graph this yields every node ``u`` such that
        an edge ``u -> nodeId`` exists. On an undirected graph the
        notion of direction does not apply and this falls back to
        :meth:`neighbors`.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            in-neighbors are requested.

        :rtype: Iterator[NodeID]
        :returns: An iterator over predecessor node identifiers.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def successors(self, nodeId : NodeID) -> Iterator[NodeID]:
        """
        Iterate over the out-neighbors of a node.

        On a directed graph this yields every node ``v`` such that
        an edge ``nodeId -> v`` exists. On an undirected graph the
        notion of direction does not apply and this falls back to
        :meth:`neighbors`.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            out-neighbors are requested.

        :rtype: Iterator[NodeID]
        :returns: An iterator over successor node identifiers.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def degree(self, nodeId : NodeID) -> int:
        """
        Return the total degree of a node.

        On an undirected graph this is the count of incident edges.
        On a directed graph this is ``inDegree(nodeId) +
        outDegree(nodeId)``; on a multigraph each parallel edge
        contributes independently.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose degree
            is requested.

        :rtype: int
        :returns: The total degree of ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def inDegree(self, nodeId : NodeID) -> int:
        """
        Return the in-degree of a node.

        On a directed graph this is the count of incoming edges. On
        an undirected graph it falls back to :meth:`degree` because
        in- and out-degree are indistinguishable.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            in-degree is requested.

        :rtype: int
        :returns: The in-degree of ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def outDegree(self, nodeId : NodeID) -> int:
        """
        Return the out-degree of a node.

        On a directed graph this is the count of outgoing edges. On
        an undirected graph it falls back to :meth:`degree` because
        in- and out-degree are indistinguishable.

        :type  nodeId: NodeID
        :param nodeId: Hashable identifier of the node whose
            out-degree is requested.

        :rtype: int
        :returns: The out-degree of ``nodeId``.

        :raises KeyError: If ``nodeId`` is not present in the graph.
        """
        ...


    @abstractmethod
    def buildIndex(self, group : str) -> None:
        """
        Build or rebuild a secondary index over a node-payload field.

        Indexes accelerate group-wise slicing
        (:meth:`nodesInGroup`, :meth:`sliceByGroup`) from O(numNodes)
        to O(k) where k is the size of the slice. Re-calling
        :meth:`buildIndex` for a group that is already indexed
        rebuilds it from scratch; this is the supported way to
        refresh stale indexes after bulk insertions or removals that
        the backend does not transparently track.

        :type  group: str
        :param group: Name of the node-payload field to index on.

        :raises KeyError: If no node carries the requested field.
        """
        ...


    @abstractmethod
    def dropIndex(self, group : str) -> None:
        """
        Drop a previously built secondary index.

        After dropping, subsequent calls to :meth:`groups` and
        :meth:`nodesInGroup` for the same group will fall back to
        their O(numNodes) scan path.

        :type  group: str
        :param group: Name of the indexed group to drop.

        :raises KeyError: If ``group`` is not currently indexed.
        """
        ...


    @abstractmethod
    def indexedGroups(self) -> frozenset[str]:
        """
        Return the set of group names currently indexed.

        The returned set is a snapshot; mutating the graph's index
        catalog (via :meth:`buildIndex` or :meth:`dropIndex`) after
        the call does not affect previously returned values.

        :rtype: frozenset[str]
        :returns: The names of all currently maintained group
            indexes.
        """
        ...


    @abstractmethod
    def groups(self, group : str) -> Iterator[Hashable]:
        """
        Iterate over the distinct values observed for a group.

        If ``group`` has been registered via :meth:`buildIndex`, this
        is O(k) in the number of distinct values. Otherwise the
        implementation falls back to an O(numNodes) scan over the
        node payload field of the same name.

        :type  group: str
        :param group: Name of the node-payload field to enumerate.

        :rtype: Iterator[Hashable]
        :returns: An iterator over the distinct values found in
            ``group``.

        :raises KeyError: If no node carries the requested field.
        """
        ...


    @abstractmethod
    def nodesInGroup(
        self,
        group : str,
        value : Hashable,
    ) -> Iterator[NodeID]:
        """
        Iterate over the nodes whose ``group`` field equals ``value``.

        If ``group`` is indexed this is O(k) in the size of the
        slice; otherwise it falls back to an O(numNodes) scan.

        :type  group: str
        :param group: Name of the node-payload field to filter on.

        :type  value: Hashable
        :param value: The value that the field must equal.

        :rtype: Iterator[NodeID]
        :returns: An iterator over matching node identifiers.

        :raises KeyError: If no node carries the requested field.
        """
        ...


    @abstractmethod
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
        on those nodes (every edge whose endpoints both belong to
        the slice).

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
        ...


    @abstractmethod
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
        ...


    @abstractmethod
    def nodesToFrame(self, *, group : str | None = None) -> pd.DataFrame:
        """
        Materialize the node table as a :class:`pandas.DataFrame`.

        The frame's index is the node identifier and its columns
        are the fields of the typed node payload. When ``group`` is
        provided, the frame is restricted to nodes whose payload
        carries that field (and the field is surfaced as a column
        suitable for downstream :meth:`pandas.DataFrame.groupby`).

        :type  group: str | None
        :param group: Optional node-payload field to focus on.

        :rtype: pandas.DataFrame
        :returns: A tabular materialization of the node set.

        :raises KeyError: If ``group`` is given and no node carries
            that field.
        """
        ...


    @abstractmethod
    def edgesToFrame(self, *, group : str | None = None) -> pd.DataFrame:
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
        ...


    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return self.numNodes


    def __contains__(self, item : NodeID) -> bool:
        """Return ``True`` if ``item`` is a node of the graph."""
        return self.hasNode(item)


    def __iter__(self) -> Iterator[NodeID]:
        """Return an iterator over the node identifiers of the graph."""
        return self.nodes(data=False)


    @abstractmethod
    def __repr__(self) -> str:
        """
        Return a developer-facing string representation of the graph.

        Recommended shape: ``<TypeName nodes=N edges=M directed=B>``
        so that concrete implementations stay consistent across
        backends and so that log lines remain greppable.

        :rtype: str
        :returns: A short string describing the graph.
        """
        ...
