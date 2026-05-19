# -*- encoding: utf-8 -*-

"""
Abstract Supply Chain Base Graph/Network Definition
---------------------------------------------------

A supply chain problem can typically be defined as a network of nodes
and edges - where each node (or "vertex") can be considered as a
variable subjected to a change and the edges defines the relationship
that can be modeled to define a cost function.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, List, Optional

from arcline.graph.base.nodes import AbstractNode
from arcline.graph.base.edges import AbstractEdge


class AbstractGraph(ABC):
    """
    Abstract definition of the generic graph backend which is used as
    an API to interface with different modules (:mod:`networkx` or
    :mod:`igraph`) for a full-network compute or slice the graph on
    defined groups for index level optimization.

    :type  G: Union[MultiDiGraph, Graph, ...]
    :param G: An of graph objects from any compatible modules like
        :class:`networkx.Graph`, or :class:`networkx.MultiDiGraph`, or
        :class:`igraph.Graph` etc. The abstract class is designed to
        work with any concrete module implementation without creating
        any hard dependency for additional available backends.

    :type  name: str
    :param name: Name of the model, this can be any valid string value,
        useful for any type of logging or auditing purposes. Defaults
        to none, which sets the attribute as the name of class.

    :NOTE: A supply chain graph are almost always directional
    (asymmetric lead time and cost from origin to destination) thus
    the graph should always be of type :class:`networkx.DiGraph` or
    should be defined like :class:`igraph.Graph(directed - True)` and
    passed for initialization. In rare cases, when both the direction
    is possible then both ``A → B`` and ``B → A`` edges must be defined.

    :NOTE: The graph will always allow multiple-parallel edges between
    the same pair of endpoints. This ensures that distinct lanes must
    be modeled independently; e.g., different carriers, or contractors,
    or transportation modes. This also ensures that edges are to be
    defined in a manner that the distinct lanes between the source and
    the destination from the multi-graph can be sliced properly based
    on the group definitions, criteria, etc. which should be the
    property of the edge attribute - in concrete implementation. The
    native :class:`networkx.MultiDiGraph` is available, but the default
    alternate is not available directly in the :mod:`igraph` module -
    where the end user needs to define and handle the edge IDs and the
    attributes seperately which is available in the base.

    **Backend Graph Object: Advanced Usage**

    The underlying backend graph object :class:`networkx.MultiDiGraph`
    or the :class:`igraph.Graph` is available as ``G`` attribute to
    the class. This is useful for advanced users who wants to work with
    backend-specific routines that the abstract surface does not
    automatically expose (custom :mod:`igraph` community detection, or
    :mod:`networkx` internal algorithms, etc.) directly.

    .. code-block:: python

        import arcline
        scnetwork = arcline.graph[...]

        print(type(scnetwork.G))
        >> nx.MultiDiGraph or igraph.Graph

    **Backend Agnostic Base Methods**

    Some basic functions/graph attributes are directly available as
    part of the base class definition. This ensures that agnostic
    approach is maintained when switching modules to define the graph.

    .. code-block:: python

        # .numNodes → Get the Total Number of Nodes in the Graph
        # iGraph Method   : igraph.Graph.vcount()
        # NetworkX Method : networkx.MultiDiGraph.number_of_nodes()
        print(scnetwork.numNodes)
        >> XYZ

        # .numEdges → Get the Total Number of Edges in the Graph
        # iGraph Method   : igraph.Graph.ecount()
        # NetworkX Method : networkx.MultiDiGraph.number_of_edges()
        print(scnetwork.numEdges)
        >> XYZ
    """

    def __init__(
            self,
            G : Any,
            nodes : List[AbstractNode],
            edges : List[AbstractEdge],
            name : Optional[str] = None
    ) -> None:
        self.G = G

        # define the components of the graph in the base class
        self.nodes = nodes
        self.edges = edges

        # name of the graph class; defaults to class name
        self.name = self.__set_name__(name = name)


    @property
    def numNodes(self) -> int:
        """
        The base abstract class expects all the nodes to be passed as
        an iterable (``list``) which can be directly used to return
        the number of nodes. However, in case of module agnostic way,
        for advanced users, the value can be checked agains the module
        like below:

        .. code-block:: python

            scnetwork = ... # Instance of AbstractGraph()

            # NetworkX Compatible Callings::
            assert scnetwork.G.number_of_nodes() == scnetwork.numNodes
            >> True

            # iGraph Compatible Callings::
            assert scnetwork.G.vcount() == scnetwork.numNodes
            >> True

        :rtype:   int
        :returns: Default agnostic class property to return the total
            number of nodes in the graph.
        """

        return len(self.nodes)


    @property
    def numEdges(self) -> int:
        """
        The base abstract class expects all the edges to be passed as
        an iterable (``list``) which can be directly used to return
        the number of edges. However, in case of module agnostic way,
        for advanced users, the value can be checked agains the module
        like below:

        .. code-block:: python

            scnetwork = ... # Instance of AbstractGraph()

            # NetworkX Compatible Callings::
            assert scnetwork.G.number_of_edges() == scnetwork.numEdges
            >> True

            # iGraph Compatible Callings::
            assert scnetwork.G.ecount() == scnetwork.numEdges
            >> True

        :rtype:   int
        :returns: Default agnostic class property to return the total
            number of edges in the graph.
        """

        return len(self.edges)


    @abstractmethod
    def buildGraph(self, *args, **kwargs) -> bool:
        """
        Module agnostic method to build the graph with nodes and edges
        based on the module of choice.
        """

        pass


    @abstractmethod
    def removeNode(self, node : AbstractNode) -> None:
        """
        Remove a node and all of its incident edges from the graph.
        Removal is cascading: every edge that touches ``node`` is
        also deleted, regardless of direction. Indexes that reference
        the node should be updated transparently by the backend.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :raises KeyError: The the node is not present in the graph,
            then key error is raised.
        """

        pass


    @abstractmethod
    def hasNode(self, node : AbstractNode) -> bool:
        """
        Test whether a node is present in the graph. This is the cheap
        membership check used by :meth:`__contains__` and by any
        caller that needs an existence test without materializing the
        node payload.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :rtype:   bool
        :returns: Returns ``True`` if the node is present, else
            returns ``False``.
        """

        pass


    @abstractmethod
    def removeEdge(self, edge : AbstractEdge) -> None:
        """
        Remove the edge between ``u`` (source) and ``v`` (destination)
        from the graph. In a multi-graph, this method ensures that
        only the selected edge is removed - the ``edge`` object
        ensures that only the selected hash key is removed keeping all
        other edges between the source and destination intact.

        :type  edge: AbstractEdge
        :param edge: An instance of ``AbstractEdge`` which ensures
            that only one edge is removed between the source and the
            destination node.

        :raises KeyError: If the edge (``AbstractEdge``) does not
            exists in the graph.
        """

        pass


    @abstractmethod
    def hasEdge(self, src : AbstractNode, dst : AbstractNode) -> bool:
        """
        Test whether an edge exists between two nodes. In a multigraph,
        this returns all the edges connected between the nodes.

        :rtype:   bool
        :returns: Returns ``True`` if an edge (directional) is present,
            else returns ``False``.
        """

        pass


    @abstractmethod
    def neighbors(self, node : AbstractNode) -> Iterator[AbstractNode]:
        """
        Iterate over the neighbors of a node. For a multi-directed
        graph, this returns only the unique nodes which are neighbors.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :raises KeyError: The the node is not present in the graph,
            then key error is raised.

        :rtype:   Iterator[AbstractNode]
        :returns: An iterator of all the unique neighbors (nodes)
            present in the graph.
        """

        pass


    @abstractmethod
    def predecessors(self, node : AbstractNode) -> Iterator[AbstractNode]:
        """
        Iterate over the in-neighbors of a node. This yields the
        nodes connected via an edge like ``src → node`` where the list
        of ``src`` (source) nodes are returned.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :raises KeyError: The the node is not present in the graph,
            then key error is raised.

        :rtype:   Iterator[AbstractNode]
        :returns: An iterator of all the unique predecessors (nodes)
            present in the graph.
        """

        pass


    @abstractmethod
    def successors(self, node : AbstractNode) -> Iterator[AbstractNode]:
        """
        Iterate over the out-neighbors of a node. This methods yields
        the nodes connected like ``node → dst`` where the list of
        ``dst`` (destination) nodes are returned.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :raises KeyError: The the node is not present in the graph,
            then key error is raised.

        :rtype:   Iterator[AbstractNode]
        :returns: An iterator of all the unique successors (nodes)
            present in the graph.
        """

        pass


    def degree(self, node : AbstractNode) -> int:
        """
        Return the total degree of a node. This returns a total of
        the in-degrees and out-degrees for the node, and in a
        multi-graph (default) each edge contributes to the counter.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :raises KeyError: The the node is not present in the graph,
            then key error is raised.

        :rtype:   int
        :returns: Total number of in-degress and out-degrees to the
            node of the graph.
        """

        return self.inDegree(node = node) + self.outDegree(node = node)


    @abstractmethod
    def inDegree(self, node : AbstractNode) -> int:
        """
        Return the in-degree of a node. For a multi-graph (default)
        each connected edge contributes independently.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :raises KeyError: The the node is not present in the graph,
            then key error is raised.

        :rtype:   int
        :returns: Total number of in-degress connected to the node of
            the graph.
        """

        pass


    @abstractmethod
    def outDegree(self, node : AbstractNode) -> int:
        """
        Return the out-degree of a node. For a multi-graph (default)
        each connected edge contributes independently.

        :type  node: AbstractNode
        :param node: The node object to be removed from the graph,
            this should be an instance of ``AbstractNode`` and the
            node must be present in the graph.

        :raises KeyError: The the node is not present in the graph,
            then key error is raised.

        :rtype:   int
        :returns: Total number of out-degress connected to the node of
            the graph.
        """

        pass


    def __set_name__(self, name : Optional[str]) -> str:
        """
        Set the default name property of the model based on the name
        of the class, when not provided.
        """

        return name or self.__class__.__name__


    def __contains__(self, node : AbstractNode) -> bool:
        """
        A :mod:`math` compatible function that calls the dunder method
        such that ``math.contains(...)`` can be called.
        """

        return self.hasNode(node = node)
