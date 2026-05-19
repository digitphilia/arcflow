# -*- encoding: utf-8 -*-

"""
Abstract Base Class & Explicit Nodes for a Supply Chain Problem
---------------------------------------------------------------

A node (or "vertex") is the fundamental component in graph theory. In
supply chain optimization problems, a node is a variable which is
subjected to change based on different cost functions that is derived
from the interconnected edges of the graph.

The module defines the abstract base node improved with data validation
using :mod:`pydantic` with other default configurations.
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Final, Optional

class AbstractNode(BaseModel, ABC):
    """
    Abstract definition of a graph node which holds the critical
    properties of the node. In a supply chain optimization problem a
    node's property are subjected to change; however there can be
    some default property (like ``name`` or ``id``) which are  set to
    frozen values and can only be defined during initialization.

    :param nodeData: A node can have multiple default attributes that
        is typically allowed in :mod:`networkx` and :mod:`igraph` to
        be defined as nodes' property during initialization.
    """

    name : Final[str] = Field(
        ..., frozen = True, description = "Human Redable Node Name"
    )

    hashKey : Final[str] = Field(
        ..., frozen = True, description = "Machine Reabable Node Name"
    )

    nodeData : Optional[Dict[str, Any]] = Field(
        None, description = "Any Additional Node Attribute(s)"
    )


    @property
    @abstractmethod
    def imagePath(self) -> Optional[str]:
        """
        A custom icon path is supported by :mod:`networkx` which can
        be passed to visualization library like :mod:`gravis` or
        :mod:`matplotlib` for optional visualization enhancement. This
        property should be set to ``None`` if not required. The image
        should be of type ``.png`` as supported by the module.
        """

        pass


    @property
    @abstractmethod
    def nodeColor(self) -> Optional[str]:
        """
        An optional color settings which can be used to enhance the
        graph visualization. This can be any valid ``HEX`` or any
        supported values based on the visualization module of choice.
        """

        pass
