# -*- encoding: utf-8 -*-

"""
Abstract Base Class & Explicit Edges for a Supply Chain Problem
---------------------------------------------------------------

An edge is a fundamental component in graph theory, which is used to
define the cost function in a supply chain optimization problem and
is the default that establishes a relationship between two nodes 
(or "vertices) "of the graph.

The module defines the abstract base edge improved with data validation
using :mod:`pydantic` with other default configurations.
"""

from typing import Final, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from arcline.graph.base.nodes import AbstractNode

class AbstractEdge(BaseModel, ABC):
    """
    Abstract definition of a graph edge which defines the relationship
    between two nodes of the graph and is the fundamental property
    that establishes a cost function in an optimization problem.
    """

    name : Final[str] = Field(
        ..., frozen = True, description = "Human Redable Edge Name"
    )

    hashKey : Final[str] = Field(
        ..., frozen = True, description = "Machine Reabable Edge Name"
    )

    srcNode : Final[AbstractNode] = Field(
        ..., frozen = True, description = "Source Node Object"
    )

    dstNode : Final[AbstractNode] = Field(
        ..., frozen = True, description = "Destination Node Object"
    )


    @property
    @abstractmethod
    def edgeColor(self) -> Optional[str]:
        """
        An optional color settings which can be used to enhance the
        graph visualization. This can be any valid ``HEX`` or any
        supported values based on the visualization module of choice.
        """

        pass
