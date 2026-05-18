# -*- encoding: utf-8 -*-

"""
Reference Pydantic Payload Models for the Supply-Chain Example
--------------------------------------------------------------

The :class:`arcline.graph.base.AbstractGraph` API is generic over the
node-payload type ``N`` and the edge-payload type ``E``. This module
ships the reference payload models used by the supply-chain example
and by the package's own tests:

* :class:`NodeType` enumerates the four canonical node categories
  (supplier, plant, warehouse, customer).
* :class:`SupplyChainNode` is the canonical node payload.
* :class:`SupplyChainLane` is the canonical edge payload, modeling a
  transportation lane between two nodes.

User code is free to define alternate Pydantic models with arbitrary
fields and bind them to a concrete backend at construction time; these
classes are conveniences, not a required schema.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """
    Canonical node categories for a supply-chain graph.

    The enum is a :class:`str` subclass so members serialize cleanly
    through Pydantic, JSON, and pandas without an explicit converter.
    Each member's value is the lowercased form of its name.
    """

    SUPPLIER  = "supplier"
    PLANT     = "plant"
    WAREHOUSE = "warehouse"
    CUSTOMER  = "customer"


class SupplyChainNode(BaseModel):
    """
    Reference node payload for a supply-chain graph.

    A node carries a human-readable ``name``, a categorical
    ``nodeType``, a ``region`` for geographic grouping, and optional
    purchasing/material classification fields that map onto common
    ERP attributes. ``capacity`` is the throughput ceiling expressed
    in payload-defined units and is non-negative when present.
    """

    name           : str
    nodeType       : NodeType
    region         : str
    purchaseGroup  : str | None   = None
    materialsGroup : str | None   = None
    capacity       : float | None = Field(default=None, ge=0)


class SupplyChainLane(BaseModel):
    """
    Reference edge payload modeling a transportation lane.

    A lane connects two nodes and carries the operational parameters
    a planner cares about: ``leadTime`` and ``transportCost`` are
    non-negative, ``capacity`` is strictly positive (a lane with zero
    capacity is not a lane), and ``carrier`` identifies the logistics
    provider. ``mode`` defaults to ``"road"`` for the common case.
    """

    leadTime      : float = Field(ge=0)
    transportCost : float = Field(ge=0)
    carrier       : str
    capacity      : float = Field(gt=0)
    mode          : str   = "road"
