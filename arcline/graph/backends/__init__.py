# -*- encoding: utf-8 -*-

"""
Concrete Backend Implementations of :class:`AbstractGraph`
----------------------------------------------------------

This subpackage hosts the concrete, backend-specific implementations
of :class:`arcline.graph.base.AbstractGraph`. Two backends are
planned: :mod:`networkx` for developer ergonomics and small subgraphs,
and :mod:`igraph` for C-level performance on the full supply-chain
network.

No symbols are re-exported here. Callers import the concrete class
they want by its full path (for example
``from arcline.graph.backends.networkx import NetworkXGraph``) so that
the choice of backend stays explicit at every call site.
"""
