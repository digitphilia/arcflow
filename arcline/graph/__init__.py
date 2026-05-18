# -*- encoding: utf-8 -*-

"""
A Full-Fledged Supply Chain Network Capable of Handling Complex Tasks
=====================================================================

A supply-chain network is typically a graph of inter-connected nodes
between various plants, suppliers, manufacturing units, customers, etc.
and edges which defines the lead time, delivery time etc. which can
generally have thousands of nodes and millions of edges (depending on
the organization structure) that sits at the boundary between an OLAP
analytics workload and a graph-traversal workload.

A dynamic network built on :mod:`NetworkX` module collapses past ~1M
edges (pure-Python overhead) and the alternate :mod:`igraph` (C-level
performance for full network compute) can be used to address larger
problems. To address this, a backend abstraction logic is defined which
is module agnostic with capabilities of indexation on groups.
"""

