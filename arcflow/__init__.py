# -*- encoding: utf-8 -*-

"""
A Python Framework for Optimization of Supply-Chain Networks
============================================================

The **`arcline`** is a Python framework for building, solving, and
analyzing supply chain optimization problems as network flow models.
Real supply chains are graphs. Suppliers, plants, warehouses,
distribution centers, and customers are nodes; the lanes between them
are arcs (or edges); and the optimization questions that matter - where
to source, how much to produce, which routes to use, when to open or
close a facility - are all decisions about flow on those arcs. The
project gives you a declarative API for modeling these networks, a
solver-agnostic backend (CBC, HiGHS, Gurobi, CPLEX), and first-class
tooling for the parts of the workflow that real practitioners spend
most of their time on: data ingestion, scenario comparison, sensitivity
analysis, and visualization. It is designed to bridge the gap between
the academic clarity of textbook formulations and the messy practical
needs of production supply chain teams. Whether you are running a
one-off facility location study, building a digital twin of a global
distribution network, or embedding a recurring optimization into a
daily planning pipeline, `arcline` aims to be the layer that makes the
network the first-class object - and the math, the I/O, and the solver
plumbing fade into the background.
"""

__version__ = "v0.0.1.dev0"

# init-time options registrations, use api/ for public tools exports
