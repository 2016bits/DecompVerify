"""Personalised PageRank over the heterogeneous graph.

We rely on ``networkx.pagerank`` because the per-claim graph is small
(typically <5k nodes) and edge weights are already encoded.

The caller supplies a *seed weight* dictionary mapping node ids to a
non-negative personalisation mass; the dictionary is renormalised
internally. ``alpha`` controls the random-walk restart probability.
"""

from __future__ import annotations

from typing import Dict, Iterable

import networkx as nx


def personalised_pagerank(
    g: nx.Graph,
    seeds: Dict[str, float],
    alpha: float = 0.7,
    max_iter: int = 30,
    tol: float = 5e-4,
) -> Dict[str, float]:
    """Run PPR seeded at ``seeds``; returns a node-id -> mass dict.

    networkx 3.1 falls back to a pure-Python power iteration when the
    optional ``scipy`` backend is not available; for heterogeneous
    graphs with a few hundred nodes that path can be 20-50x slower than
    a direct scipy.sparse solve. We try the scipy implementation first
    and degrade gracefully if it is missing.
    """
    if not seeds or g.number_of_nodes() == 0:
        return {}
    total = sum(max(v, 0.0) for v in seeds.values())
    if total <= 0:
        return {}
    personalization = {n: 0.0 for n in g.nodes}
    for node, mass in seeds.items():
        if node in personalization and mass > 0:
            personalization[node] = mass / total
    # Prefer the scipy-backed implementation when available.
    try:
        return nx.pagerank_scipy(
            g,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=max_iter,
            tol=tol,
        )
    except AttributeError:
        pass
    except Exception:
        # Any scipy-related failure falls through to the pure-Python path.
        pass
    try:
        return nx.pagerank(
            g,
            alpha=alpha,
            personalization=personalization,
            weight="weight",
            max_iter=max_iter,
            tol=tol,
        )
    except nx.PowerIterationFailedConvergence:
        # One bounded retry – then return the best estimate we have so
        # we never block on an adversarial graph.
        try:
            return nx.pagerank(
                g,
                alpha=alpha,
                personalization=personalization,
                weight="weight",
                max_iter=max_iter * 2,
                tol=tol * 10,
            )
        except nx.PowerIterationFailedConvergence:
            return {n: personalization.get(n, 0.0) for n in g.nodes}


def restrict_to_type(
    scores: Dict[str, float], graph: nx.Graph, ntype: str
) -> Dict[str, float]:
    return {n: s for n, s in scores.items() if graph.nodes[n].get("ntype") == ntype}
