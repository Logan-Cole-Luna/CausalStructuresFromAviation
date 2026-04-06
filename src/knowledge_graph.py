"""
Knowledge graph construction from extracted causal triples.

Builds a NetworkX DiGraph, computes statistics, exports Neo4j Cypher,
and visualizes a subgraph.
"""
import re
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def _check_networkx():
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx is required. Install it with: pip install networkx")


# Leading articles/determiners stripped during normalization
_STRIP_ARTICLES = re.compile(
    r'^(?:the|a|an|this|these|those|its|their|his|her)\s+',
    re.IGNORECASE,
)


def _normalize_entity(text: str) -> str:
    """
    Lowercase and strip leading articles so that "the loss of engine power",
    "a loss of engine power", and "loss of engine power" all map to the same node.
    Applied iteratively to handle stacked articles ("the a ...").
    """
    text = text.strip().lower()
    prev = None
    while prev != text:
        prev = text
        text = _STRIP_ARTICLES.sub('', text)
    return text.strip()


# Artifact phrases that don't represent meaningful aviation concepts
_NOISE_NODES: frozenset = frozenset({
    "the accident", "this accident", "the incident", "this incident",
    "an accident", "an incident", "the crash", "a crash",
    "the event", "this event", "the occurrence",
    "substantial damage", "the damage", "damage",
    "the airplane", "the aircraft", "the helicopter",
    "the flight", "the approach", "the landing", "the takeoff",
    "the pilot", "the crew", "the captain",
})


def _is_noise(text: str) -> bool:
    return text.lower().strip() in _NOISE_NODES or len(text.strip()) < 4


def build_graph(triples: List[dict], noise_filter: bool = True, normalize: bool = True):
    """
    Build a NetworkX DiGraph from a list of causal triple dicts.

    Each triple must have keys: 'cause', 'effect', 'relation'.
    Nodes get a 'type' attribute ('cause_node' or 'effect_node').
    Edges get 'relation' and 'weight' (incremented for duplicate edges).

    Returns a networkx.DiGraph.
    """
    _check_networkx()
    G = nx.DiGraph()

    # Use a separate counter dict for parallel edge weights
    edge_weights: dict = {}

    for triple in triples:
        cause = str(triple.get('cause', '')).strip()
        effect = str(triple.get('effect', '')).strip()
        relation = str(triple.get('relation', 'causes')).strip()

        if not cause or not effect:
            continue
        if normalize:
            cause  = _normalize_entity(cause)
            effect = _normalize_entity(effect)
        if noise_filter and (_is_noise(cause) or _is_noise(effect)):
            continue

        if not G.has_node(cause):
            G.add_node(cause, type='cause_node')
        if not G.has_node(effect):
            G.add_node(effect, type='effect_node')

        key = (cause, effect, relation)
        edge_weights[key] = edge_weights.get(key, 0) + 1

    # Add edges with accumulated weights
    for (cause, effect, relation), weight in edge_weights.items():
        if G.has_edge(cause, effect):
            # If an edge already exists (different relation), keep max weight
            G[cause][effect]['weight'] = G[cause][effect].get('weight', 1) + weight
        else:
            G.add_edge(cause, effect, relation=relation, weight=weight)

    return G


def graph_stats(G) -> dict:
    """
    Compute and return graph statistics dict:
      num_nodes, num_edges, density, weakly_connected_components,
      top_causes (top 10 by out_degree), top_effects (top 10 by in_degree),
      top_nodes_by_betweenness (top 5 by betweenness centrality).
    """
    _check_networkx()

    if len(G) == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'density': 0.0,
            'weakly_connected_components': 0,
            'top_causes': [],
            'top_effects': [],
            'top_nodes_by_betweenness': [],
        }

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    wcc = nx.number_weakly_connected_components(G)

    top_causes = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]
    top_effects = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:10]

    # Betweenness centrality — approximate for large graphs
    k_approx = min(100, len(G))
    try:
        betweenness = nx.betweenness_centrality(G, k=k_approx, normalized=True)
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    except Exception:
        top_betweenness = []

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': round(density, 6),
        'weakly_connected_components': wcc,
        'top_causes': [(n, d) for n, d in top_causes],
        'top_effects': [(n, d) for n, d in top_effects],
        'top_nodes_by_betweenness': [(n, round(v, 4)) for n, v in top_betweenness],
    }


def _sanitize_cypher_string(s: str) -> str:
    """Escape double-quotes and backslashes for Cypher string literals."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    # Remove newlines that would break inline statements
    s = re.sub(r'[\r\n]+', ' ', s)
    return s.strip()


def to_neo4j_cypher(triples: List[dict], path: str, noise_filter: bool = True, normalize: bool = True):
    """
    Write Neo4j Cypher MERGE statements for all triples to the given file path.

    Format per triple:
        MERGE (c:Concept {name: "..."})
        MERGE (e:Concept {name: "..."})
        MERGE (c)-[:CAUSAL_RELATION {type: "..."}]->(e);
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    lines = [
        "// Neo4j Cypher import — NTSB Causal Knowledge Graph",
        "// Generated by src/knowledge_graph.py",
        "",
    ]

    for triple in triples:
        cause = str(triple.get('cause', '')).strip()
        effect = str(triple.get('effect', '')).strip()
        relation = str(triple.get('relation', 'causes')).strip()

        if not cause or not effect:
            continue
        if normalize:
            cause  = _normalize_entity(cause)
            effect = _normalize_entity(effect)
        if noise_filter and (_is_noise(cause) or _is_noise(effect)):
            continue

        key = (cause, effect, relation.lower())
        if key in seen:
            continue
        seen.add(key)

        c_safe = _sanitize_cypher_string(cause)
        e_safe = _sanitize_cypher_string(effect)
        r_safe = _sanitize_cypher_string(relation)

        lines.append(f'MERGE (c:Concept {{name: "{c_safe}"}})')
        lines.append(f'MERGE (e:Concept {{name: "{e_safe}"}})')
        lines.append(f'MERGE (c)-[:CAUSAL_RELATION {{type: "{r_safe}"}}]->(e);')
        lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[KG] Wrote {len(seen)} Cypher statements to {output_path}")


def visualize_subgraph(G, top_n: int = 30, save_path: Optional[str] = None):
    """
    Plot a subgraph of the top_n highest-degree nodes using spring layout.
    """
    _check_networkx()

    if not PLOTTING_AVAILABLE:
        print("[KG] matplotlib not available — skipping visualization.")
        return

    if len(G) == 0:
        print("[KG] Graph is empty — skipping visualization.")
        return

    # Select top_n nodes by total degree
    top_nodes = sorted(
        G.nodes(),
        key=lambda n: G.degree(n),
        reverse=True,
    )[:top_n]

    subgraph = G.subgraph(top_nodes)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(subgraph, seed=42, k=1.5)

    # Node sizes proportional to degree
    node_sizes = [300 + 100 * subgraph.degree(n) for n in subgraph.nodes()]

    # Color nodes by type
    node_colors = [
        '#e74c3c' if subgraph.nodes[n].get('type') == 'cause_node' else '#3498db'
        for n in subgraph.nodes()
    ]

    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.85,
    )
    nx.draw_networkx_edges(
        subgraph, pos, ax=ax,
        edge_color='#95a5a6',
        arrows=True,
        arrowsize=15,
        width=1.2,
        alpha=0.6,
    )

    # Label only the highest-degree nodes to avoid clutter
    top_label_nodes = set(sorted(subgraph.nodes(), key=lambda n: subgraph.degree(n), reverse=True)[:15])
    labels = {n: (n[:30] + '…' if len(n) > 30 else n) for n in top_label_nodes}
    nx.draw_networkx_labels(subgraph, pos, labels=labels, ax=ax, font_size=7)

    ax.set_title(f'Causal Knowledge Graph — Top {top_n} Nodes', fontsize=13)
    ax.axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Cause node'),
        Patch(facecolor='#3498db', label='Effect node'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[KG] Saved knowledge graph visualization to {save_path}")

    plt.close(fig)
