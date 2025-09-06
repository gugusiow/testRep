# app.py
from flask import Flask, request, jsonify
import os
from collections import defaultdict

app = Flask(__name__)
app.url_map.strict_slashes = False

def extra_channels_tarjan(edges):
    """
    edges: list of dicts [{"spy1": str, "spy2": str}, ...]
    returns: list of dicts (same order/shape) that are NOT bridges (i.e., safe to cut)
    Notes:
      - Self-loops are always non-bridges.
      - Parallel edges between the same two nodes mean none of those parallel edges is a bridge.
    """
    n_edges = len(edges)

    # Map edge id -> (u, v) with undirected normalization key for multiplicity
    u = [None] * n_edges
    v = [None] * n_edges
    key = [None] * n_edges

    # Build adjacency with edge ids (keep MULTI-EDGES)
    adj = defaultdict(list)
    multiplicity = defaultdict(int)

    for i, e in enumerate(edges):
        a, b = e["spy1"], e["spy2"]
        u[i], v[i] = a, b
        if a == b:
            # self-loop: treat specially later
            key[i] = (a, a)  # not using frozenset to avoid type mix; self-loop unique
            multiplicity[key[i]] += 1
            # no need to add to adjacency (doesn't affect connectivity/bridges)
            continue
        k = frozenset((a, b))
        key[i] = k
        multiplicity[k] += 1
        adj[a].append((b, i))
        adj[b].append((a, i))

    # Tarjan bridge-finding
    time = 0
    tin = {}
    low = {}
    visited = set()
    parent_edge = {}  # node -> incoming edge id from parent (to skip immediate back)

    bridges = set()

    def dfs(start):
        nonlocal time
        stack = [(start, None, iter(adj[start]))]
        visited.add(start)
        time += 1
        tin[start] = low[start] = time
        parent_edge[start] = None

        while stack:
            node, parent, it = stack[-1]
            try:
                to, eid = next(it)
            except StopIteration:
                # On unwind, try to propagate low-link to parent
                stack.pop()
                if parent is not None:
                    # parent-edge id
                    peid = parent_edge[node]
                    # node was discovered from parent
                    low[parent] = min(low[parent], low[node])
                    # Bridge check only applies to tree edges, and only if single-edge between endpoints
                    # (multi-edge pair can never be a bridge)
                    pair_key = frozenset((parent, node))
                    if low[node] > tin[parent] and multiplicity.get(pair_key, 0) == 1:
                        # mark the unique tree edge (parent-node) as a bridge;
                        # we must record *that edge id*. It's the one that led from parent->node.
                        bridges.add(peid)
                continue

            # Skip the edge we came from (by edge id), but allow parallel back-edges
            if eid == parent_edge.get(node):
                continue

            if to not in visited:
                visited.add(to)
                parent_edge[to] = eid
                time += 1
                tin[to] = low[to] = time
                stack.append((to, node, iter(adj[to])))
            else:
                # Back/parallel edge: update low
                low[node] = min(low[node], tin[to])

    # Run DFS on all components
    for node in list(adj.keys()):
        if node not in visited:
            dfs(node)

    # Build result: any edge that is NOT a bridge is an extra channel.
    # Self-loops are always extra. For parallel edges between the same two nodes,
    # none of those parallel edges is a bridge, by definition above.
    extra = []
    for i in range(n_edges):
        a, b = u[i], v[i]
        if a == b:
            extra.append({"spy1": a, "spy2": b})
            continue
        k = key[i]
        if multiplicity[k] > 1:
            extra.append({"spy1": a, "spy2": b})
        else:
            if i not in bridges:
                extra.append({"spy1": a, "spy2": b})
    return extra

@app.post("/investigate")
def investigate():
    # Be permissive with JSON parsing
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Body is not valid JSON"}), 400

    # Accept either {"networks":[...]} or a raw list
    if isinstance(data, dict):
        networks = data.get("networks", [])
        if not isinstance(networks, list):
            return jsonify({"error": "Invalid 'networks' format; must be a list"}), 400
    elif isinstance(data, list):
        networks = data
    else:
        return jsonify({"error": "Invalid input; expected an object with 'networks' or a list"}), 400

    out = {"networks": []}
    for net in networks:
        if not isinstance(net, dict):
            return jsonify({"error": "Each network must be an object"}), 400
        nid = net.get("networkId")
        edges = net.get("network")
        if edges is None or not isinstance(edges, list):
            return jsonify({"error": f"networkId={nid}: 'network' must be a list"}), 400

        extra = extra_channels_tarjan(edges)
        out["networks"].append({
            "networkId": nid,
            "extraChannels": extra
        })
    return jsonify(out), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
