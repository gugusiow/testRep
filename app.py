# app.py
from flask import Flask, request, jsonify

app = Flask(__name__)
app.url_map.strict_slashes = False

def find_extra_channels(edges):
    """
    edges: list of {"spy1": str, "spy2": str} for one undirected network
    returns: list of edges (same shape) that are NOT bridges (safe to cut)
    """
    # Build adjacency and keep original indices for each (u,v)
    adj = {}
    undirected_edges = []  # list of (u, v) with u != v, normalized by original order
    for i, e in enumerate(edges):
        u, v = e["spy1"], e["spy2"]
        if u == v:
            # self-loops are always non-bridges (they form a cycle of length 1)
            undirected_edges.append((u, v))
            continue
        adj.setdefault(u, []).append((v, i))
        adj.setdefault(v, []).append((u, i))
        undirected_edges.append((u, v))

    # Handle multi-edges: if multiple identical undirected edges exist between u and v,
    # then none of them is a bridge. We'll detect multiplicity over frozenset({u,v}).
    multiplicity = {}
    for (u, v) in undirected_edges:
        key = (u, v) if u == v else frozenset((u, v))
        multiplicity[key] = multiplicity.get(key, 0) + 1

    time = 0
    tin = {}
    low = {}
    visited = set()
    parent = {}
    bridge_edge_ids = set()  # set of original edge indices that are bridges

    def dfs(u):
        nonlocal time
        visited.add(u)
        time += 1
        tin[u] = low[u] = time
        for (v, eid) in adj.get(u, []):
            if v == parent.get(u):
                continue
            if v in visited:
                # back edge
                low[u] = min(low[u], tin[v])
            else:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                # Tree edge u-v is a bridge iff low[v] > tin[u] AND not a multi-edge
                # (Parallel edges mean the pair is never a bridge.)
                key = frozenset((u, v))
                if low[v] > tin[u] and multiplicity.get(key, 0) == 1:
                    bridge_edge_ids.add(eid)

    # Run DFS from all components
    for node in adj.keys():
        if node not in visited:
            parent[node] = None
            dfs(node)

    # Everything not in bridge_edge_ids is a non-bridge (extra channel)
    extra = []
    for i, e in enumerate(edges):
        u, v = e["spy1"], e["spy2"]
        if u == v:
            # self-loop -> always extra
            extra.append({"spy1": u, "spy2": v})
        else:
            if i not in bridge_edge_ids:
                extra.append({"spy1": u, "spy2": v})
    return extra

@app.post("/investigate")
def investigate():
    data = request.get_json(silent=True)
    if not data or "networks" not in data or not isinstance(data["networks"], list):
        return jsonify({"error": "Invalid input; expected {'networks': [...]}"}), 400

    result = {"networks": []}
    for net in data["networks"]:
        network_id = net.get("networkId")
        edges = net.get("network", [])
        if not isinstance(edges, list):
            return jsonify({"error": f"Invalid 'network' for networkId={network_id}"}), 400
        extra = find_extra_channels(edges)
        result["networks"].append({
            "networkId": network_id,
            "extraChannels": extra
        })

    return jsonify(result), 200

if __name__ == "__main__":
    # Default to port 8000 for local testing
    app.run(host="0.0.0.0", port=8000)
