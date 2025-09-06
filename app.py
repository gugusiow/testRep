from math import inf
from bisect import bisect_right
from heapq import heappush, heappop
from flask import Flask, request, jsonify

# ------------- helpers -------------
def dijkstra_multi_target(adj, src, target_set):
    """
    Single-source Dijkstra from src, but we only care about distances
    to nodes in target_set. We stop early when all targets are settled.
    Returns: dict {node: dist}
    """
    dist = {}
    pq = [(0, src)]
    needed = set(target_set)  # copy
    while pq and needed:
        d, u = heappop(pq)
        if u in dist:
            continue
        dist[u] = d
        if u in needed:
            needed.remove(u)
        for v, w in adj.get(u, ()):
            if v not in dist:
                heappush(pq, (d + w, v))
    return dist

def build_pairwise_dist(adj, important_nodes):
    """
    Compute pairwise shortest path fees between stations in 'important_nodes'
    using early-stopping Dijkstra. Returns a dict-of-dicts d[u][v].
    """
    important = sorted(important_nodes)
    pair = {u: {} for u in important}
    target_set = set(important)
    for u in important:
        du = dijkstra_multi_target(adj, u, target_set)
        # keep only important nodes
        for v in important:
            pair[u][v] = du.get(v, inf)
    return pair

# ------------- solver -------------
def solve(input_obj):
    tasks = input_obj.get("tasks", [])
    subway = input_obj.get("subway", [])
    s0_id = input_obj.get("starting_station")

    # No tasks -> trivial
    if not tasks:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # Build adjacency list (undirected)
    adj = {}
    def add_edge(a, b, w):
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))

    stations_needed = {s0_id}
    for t in tasks:
        stations_needed.add(t["station"])
    for r in subway:
        a, b = r["connection"]; w = r["fee"]
        add_edge(a, b, w)

    # All-pairs distances for the *needed* stations only
    pair = build_pairwise_dist(adj, stations_needed)
    if any(pair[s0_id][t["station"]] == inf for t in tasks):
        # Graph claimed connected, but just in case
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # Normalize tasks and sort by end time
    T = [{
        "name": t["name"],
        "start": int(t["start"]),
        "end":   int(t["end"]),
        "station": t["station"],
        "score": int(t["score"]),
    } for t in tasks]

    # Sort by (end, start, name) → stable & deterministic
    T.sort(key=lambda x: (x["end"], x["start"], x["name"]))
    m = len(T)

    # Precompute prev non-overlap via binary search on ends
    ends = [t["end"] for t in T]
    prev_idx = []
    for i in range(m):
        # last j with T[j].end <= T[i].start
        j = bisect_right(ends, T[i]["start"]) - 1
        prev_idx.append(j)

    # DP: best (score, minFeeToEnd) for schedules ending at i
    bestScore = [0]*m
    minFeeToEnd = [inf]*m
    parent = [-1]*m

    for i in range(m):
        si = T[i]["station"]
        sc_i = T[i]["score"]

        # case: start directly from s0 to task i
        best_s = sc_i
        best_f = pair[s0_id][si]
        best_p = -1

        # case: extend from prev non-overlapping task j = prev_idx[i]
        j = prev_idx[i]
        # We may have multiple candidates with same end but different starts;
        # walk left across tasks that still end <= T[i].start to consider all.
        k = j
        while k >= 0 and T[k]["end"] <= T[i]["start"]:
            cand_s = bestScore[k] + sc_i
            cand_f = minFeeToEnd[k] + pair[T[k]["station"]][si]
            if (cand_s > best_s) or (cand_s == best_s and cand_f < best_f):
                best_s, best_f, best_p = cand_s, cand_f, k
            k -= 1

        bestScore[i] = best_s
        minFeeToEnd[i] = best_f
        parent[i] = best_p

    # Choose best ending i with minimal return fee to s0
    ansScore = 0
    ansFee = 0
    last = -1
    for i in range(m):
        totalScore = bestScore[i]
        totalFee = minFeeToEnd[i] + pair[T[i]["station"]][s0_id]
        if (totalScore > ansScore) or (totalScore == ansScore and totalFee < ansFee):
            ansScore, ansFee, last = totalScore, totalFee, i

    if ansScore == 0:
        # Either no feasible or all zero-score (scores >=1 per constraints, but safe)
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # Reconstruct tasks
    chosen = []
    cur = last
    while cur != -1:
        chosen.append(T[cur])
        cur = parent[cur]

    # API requires schedule sorted by starting time (tie: by end, then name)
    chosen.sort(key=lambda x: (x["start"], x["end"], x["name"]))
    schedule_names = [t["name"] for t in chosen]

    return {
        "max_score": ansScore,
        "min_fee": int(ansFee),
        "schedule": schedule_names,
    }

# ------------- Flask app -------------
app = Flask(__name__)

@app.get("/")
def health():
    return "Princess Diaries API is running."

@app.post("/princess-diaries")
def princess_diaries():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid input: object expected."}), 400
        required = {"tasks", "subway", "starting_station"}
        if not required.issubset(payload):
            return jsonify({"error": "Invalid input shape."}), 400
        result = solve(payload)
        return jsonify(result)
    except Exception as e:
        # print(e)  # optionally log
        return jsonify({"error": "Invalid input or internal error."}), 400

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
