from flask import Flask, request, jsonify
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any
from math import log, exp

app = Flask(__name__)

@app.route('/payload_crackme', methods=['GET'])
def get_payload():
    return "111-1111111"

@app.route('/payload_sqlinject', methods=['GET'])
def get_sql():
    return "Alice'; UPDATE salary SET salary=999999 WHERE name='Alice'; --"

@app.route('/payload_stack', methods=['GET'])
def get_stack():
    return "congratulations!_you_got_the_flag!"

@app.route('/chasetheflag', methods=['POST'])
def chase_flags():
    flags = {
        "challenge1": "your_flag_1",
        "challenge2": "your_flag_2",
        "challenge3": "your_flag_3",
        "challenge4": "your_flag_4",
        "challenge5": "your_flag_5"
    }
    return jsonify(flags), 201

####### mages gambit start
def earliest_clear_time(intel: List[List[int]], reserve: int, stamina: int) -> int:
    waves: List[Tuple[int, int]] = [(f, c) for f, c in intel]
    n = len(waves)
    if any(c > reserve for _, c in waves):
        return -1

    @lru_cache(maxsize=None)
    def dp(i: int, mana: int, stam: int, last_front: Optional[int]) -> int:
        if i == n:
            return 10  # final cooldown
        front, cost = waves[i]
        same = (last_front == front)
        can_cast = (cost <= mana) and (stam > 0)
        best = float('inf')
        if can_cast:
            time_cost = 0 if same else 10
            best = min(best, time_cost + dp(i + 1, mana - cost, stam - 1, front))
        if not can_cast:
            best = min(best, 10 + dp(i, reserve, stamina, None))
        return best

    return dp(0, reserve, stamina, None)

@app.route('/the-mages-gambit', methods=['POST'])
def get_gambit():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid or missing JSON"}), 400

    if not isinstance(payload, list):
        return jsonify({"error": "Top-level JSON must be a list of scenario objects"}), 400

    results: List[Dict[str, Any]] = []
    required_fields = ["intel", "reserve", "stamina", "fronts"]

    for idx, scenario in enumerate(payload):
        if not isinstance(scenario, dict):
            return jsonify({"error": f"Scenario at index {idx} is not an object"}), 400
        missing = [f for f in required_fields if f not in scenario]
        if missing:
            return jsonify({"error": f"Scenario {idx} missing fields: {', '.join(missing)}"}), 400

        intel = scenario["intel"]
        reserve = scenario["reserve"]
        stamina = scenario["stamina"]
        fronts = scenario["fronts"]  # currently unused but validated

        # Validations
        if not isinstance(intel, list) or any(not isinstance(x, list) or len(x) != 2 for x in intel):
            return jsonify({"error": f"Scenario {idx}: intel must be list of [front, mana_cost]"}), 400
        if not all(isinstance(f, int) and isinstance(c, int) and f >= 0 and c > 0 for f, c in intel):
            return jsonify({"error": f"Scenario {idx}: each intel entry must be positive integers [front>=0, mana_cost>0]"}), 400
        if not (isinstance(reserve, int) and reserve > 0):
            return jsonify({"error": f"Scenario {idx}: reserve must be positive int"}), 400
        if not (isinstance(stamina, int) and stamina > 0):
            return jsonify({"error": f"Scenario {idx}: stamina must be positive int"}), 400
        if not (isinstance(fronts, int) and fronts >= 0):
            return jsonify({"error": f"Scenario {idx}: fronts must be non-negative int"}), 400

        time_needed = earliest_clear_time(intel, reserve, stamina)
        results.append({"time": time_needed})

    return jsonify(results)
###### mages gambit end

###### ink archive start
# def find_gain_cycle(ratios: List[List[float]], goods: List[str], mode: str):
#     n = len(goods)
#     adj: Dict[int, List[Tuple[int,float]]] = {i: [] for i in range(n)}
#     for r in ratios:
#         if len(r) != 3:
#             continue
#         u, v, val = int(r[0]), int(r[1]), float(r[2])
#         if 0 <= u < n and 0 <= v < n and val > 0:
#             adj[u].append((v, val))

#     if mode == 'shortest_positive':
#         best_cycle: List[int] = []
#         best_len = math.inf
#         best_product = 1.0
#         visited = [False]*n

#         def dfs_short(start: int, node: int, product: float, path: List[int]):
#             nonlocal best_cycle, best_len, best_product
#             if len(path) > best_len:  # prune by current best length
#                 return
#             for nxt, r in adj[node]:
#                 if nxt == start and len(path) >= 2:
#                     total = product * r
#                     edges = len(path)  # number of edges so far; closing edge makes len(path)+1 nodes
#                     if total > 1.0 + 1e-12 and (edges < best_len or (edges == best_len and total > best_product + 1e-12)):
#                         best_len = edges
#                         best_product = total
#                         best_cycle = path + [nxt]
#                     continue
#                 if not visited[nxt] and len(path) + 1 <= n:  # simple cycle constraint
#                     # optimistic pruning: even if all remaining multipliers were max outgoing from nxt, 
#                     # skip (light optimization omitted for simplicity for small n)
#                     visited[nxt] = True
#                     dfs_short(start, nxt, product * r, path + [nxt])
#                     visited[nxt] = False

#         for s in range(n):
#             visited[s] = True
#             dfs_short(s, s, 1.0, [s])
#             visited[s] = False
#         if best_product <= 1.0 or not best_cycle:
#             return [], 0.0
#         return [goods[i] for i in best_cycle], (best_product - 1.0) * 100.0

#     # mode == 'max_gain'
#     best_product = 1.0
#     best_cycle: List[int] = []
#     max_out = [max((w for _, w in adj[i]), default=1.0) for i in range(n)]
#     visited = [False]*n

#     def upper_bound(prod: float, remaining: int) -> float:
#         if remaining <= 0: return prod
#         gm = max(max_out) if max_out else 1.0
#         return prod * (gm ** remaining)

#     def dfs_max(start: int, node: int, product: float, path: List[int]):
#         nonlocal best_product, best_cycle
#         for nxt, r in adj[node]:
#             if nxt == start and len(path) >= 2:
#                 total = product * r
#                 if total > best_product + 1e-12:
#                     best_product = total
#                     best_cycle = path + [nxt]
#                 continue
#             if not visited[nxt] and len(path) + 1 < n:
#                 est = upper_bound(product * r, n - (len(path) + 1))
#                 if est <= best_product + 1e-12:
#                     continue
#                 visited[nxt] = True
#                 dfs_max(start, nxt, product * r, path + [nxt])
#                 visited[nxt] = False

#     for s in range(n):
#         visited[s] = True
#         dfs_max(s, s, 1.0, [s])
#         visited[s] = False

#     if best_product <= 1.0 or not best_cycle:
#         return [], 0.0
#     return [goods[i] for i in best_cycle], (best_product - 1.0) * 100.0


# @app.route('/The-Ink-Archive', methods=['POST'])
# def ink_archive():
#     try:
#         payload = request.get_json(force=True)
#     except Exception:
#         return jsonify({"error":"Invalid or missing JSON"}), 400
#     if not isinstance(payload, list):
#         return jsonify({"error":"Top-level JSON must be a list"}), 400
#     results = []
#     for idx, scenario in enumerate(payload):
#         if not isinstance(scenario, dict):
#             return jsonify({"error":f"Scenario {idx} not an object"}), 400
#         if 'ratios' not in scenario or 'goods' not in scenario:
#             return jsonify({"error":f"Scenario {idx} missing 'ratios' or 'goods'"}), 400
#         ratios = scenario['ratios']
#         goods = scenario['goods']
#         if not isinstance(goods, list) or not all(isinstance(g, str) for g in goods):
#             return jsonify({"error":f"Scenario {idx} goods invalid"}), 400
#         if not isinstance(ratios, list):
#             return jsonify({"error":f"Scenario {idx} ratios invalid"}), 400
#         mode = 'shortest_positive' if idx == 0 else 'max_gain'
#         path, gain = find_gain_cycle(ratios, goods, mode)
#         results.append({"path": path, "gain": gain})
#     return jsonify(results)

from math import log
EPS = 1e-9  # numerical tolerance; adjust if needed

def _reduce_to_simple_cycle(closed_nodes):
    """
    closed_nodes: like [n0, n1, ..., nk, n0]
    Returns a closed simple cycle [m0, m1, ..., mr, m0] with no repeated nodes except closure.
    If multiple repeats exist, this keeps the innermost simple loop.
    """
    # Map node -> last index seen
    last_pos = {}
    for i, u in enumerate(closed_nodes):
        last_pos[u] = i

    # Walk and cut to the last repeated segment
    seen = {}
    start = 0
    end = len(closed_nodes) - 1  # last element equals first
    for i, u in enumerate(closed_nodes):
        if u in seen:
            # We found a loop from seen[u] to i
            start = seen[u]
            end = i
        seen[u] = i

    simple = closed_nodes[start:end+1]
    # Ensure closure
    if simple[0] != simple[-1]:
        simple.append(simple[0])
    return simple

def find_profitable_cycles(num_nodes, edges):
    # adjacency map
    adj_rate = {(int(u), int(v)): float(r) for u, v, r in edges if r > 0}

    weighted_edges = []
    for u, v, r in edges:
        r = float(r)
        if r > 0:
            weighted_edges.append((int(u), int(v), -log(r)))

    all_cycles = []
    seen_canon = set()

    def canonicalize_cycle(cyc):
        core = cyc[:-1]
        n = len(core)
        if n <= 1:
            return None
        # minimal rotation and reversed version
        min_pos = min(range(n), key=lambda i: core[i])
        rot = core[min_pos:] + core[:min_pos]
        rev = list(reversed(core))
        min_pos_r = min(range(n), key=lambda i: rev[i])
        rot_r = rev[min_pos_r:] + rev[:min_pos_r]
        return tuple(rot) if rot < rot_r else tuple(rot_r)

    for src in range(num_nodes):
        dist = [0.0] * num_nodes
        pred = [-1] * num_nodes

        # Relax
        for _ in range(num_nodes - 1):
            updated = False
            for u, v, w in weighted_edges:
                if dist[u] + w < dist[v] - 1e-15:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    updated = True
            if not updated:
                break

        # Detect neg cycle
        cycle_entry = -1
        for u, v, w in weighted_edges:
            if dist[u] + w < dist[v] - 1e-15:
                pred[v] = u
                cycle_entry = v
                break
        if cycle_entry == -1:
            continue

        # Move inside cycle
        x = cycle_entry
        for _ in range(num_nodes):
            x = pred[x]
            if x == -1:
                break
        if x == -1:
            continue

        # Collect raw closed walk
        cur = x
        seen_local = set()
        path = []
        while cur not in seen_local and cur != -1:
            seen_local.add(cur)
            path.append(cur)
            cur = pred[cur]
        if cur == -1:
            continue

        # Build closed cycle [start ... start]
        start_idx = path.index(cur)
        raw_cycle = path[start_idx:][::-1]  # reverse to get forward order
        raw_cycle.append(raw_cycle[0])

        # Reduce to simple cycle
        simple_cycle = _reduce_to_simple_cycle(raw_cycle)

        # Canonicalize and deduplicate
        canon = canonicalize_cycle(simple_cycle)
        if canon is None or canon in seen_canon:
            continue

        # Verify directed edges and compute product
        prod = 1.0
        valid = True
        for i in range(len(simple_cycle) - 1):
            u = simple_cycle[i]
            v = simple_cycle[i + 1]
            r = adj_rate.get((u, v))
            if r is None:
                valid = False
                break
            prod *= r
        if not valid:
            continue

        if prod > 1.0 + EPS:
            seen_canon.add(canon)
            all_cycles.append((simple_cycle, prod))

    return all_cycles

def pick_first_profitable_cycle(cycles):
    if not cycles:
        return None
    # Prefer shortest simple cycle first (to avoid extra nodes like Blue Moss),
    # then higher gain, then deterministic tie-break.
    return sorted(cycles, key=lambda cp: (len(cp[0]), -cp[1], tuple(cp[0])))[0]

def pick_max_gain_cycle(cycles):
    """
    For challenge 2: just pick the absolute max gain.
    """
    if not cycles:
        return None
    return max(cycles, key=lambda cp: cp[1])

def format_response_cycle(cycle_nodes, prod_gain, goods):
    # Map node indices to names for path, ensure names list closes the loop
    path_names = [goods[i] for i in cycle_nodes]
    gain_percent = (prod_gain - 1.0) * 100.0
    return {"path": path_names, "gain": gain_percent}

@app.route("/The-Ink-Archive", methods=['POST'])
def solve():
    payload = request.get_json(force=True)
    if not isinstance(payload, list) or len(payload) != 2:
        return jsonify({"error": "Body must be a JSON array with 2 items"}), 400

    results = []

    # Process each challenge
    for idx, item in enumerate(payload):
        goods = item.get("goods", [])
        ratios = item.get("ratios", [])
        if not goods or not ratios:
            return jsonify({"error": f"Item {idx}: 'goods' and 'ratios' required"}), 400

        n = len(goods)
        edges = []
        for trip in ratios:
            if not (isinstance(trip, list) and len(trip) == 3):
                return jsonify({"error": f"Item {idx}: each ratio must be [u, v, rate]"}), 400
            u, v, r = trip
            u = int(u)
            v = int(v)
            r = float(r)
            if u < 0 or u >= n or v < 0 or v >= n:
                return jsonify({"error": f"Item {idx}: u,v out of range"}), 400
            edges.append((u, v, r))

        cycles = find_profitable_cycles(n, edges)

        if idx == 0:
            chosen = pick_first_profitable_cycle(cycles)
        else:
            chosen = pick_max_gain_cycle(cycles)

        if not chosen:
            # No arbitrage found
            results.append({"path": [], "gain": 0.0})
            continue

        cycle_nodes, prod = chosen
        results.append(format_response_cycle(cycle_nodes, prod, goods))

    return jsonify(results), 200

###### ink archive end

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)      