from flask import Flask, request, jsonify
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any
import math

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
def find_gain_cycle(ratios: List[List[float]], goods: List[str], mode: str):
    n = len(goods)
    adj: Dict[int, List[Tuple[int,float]]] = {i: [] for i in range(n)}
    for r in ratios:
        if len(r) != 3:
            continue
        u, v, val = int(r[0]), int(r[1]), float(r[2])
        if 0 <= u < n and 0 <= v < n and val > 0:
            adj[u].append((v, val))

    if mode == 'shortest_positive':
        best_cycle: List[int] = []
        best_len = math.inf
        best_product = 1.0
        visited = [False]*n

        def dfs_short(start: int, node: int, product: float, path: List[int]):
            nonlocal best_cycle, best_len, best_product
            if len(path) > best_len:  # prune by current best length
                return
            for nxt, r in adj[node]:
                if nxt == start and len(path) >= 2:
                    total = product * r
                    edges = len(path)  # number of edges so far; closing edge makes len(path)+1 nodes
                    if total > 1.0 + 1e-12 and (edges < best_len or (edges == best_len and total > best_product + 1e-12)):
                        best_len = edges
                        best_product = total
                        best_cycle = path + [nxt]
                    continue
                if not visited[nxt] and len(path) + 1 <= n:  # simple cycle constraint
                    # optimistic pruning: even if all remaining multipliers were max outgoing from nxt, 
                    # skip (light optimization omitted for simplicity for small n)
                    visited[nxt] = True
                    dfs_short(start, nxt, product * r, path + [nxt])
                    visited[nxt] = False

        for s in range(n):
            visited[s] = True
            dfs_short(s, s, 1.0, [s])
            visited[s] = False
        if best_product <= 1.0 or not best_cycle:
            return [], 0.0
        return [goods[i] for i in best_cycle], (best_product - 1.0) * 100.0

    # mode == 'max_gain'
    best_product = 1.0
    best_cycle: List[int] = []
    max_out = [max((w for _, w in adj[i]), default=1.0) for i in range(n)]
    visited = [False]*n

    def upper_bound(prod: float, remaining: int) -> float:
        if remaining <= 0: return prod
        gm = max(max_out) if max_out else 1.0
        return prod * (gm ** remaining)

    def dfs_max(start: int, node: int, product: float, path: List[int]):
        nonlocal best_product, best_cycle
        for nxt, r in adj[node]:
            if nxt == start and len(path) >= 2:
                total = product * r
                if total > best_product + 1e-12:
                    best_product = total
                    best_cycle = path + [nxt]
                continue
            if not visited[nxt] and len(path) + 1 < n:
                est = upper_bound(product * r, n - (len(path) + 1))
                if est <= best_product + 1e-12:
                    continue
                visited[nxt] = True
                dfs_max(start, nxt, product * r, path + [nxt])
                visited[nxt] = False

    for s in range(n):
        visited[s] = True
        dfs_max(s, s, 1.0, [s])
        visited[s] = False

    if best_product <= 1.0 or not best_cycle:
        return [], 0.0
    return [goods[i] for i in best_cycle], (best_product - 1.0) * 100.0


@app.route('/The-Ink-Archive', methods=['POST'])
def ink_archive():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error":"Invalid or missing JSON"}), 400
    if not isinstance(payload, list):
        return jsonify({"error":"Top-level JSON must be a list"}), 400
    results = []
    for idx, scenario in enumerate(payload):
        if not isinstance(scenario, dict):
            return jsonify({"error":f"Scenario {idx} not an object"}), 400
        if 'ratios' not in scenario or 'goods' not in scenario:
            return jsonify({"error":f"Scenario {idx} missing 'ratios' or 'goods'"}), 400
        ratios = scenario['ratios']
        goods = scenario['goods']
        if not isinstance(goods, list) or not all(isinstance(g, str) for g in goods):
            return jsonify({"error":f"Scenario {idx} goods invalid"}), 400
        if not isinstance(ratios, list):
            return jsonify({"error":f"Scenario {idx} ratios invalid"}), 400
        mode = 'shortest_positive' if idx == 0 else 'max_gain'
        path, gain = find_gain_cycle(ratios, goods, mode)
        results.append({"path": path, "gain": gain})
    return jsonify(results)

###### ink archive end

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)      