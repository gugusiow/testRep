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
def find_best_gain_cycle(ratios: List[List[float]], goods: List[str]):
    n = len(goods)
    # Build edge list (u,v,ratio)
    edges: List[Tuple[int,int,float]] = []
    ratio_lookup = {}
    for r in ratios:
        if len(r) != 3:
            continue
        u, v, val = int(r[0]), int(r[1]), float(r[2])
        if u < 0 or u >= n or v < 0 or v >= n:
            continue
        edges.append((u, v, val))
        ratio_lookup[(u, v)] = val

    best_product = 1.0
    best_cycle: List[int] = []

    # Bellman-Ford style search for negative cycles on -log(r)
    for start in range(n):
        dist = [math.inf]*n
        pred: List[Optional[int]] = [None]*n
        dist[start] = 0.0
        updated_node = None
        for _ in range(n):
            updated_node = None
            for u, v, r in edges:
                w = -math.log(r)
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    pred[v] = u
                    updated_node = v
            if updated_node is None:
                break
        # If updated on nth iteration -> negative cycle reachable from start
        if updated_node is None:
            continue
        # Move into cycle
        x = updated_node
        for _ in range(n):
            if x is None: break
            x = pred[x]
        if x is None:
            continue
        # Extract cycle
        cycle = []
        cur = x
        while True:
            cycle.append(cur)
            cur = pred[cur]  # type: ignore
            if cur is None or cur == x:
                break
        if not cycle or cur is None:
            continue
        cycle.reverse()  # order in traversal direction
        # Ensure cycle is closed sequentially; compute product
        product = 1.0
        for i in range(len(cycle)):
            a = cycle[i]
            b = cycle[(i+1) % len(cycle)]
            r = ratio_lookup.get((a,b))
            if r is None:
                product = 1.0
                break
            product *= r
        if product > best_product + 1e-12:
            best_product = product
            best_cycle = cycle[:]

    if best_product <= 1.0 or not best_cycle:
        return [], 0.0

    # Rotate cycle to include starting node at both ends for output
    path_names = [goods[i] for i in best_cycle] + [goods[best_cycle[0]]]
    gain = (best_product - 1.0) * 100.0
    return path_names, gain


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
        path, gain = find_best_gain_cycle(ratios, goods)
        results.append({"path": path, "gain": gain})
    return jsonify(results)

###### ink archive end

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)      