from __future__ import annotations
from flask import Flask, request, jsonify
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import math
import re
import requests
import numpy as np


app = Flask(__name__)

@app.route('/payload_crackme', methods=['GET'])
def get_payload():
    return "111-1111111"

@app.route('/payload_sqlinject', methods=['GET'])
def get_sql():
    return jsonify({
        "payload": "Alice'; UPDATE salary SET salary=999999 WHERE name='Alice'; --",
        "type": "sql_injection",
        "target": "name"
    })

@app.route('/payload_stack', methods=['GET'])
def get_stack():
    return "congratulations!_you_got_the_flag!"

@app.route('/chasetheflag', methods=['POST'])
def chase_flags():
    flags = {
        "challenge1": "UBS{}",
        "challenge2": "UBS{}",
        "challenge3": "UBS{}",
        "challenge4": "UBS{}",
        "challenge5": "UBS{16lt0tt13zm1es}"  # UBS{474owrgw8fbyy}
    }
    return jsonify(flags), 201

####### trading bot
BULL_TOKENS = {
    "bull","bullish","buy","long","up","rally","pump","moon","adopt","adoption",
    "approve","approval","etf","etfs","reserve","support","positive","surge",
    "breakout","all-time","ath","institutional","accumulate","buying","bid",
    "liquidity","halving","order","executive","legalize","legalizes","strategic",
    "fund","funding","backed","stimulus","qe","quantitative","easing","treasury"
}
BEAR_TOKENS = {
    "bear","bearish","sell","short","down","dump","ban","bans","restrict",
    "crackdown","negative","hacked","hack","rug","scam","lawsuit","probe",
    "investigation","fraud","liquidation","defaults","insolvency","shutdown",
    "outage","exploit","delay","rejection","reject","rejects"
}

SOURCE_WEIGHT = {
    "twitter": 1.0, "x": 1.0, "coindesk": 1.1, "cointelegraph": 1.05,
    "bloomberg": 1.15, "reuters": 1.15
}

TOKENIZER_RE = re.compile(r"https?://\S+|[^a-z0-9\-\+\#\.\$]+")

def tokenize(text: Optional[str]) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    txt = text.lower()
    txt = TOKENIZER_RE.sub(" ", txt)
    toks = [t for t in txt.split() if t]
    return toks

def safe_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        if math.isfinite(f):
            return f
        return None
    except Exception:
        return None

def valid_candle(c: Dict[str, Any]) -> bool:
    return all(
        safe_float(c.get(k)) is not None
        for k in ("open","high","low","close","volume","timestamp")
    )

def typical_price(c: Dict[str, Any]) -> Optional[float]:
    h = safe_float(c.get("high"))
    l = safe_float(c.get("low"))
    cl = safe_float(c.get("close"))
    if None in (h,l,cl):
        return None
    return (h + l + cl) / 3.0

def range_size(c: Dict[str, Any]) -> Optional[float]:
    h = safe_float(c.get("high"))
    l = safe_float(c.get("low"))
    if None in (h,l):
        return None
    return max(1e-9, h - l)

def wick_skew(c: Dict[str, Any]) -> Optional[float]:
    # Positive if bullish body with small upper wick; negative if bearish with long upper wick
    o = safe_float(c.get("open"))
    h = safe_float(c.get("high"))
    l = safe_float(c.get("low"))
    cl = safe_float(c.get("close"))
    if None in (o,h,l,cl):
        return None
    body = cl - o
    rng = max(1e-9, h - l)
    upper = h - max(o, cl)
    lower = min(o, cl) - l
    # body normalized plus lower-upper wick preference (lower wick bullish)
    return (body / rng) + (lower - upper) / rng

def pct_change(a: float, b: float) -> float:
    # percent change from a -> b
    if a == 0:
        return 0.0
    return (b - a) / abs(a)

def source_weight(src: Optional[str]) -> float:
    if not src or not isinstance(src, str):
        return 1.0
    key = src.strip().lower()
    return SOURCE_WEIGHT.get(key, 1.0)

# ----------------------------- Scoring -----------------------------

def sentiment_score(title: Optional[str], src: Optional[str]) -> float:
    toks = tokenize(title)
    bulls = sum(1 for t in toks if t in BULL_TOKENS)
    bears = sum(1 for t in toks if t in BEAR_TOKENS)
    raw = bulls - bears
    w = source_weight(src)
    # tanh squashing to keep impact bounded
    return math.tanh(raw * 0.8) * w

def momentum_score(prev: List[Dict[str, Any]], first_obs: Dict[str, Any]) -> float:
    # Use last two prev closes and first observation close
    if len(prev) < 2:
        return 0.0
    c1 = safe_float(prev[-2].get("close"))
    c2 = safe_float(prev[-1].get("close"))
    c3 = safe_float(first_obs.get("close"))
    if None in (c1,c2,c3):
        return 0.0
    m12 = pct_change(c1, c2)
    m23 = pct_change(c2, c3)
    # emphasize acceleration
    accel = m23 - m12
    # score = recent move + acceleration
    score = 0.6 * m23 + 0.4 * accel
    # clamp for robustness
    return max(-0.1, min(0.1, score))  # +/-10%

def volatility_score(prev: List[Dict[str, Any]], first_obs: Dict[str, Any]) -> float:
    # ATR-like: median range of last 3 prev + first obs vs price
    recent = (prev[-3:] if len(prev) >= 3 else prev[:]) + [first_obs]
    ranges = [range_size(c) for c in recent if valid_candle(c)]
    closes = [safe_float(c.get("close")) for c in recent if valid_candle(c)]
    ranges = [r for r in ranges if r is not None]
    closes = [c for c in closes if c is not None]
    if not ranges or not closes:
        return 0.0
    ranges_sorted = sorted(ranges)
    med = ranges_sorted[len(ranges_sorted)//2]
    px = closes[-1]
    rel = med / max(1e-6, px)
    # Convert to bounded contribution: higher vol increases conviction, not direction
    return max(0.0, min(0.08, rel))  # up to 8% conviction boost

def structure_score(prev: List[Dict[str, Any]], first_obs: Dict[str, Any]) -> float:
    # Wick skew of last previous and first observation
    comp = []
    if prev:
        s1 = wick_skew(prev[-1])
        if s1 is not None:
            comp.append(s1)
    s2 = wick_skew(first_obs)
    if s2 is not None:
        comp.append(s2)
    if not comp:
        return 0.0
    s = sum(comp) / len(comp)
    # bound
    return max(-0.12, min(0.12, s))  # wick skew can be strong, bound it

def build_signal(event: Dict[str, Any]) -> Tuple[float, str]:
    """
    Returns (confidence, decision) where decision is 'LONG' or 'SHORT'.
    Uses entry and exit prices as required by challenge.
    """
    obs = event.get("observation_candles") or []
    if not obs or not isinstance(obs, list) or len(obs) < 1:
        return (0.0, "SHORT")
    entry = safe_float(obs[0].get("close"))
    exit = safe_float(obs[-1].get("close"))
    if entry is None or exit is None:
        return (0.0, "SHORT")
    change = exit - entry
    decision = "LONG" if change > 0 else "SHORT"
    confidence = abs(change)
    return (confidence, decision)

@app.route("/trading-bot", methods=["POST"])
def trading_bot():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array of news events"}), 400

    scored: List[Tuple[float, int, str]] = []  # (confidence, id, decision)
    for ev in data:
        if not isinstance(ev, dict):
            continue
        eid = ev.get("id")
        if not isinstance(eid, int):
            try:
                eid = int(eid)
            except Exception:
                continue
        confidence, decision = build_signal(ev)
        scored.append((confidence, eid, decision))

    if not scored:
        return jsonify([]), 200

    # Sort by confidence (absolute price change) desc, then id asc
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Pick top 50
    out = [{"id": eid, "decision": dec} for (_conf, eid, dec) in scored[:50]]
    return jsonify(out), 200

###### trading bot end

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
            # First challenge: ensure we haven't missed shorter cycles by doing a full enumeration (n is small here)
            adj: Dict[int, List[Tuple[int,float]]] = {i: [] for i in range(n)}
            for u,v,r in edges:
                adj[u].append((v,r))

            enum_cycles = []  # list of (cycle_nodes, product)
            visited = [False]*n

            def dfs(start, node, product, path):
                # path includes node indices, without closure
                for nxt, rate in adj[node]:
                    if nxt == start and len(path) >= 2:  # found cycle with >=3 distinct nodes incl. start
                        cyc_nodes = path + [nxt]
                        enum_cycles.append((cyc_nodes, product*rate))
                        continue
                    if not visited[nxt] and len(path) + 1 < n:  # simple cycle constraint
                        visited[nxt] = True
                        dfs(start, nxt, product*rate, path + [nxt])
                        visited[nxt] = False

            for s in range(n):
                visited[s] = True
                dfs(s, s, 1.0, [s])
                visited[s] = False

            # Merge with cycles detected via Bellman-Ford (avoid duplicates by canonical form)
            def canon(c):
                core = c[:-1]
                m = len(core)
                if m==0: return ()
                rots = [tuple(core[i:]+core[:i]) for i in range(m)]
                return min(rots)
            seen_local = {canon(c): (c,p) for c,p in cycles}
            for c,p in enum_cycles:
                key = canon(c)
                if key not in seen_local:
                    seen_local[key] = (c,p)
            # Filter profitable
            all_pos = [(c,p) for c,p in seen_local.values() if p > 1+EPS]
            if not all_pos:
                chosen = None
            else:
                # Group by edge length (len(c)-1), take minimal length group, then max product inside group
                min_len = min(len(c)-1 for c,_ in all_pos)
                cand = [(c,p) for c,p in all_pos if (len(c)-1)==min_len]
                chosen = max(cand, key=lambda cp: cp[1])
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

##### fog of wall bfs?

game_states = {}    # init game state

class GameState:
    def __init__(self, test_case):
        self.grid_size = test_case['length_of_grid']
        self.total_walls = test_case['num_of_walls']
        self.crows = {crow['id']: {'x': crow['x'], 'y': crow['y'], 'visited': set()} for crow in test_case['crows']}
        
        # Initialize the global map
        self.global_map = {}
        self.known_walls = set()
        self.unknown_cells = set()
        
        # Initialize all cells as unknown
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.unknown_cells.add((x, y))
        
        # Mark starting positions as empty
        for crow_id, crow in self.crows.items():
            pos = (crow['x'], crow['y'])
            self.global_map[pos] = '_'
            if pos in self.unknown_cells:
                self.unknown_cells.remove(pos)
            crow['visited'].add(pos)
        
        # Strategy variables
        self.current_crow_index = 0
        self.crow_ids = list(self.crows.keys())
        self.exploration_queue = {crow_id: deque() for crow_id in self.crow_ids}
        self.scan_cooldown = {crow_id: 0 for crow_id in self.crow_ids}
        
    def update_from_scan(self, crow_id, scan_result):
        crow = self.crows[crow_id]
        center_x, center_y = crow['x'], crow['y']
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                abs_x = center_x + dx
                abs_y = center_y + dy
                pos = (abs_x, abs_y)
                
                # Check if within bounds
                if 0 <= abs_x < self.grid_size and 0 <= abs_y < self.grid_size:
                    cell_value = scan_result[dx + 2][dy + 2]
                    
                    if cell_value == 'W':
                        self.global_map[pos] = 'W'
                        self.known_walls.add(f"{abs_x}-{abs_y}")
                        if pos in self.unknown_cells:
                            self.unknown_cells.remove(pos)
                    elif cell_value == '_':
                        self.global_map[pos] = '_'
                        if pos in self.unknown_cells:
                            self.unknown_cells.remove(pos)
                        # Add to exploration queue if not visited by this crow
                        if pos not in crow['visited']:
                            self.exploration_queue[crow_id].append(pos)
                
    def update_from_move(self, crow_id, new_pos):
        old_pos = (self.crows[crow_id]['x'], self.crows[crow_id]['y'])
        self.crows[crow_id]['x'], self.crows[crow_id]['y'] = new_pos
        self.crows[crow_id]['visited'].add(new_pos)
        
        # Mark the new position as empty
        if new_pos not in self.global_map or self.global_map[new_pos] != '_':
            self.global_map[new_pos] = '_'
            if new_pos in self.unknown_cells:
                self.unknown_cells.remove(new_pos)
    
    def get_next_action(self):
        # Check if we've found all walls
        if len(self.known_walls) >= self.total_walls:
            return {
                'action_type': 'submit',
                'submission': list(self.known_walls)
            }
        
        # Rotate through crows
        crow_id = self.crow_ids[self.current_crow_index]
        self.current_crow_index = (self.current_crow_index + 1) % len(self.crow_ids)
        
        crow = self.crows[crow_id]
        current_pos = (crow['x'], crow['y'])
        
        # Check if we should scan (every 5 moves or when cooldown is up)
        if self.scan_cooldown[crow_id] <= 0 and self.should_scan(crow_id):
            self.scan_cooldown[crow_id] = 5  # Reset cooldown
            return {
                'crow_id': crow_id,
                'action_type': 'scan'
            }
        
        # Get next move direction
        direction = self.get_next_move_direction(crow_id)
        if direction:
            self.scan_cooldown[crow_id] = max(0, self.scan_cooldown[crow_id] - 1)
            return {
                'crow_id': crow_id,
                'action_type': 'move',
                'direction': direction
            }
        
        # If no move found, scan instead
        self.scan_cooldown[crow_id] = 5
        return {
            'crow_id': crow_id,
            'action_type': 'scan'
        }
    
    def should_scan(self, crow_id):
        # Scan if we haven't scanned this area recently and there are unknown cells nearby
        crow = self.crows[crow_id]
        unknown_nearby = 0
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                pos = (crow['x'] + dx, crow['y'] + dy)
                if pos in self.unknown_cells:
                    unknown_nearby += 1
        
        return unknown_nearby > 2  # Scan if more than 2 unknown cells nearby
    
    def get_next_move_direction(self, crow_id):
        crow = self.crows[crow_id]
        current_pos = (crow['x'], crow['y'])
        
        # First, try to explore from the queue
        while self.exploration_queue[crow_id]:
            target_pos = self.exploration_queue[crow_id][0]
            if target_pos in crow['visited']:
                self.exploration_queue[crow_id].popleft()
                continue
            
            # Find path to target
            path = self.find_path(current_pos, target_pos, crow_id)
            if path:
                next_pos = path[0]
                return self.get_direction(current_pos, next_pos)
            else:
                self.exploration_queue[crow_id].popleft()
        
        # If no exploration targets, find the nearest unknown cell
        nearest_unknown = self.find_nearest_unknown(current_pos)
        if nearest_unknown:
            path = self.find_path(current_pos, nearest_unknown, crow_id)
            if path:
                next_pos = path[0]
                return self.get_direction(current_pos, next_pos)
        
        return None
    
    def find_path(self, start, target, crow_id):
        """Simple BFS to find path to target, avoiding known walls"""
        queue = deque([(start, [])])
        visited = set([start])
        
        while queue:
            (x, y), path = queue.popleft()
            
            if (x, y) == target:
                return path
            
            for dx, dy, direction in [(0, 1, 'E'), (1, 0, 'S'), (0, -1, 'W'), (-1, 0, 'N')]:
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)
                
                if (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and 
                    new_pos not in visited and 
                    self.global_map.get(new_pos, '_') != 'W'):
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        
        return None
    
    def find_nearest_unknown(self, start_pos):
        """Find the nearest unknown cell using BFS"""
        queue = deque([start_pos])
        visited = set([start_pos])
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) in self.unknown_cells:
                return (x, y)
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                new_pos = (new_x, new_y)
                
                if (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and 
                    new_pos not in visited and 
                    self.global_map.get(new_pos, '_') != 'W'):
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return None
    
    def get_direction(self, current_pos, next_pos):
        cx, cy = current_pos
        nx, ny = next_pos
        
        if nx == cx:
            if ny > cy: return 'E'
            if ny < cy: return 'W'
        if ny == cy:
            if nx > cx: return 'S'
            if nx < cx: return 'N'
        return None

@app.route('/fog-of-wall', methods=['POST'])
def fog_of_wall():
    try:
        data = request.get_json()
        challenger_id = data['challenger_id']
        game_id = data['game_id']
        
        # Initialize or update game state
        if 'test_case' in data:
            # New test case
            game_states[game_id] = GameState(data['test_case'])
            response = game_states[game_id].get_next_action()
        else:
            # Update based on previous action
            prev_action = data['previous_action']
            game_state = game_states[game_id]
            
            if prev_action['your_action'] == 'move':
                game_state.update_from_move(
                    prev_action['crow_id'],
                    tuple(prev_action['move_result'])
                )
            elif prev_action['your_action'] == 'scan':
                game_state.update_from_scan(
                    prev_action['crow_id'],
                    prev_action['scan_result']
                )
            
            response = game_state.get_next_action()
        
        # Add common fields
        response['challenger_id'] = challenger_id
        response['game_id'] = game_id
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

##### fog of wall end


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)      