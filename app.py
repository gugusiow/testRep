# app.py
from math import inf
from flask import Flask, request, jsonify

# ----------------- Optimized solver -----------------
def solve(input_obj):
    tasks = input_obj.get("tasks", [])
    subway = input_obj.get("subway", [])
    s0_id = input_obj.get("starting_station")

    if not tasks:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # --- Station remap to [0..n-1] for fast list indexing ---
    station_set = {s0_id}
    for t in tasks:
        station_set.add(t["station"])
    for r in subway:
        a, b = r["connection"]
        station_set.add(a); station_set.add(b)

    station_ids = list(station_set)
    id2idx = {sid: i for i, sid in enumerate(station_ids)}
    n = len(station_ids)
    s0 = id2idx[s0_id]

    # --- Dense matrix with parallel-edge compression ---
    INF = 10**15
    dist = [[INF]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for r in subway:
        a, b = r["connection"]; w = int(r["fee"])
        ia, ib = id2idx[a], id2idx[b]
        if w < dist[ia][ib]:
            dist[ia][ib] = w
            dist[ib][ia] = w

    # --- Floyd–Warshall (tight inner loops) ---
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(n):
                cand = dik + dk[j]
                if cand < di[j]:
                    di[j] = cand

    # Safety (graph should be connected per spec)
    for t in tasks:
        if dist[s0][id2idx[t["station"]]] >= INF:
            return {"max_score": 0, "min_fee": 0, "schedule": []}

    # --- Normalize + sort tasks by (end, start, name) ---
    T = [{
        "name": t["name"],
        "start": int(t["start"]),
        "end":   int(t["end"]),
        "station": id2idx[t["station"]],
        "score": int(t["score"]),
    } for t in tasks]
    T.sort(key=lambda x: (x["end"], x["start"], x["name"]))
    m = len(T)

    # --- DP: best score ending at i, with fee tiebreak ---
    bestScore = [0]*m
    minFeeToEnd = [INF]*m
    parent = [-1]*m

    for i in range(m):
        si = T[i]["station"]; sc_i = T[i]["score"]
        # base: start at s0 then do i
        bS, bF, bP = sc_i, dist[s0][si], -1
        start_i = T[i]["start"]

        for j in range(i):
            if T[j]["end"] <= start_i:
                candS = bestScore[j] + sc_i
                candF = minFeeToEnd[j] + dist[T[j]["station"]][si]
                if (candS > bS) or (candS == bS and candF < bF):
                    bS, bF, bP = candS, candF, j

        bestScore[i], minFeeToEnd[i], parent[i] = bS, bF, bP

    # --- pick best ending task incl. return to s0 ---
    ansScore, ansFee, last = 0, 0, -1
    for i in range(m):
        totalS = bestScore[i]
        totalF = minFeeToEnd[i] + dist[T[i]["station"]][s0]
        if (totalS > ansScore) or (totalS == ansScore and totalF < ansFee):
            ansScore, ansFee, last = totalS, totalF, i

    if ansScore == 0:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # --- reconstruct & sort by start (spec) ---
    chosen = []
    cur = last
    while cur != -1:
        chosen.append(T[cur])
        cur = parent[cur]
    chosen.sort(key=lambda x: (x["start"], x["end"], x["name"]))

    return {
        "max_score": ansScore,
        "min_fee": int(ansFee),
        "schedule": [t["name"] for t in chosen],
    }

# ----------------- Flask app -----------------
app = Flask(__name__)

@app.get("/")
def health():
    return "Princess Diaries API is running."

@app.post("/princess-diaries")
def princess_diaries():
    try:
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict) or not {"tasks","subway","starting_station"}.issubset(payload):
            return jsonify({"error": "Invalid input shape."}), 400
        return jsonify(solve(payload))
    except Exception:
        return jsonify({"error": "Invalid input or internal error."}), 400

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
