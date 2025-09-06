from flask import Flask, request, jsonify
from math import inf

# ----------------- solver -----------------
def solve(input_obj):
    tasks = input_obj.get("tasks", [])
    subway = input_obj.get("subway", [])
    s0_id = input_obj.get("starting_station")

    if not tasks:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # station indexing
    stations = set([s0_id])
    for t in tasks:
        stations.add(t["station"])
    for r in subway:
        a, b = r["connection"]
        stations.add(a); stations.add(b)
    station_ids = sorted(stations)
    id_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    s0 = id_to_idx[s0_id]
    n = len(station_ids)

    # APSP (Floyd–Warshall)
    dist = [[inf]*n for _ in range(n)]
    for i in range(n): dist[i][i] = 0
    for r in subway:
        a, b = r["connection"]; w = r["fee"]
        ia, ib = id_to_idx[a], id_to_idx[b]
        if w < dist[ia][ib]:
            dist[ia][ib] = w; dist[ib][ia] = w
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == inf: continue
            di = dist[i]
            for j in range(n):
                cand = dik + dk[j]
                if cand < di[j]: di[j] = cand

    # normalize & sort by end time
    T = [{
        "name": t["name"],
        "start": t["start"],
        "end": t["end"],
        "station": id_to_idx[t["station"]],
        "score": t["score"],
    } for t in tasks]
    T.sort(key=lambda x: (x["end"], x["start"]))
    m = len(T)

    # DP with fee tiebreak
    bestScore = [0]*m
    minFeeToEnd = [inf]*m
    prev = [-1]*m

    for i in range(m):
        si = T[i]["station"]; score_i = T[i]["score"]
        bScore = score_i
        bFee = dist[s0][si]
        p = -1
        for j in range(i):
            if T[j]["end"] <= T[i]["start"]:
                candScore = bestScore[j] + score_i
                candFee = minFeeToEnd[j] + dist[T[j]["station"]][si]
                if candScore > bScore or (candScore == bScore and candFee < bFee):
                    bScore, bFee, p = candScore, candFee, j
        bestScore[i], minFeeToEnd[i], prev[i] = bScore, bFee, p

    # pick best ending task (add return to s0)
    ansScore, ansFee, last = 0, 0, -1
    for i in range(m):
        totalScore = bestScore[i]
        totalFee = minFeeToEnd[i] + dist[T[i]["station"]][s0]
        if totalScore > ansScore or (totalScore == ansScore and totalFee < ansFee):
            ansScore, ansFee, last = totalScore, totalFee, i

    if ansScore == 0:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # reconstruct & sort by start (as required)
    chosen = []
    cur = last
    while cur != -1:
        chosen.append(T[cur])
        cur = prev[cur]
    chosen.sort(key=lambda x: (x["start"], x["end"], x["name"]))

    return {
        "max_score": ansScore,
        "min_fee": ansFee,
        "schedule": [t["name"] for t in chosen],
    }

# ----------------- Flask app -----------------
app = Flask(_name_)

# @app.get("/")
# def health():
#     return "Princess Diaries API is running."

@app.post("/princess-diaries")
def princess_diaries():
    try:
        payload = request.get_json(force=True, silent=False)
        # minimal validation
        if not isinstance(payload, dict) or "tasks" not in payload or "subway" not in payload or "starting_station" not in payload:
            return jsonify({"error": "Invalid input shape."}), 400
        result = solve(payload)
        return jsonify(result)
    except Exception as e:
        # log if you want: print(e)
        return jsonify({"error": "Invalid input or internal error."}), 400

# if _name_ == "_main_":
#     # Use PORT env var if deploying; default 3000
#     import os
#     port = int(os.getenv("PORT", "3000"))

# app = Flask(__name__)

@app.route('/trivia', methods=['GET'])
def get_trivia():
    answer = ['2, ']
    return jsonify({"answer": answer})

# @app.route('/data', methods=['POST'])
# def receive_data():
#     data = request.get_json()
#     return jsonify({"received": data}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)