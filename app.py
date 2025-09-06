from math import inf

def solve(input_obj):
    tasks = input_obj.get("tasks", [])
    subway = input_obj.get("subway", [])
    s0_id = input_obj.get("starting_station")

    # No tasks -> trivial
    if not tasks:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # ---- Station remap to [0..n-1] (fast lists) ----
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

    # ---- Dense matrix with min-edge compression ----
    INF = 10**15
    dist = [[INF]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0

    for r in subway:
        a, b = r["connection"]; w = int(r["fee"])
        ia, ib = id2idx[a], id2idx[b]
        # keep only the minimum parallel edge
        if w < dist[ia][ib]:
            dist[ia][ib] = w
            dist[ib][ia] = w

    # ---- Floyd–Warshall (tight inner loops) ----
    # Complexity ~ n^3 = 64M ops at n=400; with local var hoisting this is typically OK.
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            # manual loop faster than comprehensions here
            for j in range(n):
                cand = dik + dk[j]
                if cand < di[j]:
                    di[j] = cand

    # Safety: if any station is unreachable (shouldn't happen per problem), bail
    for t in tasks:
        if dist[s0][id2idx[t["station"]]] >= INF:
            return {"max_score": 0, "min_fee": 0, "schedule": []}

    # ---- Normalize & sort tasks by (end, start, name) ----
    T = [{
        "name": t["name"],
        "start": int(t["start"]),
        "end":   int(t["end"]),
        "station": id2idx[t["station"]],
        "score": int(t["score"]),
    } for t in tasks]
    T.sort(key=lambda x: (x["end"], x["start"], x["name"]))
    m = len(T)

    # ---- DP over tasks (end-at-i), fee tiebreak ----
    bestScore = [0]*m
    minFeeToEnd = [INF]*m
    parent = [-1]*m

    for i in range(m):
        si = T[i]["station"]
        sc_i = T[i]["score"]

        # base: start at s0 and do only i
        bS = sc_i
        bF = dist[s0][si]
        bP = -1

        # extend from any j with end <= start_i
        start_i = T[i]["start"]
        for j in range(i):
            if T[j]["end"] <= start_i:
                candS = bestScore[j] + sc_i
                candF = minFeeToEnd[j] + dist[T[j]["station"]][si]
                if (candS > bS) or (candS == bS and candF < bF):
                    bS, bF, bP = candS, candF, j

        bestScore[i] = bS
        minFeeToEnd[i] = bF
        parent[i] = bP

    # ---- Pick best ending task + return to s0 ----
    ansScore, ansFee, last = 0, 0, -1
    for i in range(m):
        totalS = bestScore[i]
        totalF = minFeeToEnd[i] + dist[T[i]["station"]][s0]
        if (totalS > ansScore) or (totalS == ansScore and totalF < ansFee):
            ansScore, ansFee, last = totalS, totalF, i

    if ansScore == 0:
        return {"max_score": 0, "min_fee": 0, "schedule": []}

    # ---- Reconstruct and sort by start (spec) ----
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
