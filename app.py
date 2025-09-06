from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.url_map.strict_slashes = False  # avoid 308 redirects (HTML)

MAX_HOUR = 4096  # inclusive bound per spec

def solve_by_prefix(slots):
    """
    Prefix-sum on a fixed time grid [0..4096].
    - Build delta so that for each [s,e): delta[s]+=1, delta[e]-=1
    - Scan t=0..4096, maintain cur occupancy.
      * Start a busy block when cur goes 0 -> >0 at t.
      * End a busy block when cur goes >0 -> 0 at t (end time is t).
    - minBoatsNeeded = max cur during the scan.
    Returns (merged_intervals_sorted_by_end_then_start, min_boats).
    """
    # Difference array (one extra slot is fine)
    delta = [0] * (MAX_HOUR + 1)

    # Clamp inputs into [0, MAX_HOUR] and accumulate
    for s, e in slots:
        # accept anything with s < e; clamp to bounds to stay index-safe
        if not isinstance(s, int) or not isinstance(e, int):
            continue
        if s >= e:
            continue
        
        # Clamp start to [0, MAX_HOUR]
        s_orig = s
        s = max(0, min(MAX_HOUR, s))
        
        # For end, we need to handle values > MAX_HOUR differently
        # If end > MAX_HOUR, clamp to MAX_HOUR + 1 so the interval [s, MAX_HOUR] is valid
        e_orig = e
        if e > MAX_HOUR:
            e = MAX_HOUR + 1
        else:
            e = max(0, e)
        
        if s >= e:
            continue
            
        delta[s] += 1
        if e <= MAX_HOUR:
            delta[e] -= 1

    merged = []
    cur = 0
    peak = 0
    in_busy = False
    start_t = None

    # Scan all integer hours 0..4096
    for t in range(0, MAX_HOUR + 1):
        cur += delta[t]
        if cur > peak:
            peak = cur

        if cur > 0 and not in_busy:
            # entering busy
            in_busy = True
            start_t = t
        elif cur == 0 and in_busy:
            # leaving busy; block covers [start_t, t)
            merged.append([start_t, t])
            in_busy = False
            start_t = None
    
    # If we're still in a busy period at the end, close it at MAX_HOUR + 1
    if in_busy:
        merged.append([start_t, MAX_HOUR + 1])
        in_busy = False

    # merged blocks are discovered in chronological order:
    # starts and ends are strictly increasing → end-time ascending already.
    # (If you still want to be extra explicit:)
    merged.sort(key=lambda ab: (ab[1], ab[0]))

    return merged, peak

@app.post("/sailing-club/submission")
def submission():
    # Always return JSON (prevents '<!doctype ...>' parse errors)
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"solutions": []}), 200

    tcs = data.get("testCases") if isinstance(data, dict) else None
    if not isinstance(tcs, list):
        return jsonify({"solutions": []}), 200

    solutions = []
    for tc in tcs:
        # Always emit one solution per test case
        tc_id = tc.get("id") if isinstance(tc, dict) else None
        raw = tc.get("input", []) if isinstance(tc, dict) else []

        # normalize slots; if raw malformed, treat as empty
        slots = []
        if isinstance(raw, list):
            for pair in raw:
                if (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    s, e = pair
                    # keep any s,e; clamp happens inside solver
                    slots.append([s, e])

        merged, boats = solve_by_prefix(slots)

        solutions.append({
            "id": "" if tc_id is None else str(tc_id),
            "sortedMergedSlots": merged if merged else [],
            "minBoatsNeeded": int(boats)
        })

    return jsonify({"solutions": solutions}), 200

# Helpful JSON for accidental GET (still JSON, never HTML)
@app.get("/sailing-club/submission")
def submission_get():
    return jsonify({
        "message": "POST JSON to this endpoint.",
        "example": {"testCases":[{"id":"0001","input":[[1,8],[17,28],[5,8],[8,10]]}]}
    }), 200

# JSON-only error handlers
@app.errorhandler(HTTPException)
def handle_http_exc(e):
    return jsonify({"error": e.name, "status": e.code, "description": e.description}), e.code

@app.errorhandler(Exception)
def handle_exc(_):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    # Deploy: gunicorn -w 2 -b 0.0.0.0:$PORT app:app

