from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.url_map.strict_slashes = False  # avoid 308 redirect HTML on trailing slashes

# ---------- Core logic ----------
def merge_slots(slots):
    """
    Merge overlapping or touching [start, end] slots into busy blocks.
    Final result sorted by end time asc, then start time asc (per spec note).
    """
    if not slots:
        return []
    slots = sorted(slots, key=lambda ab: (ab[0], ab[1]))  # sort by start, then end

    merged = []
    cs, ce = slots[0]
    for s, e in slots[1:]:
        if s <= ce:  # merge overlap or touch
            if e > ce:
                ce = e
        else:
            merged.append([cs, ce])
            cs, ce = s, e
    merged.append([cs, ce])

    # Spec requires end-time ascending (tie-break by start)
    merged.sort(key=lambda ab: (ab[1], ab[0]))
    return merged

def min_boats_needed(slots):
    """
    Sweep-line over start/end events. Ends at time t free boats before starts at time t,
    so we sort (time, delta) with delta -1 before +1.
    """
    events = []
    for s, e in slots:
        if isinstance(s, int) and isinstance(e, int) and s < e:
            events.append((s, 1))   # start
            events.append((e, -1))  # end

    # Sort by time; at ties, -1 (end) before +1 (start)
    events.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in events:
        cur += d
        if cur > peak:
            peak = cur
    return peak

# ---------- API ----------
@app.post("/sailing-club/submission")
def submission():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        # Always JSON, never HTML
        return jsonify({"solutions": []}), 200

    tcs = data.get("testCases") if isinstance(data, dict) else None
    if not isinstance(tcs, list):
        return jsonify({"solutions": []}), 200

    solutions = []
    for tc in tcs:
        # Always produce an answer entry, even if malformed
        tc_id = tc.get("id") if isinstance(tc, dict) else None
        raw = tc.get("input", []) if isinstance(tc, dict) else []

        # Normalize slots; if raw is bad, treat as empty
        slots = []
        if isinstance(raw, list):
            for pair in raw:
                if (isinstance(pair, (list, tuple)) and len(pair) == 2 and
                    isinstance(pair[0], int) and isinstance(pair[1], int)):
                    s, e = pair
                    if s < e:  # accept; don't drop for bounds to avoid "missing solutions"
                        slots.append([s, e])

        merged = merge_slots(slots)              # merges touching (s ≤ cur_end)
        boats = min_boats_needed(slots)          # ends before starts at same t

        solutions.append({
            "id": str(tc_id) if tc_id is not None else "",
            "sortedMergedSlots": merged if merged else [],
            "minBoatsNeeded": int(boats)
        })

    return jsonify({"solutions": solutions}), 200

# Helpful JSON hint for accidental GETs (prevents HTML)
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
    # Local dev
    app.run(host="0.0.0.0", port=8000)
    # Deploy (Render/Heroku/etc.):
    # gunicorn -w 2 -b 0.0.0.0:$PORT app:app
