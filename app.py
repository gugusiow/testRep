from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.url_map.strict_slashes = False  # avoid 308 redirects (HTML)

MAX_HOUR = 4096  # inclusive bound per spec

def solve_by_prefix(slots):
    """
    Greedy algorithm for interval merging and boat counting.
    1. Filter and validate input slots
    2. Merge overlapping intervals using greedy approach
    3. Count minimum boats needed using event-based approach
    Returns (merged_intervals_sorted_by_end_then_start, min_boats).
    """
    # Filter valid slots
    valid_slots = []
    for s, e in slots:
        # Only accept valid integer intervals with s < e
        if not isinstance(s, int) or not isinstance(e, int):
            continue
        if s >= e:
            continue
        # Clamp to valid range [0, MAX_HOUR] but preserve intervals that extend beyond
        s_clamped = max(0, min(MAX_HOUR, s))
        e_clamped = max(0, min(MAX_HOUR + 1, e))  # Allow end to be MAX_HOUR + 1
        if s_clamped >= e_clamped:
            continue
        valid_slots.append([s_clamped, e_clamped])
    
    if not valid_slots:
        return [], 0
    
    # Part 1: Merge overlapping intervals using greedy approach
    # Sort by start time, then by end time
    valid_slots.sort()
    
    merged = []
    for start, end in valid_slots:
        if not merged or merged[-1][1] < start:
            # No overlap, add new interval
            merged.append([start, end])
        else:
            # Overlap, extend the last interval
            merged[-1][1] = max(merged[-1][1], end)
    
    # Part 2: Calculate minimum boats needed using event-based approach
    events = []
    for start, end in valid_slots:
        events.append((start, 1))   # boat needed at start
        events.append((end, -1))    # boat returned at end
    
    events.sort()  # Sort by time, then by type (returns before bookings at same time)
    
    max_boats = 0
    current_boats = 0
    for time, change in events:
        current_boats += change
        max_boats = max(max_boats, current_boats)
    
    # Sort merged intervals by end time, then by start time as required
    merged.sort(key=lambda interval: (interval[1], interval[0]))
    
    return merged, max_boats

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

