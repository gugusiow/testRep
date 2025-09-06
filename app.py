from __future__ import annotations
from flask import Flask, request, jsonify

app = Flask(__name__)
app.url_map.strict_slashes = False  # accept both with/without trailing slash

def merge_slots(slots):
    """Merge overlapping or touching [start, end] slots.
    Returns non-overlapping slots. Touching (end == next.start) merges.
    """
    if not slots:
        return []

    # Sort by start, then end for stable merging
    slots = sorted(slots, key=lambda ab: (ab[0], ab[1]))

    merged = []
    cur_s, cur_e = slots[0]
    for s, e in slots[1:]:
        if s <= cur_e:                 # overlap or touch → merge
            if e > cur_e:
                cur_e = e
        else:                           # gap → push current and reset
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    merged.append([cur_s, cur_e])

    # Requirement: sort by increasing end time (then start as tie-breaker)
    merged.sort(key=lambda ab: (ab[1], ab[0]))
    return merged

def min_boats_needed(slots):
    """Max number of concurrent slots using sweep line.
    End-at-t frees before start-at-t to allow reuse.
    """
    events = []
    for s, e in slots:
        # basic guard; the challenge guarantees 1 ≤ duration ≤ 48, but be defensive
        if not (isinstance(s, int) and isinstance(e, int) and s < e):
            continue
        events.append((s, +1))  # start
        events.append((e, -1))  # end

    # Sort by time; for ties process end (-1) before start (+1)
    events.sort(key=lambda x: (x[0], x[1]))

    cur = 0
    peak = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak

@app.post("/sailing-club/submission")
def submission():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    if not isinstance(data, dict) or "testCases" not in data or not isinstance(data["testCases"], list):
        return jsonify({"error": "Body must contain testCases: [...]"}), 400

    solutions = []
    for tc in data["testCases"]:
        # Validate each test case
        tc_id = tc.get("id")
        raw = tc.get("input", [])
        if tc_id is None:
            # Skip malformed cases but keep going
            continue

        # Normalize input to list of [int,int]
        slots = []
        if isinstance(raw, list):
            for pair in raw:
                if (
                    isinstance(pair, (list, tuple)) and
                    len(pair) == 2 and
                    isinstance(pair[0], int) and
                    isinstance(pair[1], int)
                ):
                    s, e = pair
                    # Only accept valid duration (>=1 hour) and within constraints
                    if 0 <= s < e <= 4096:
                        slots.append([s, e])

        merged = merge_slots(slots)
        boats = min_boats_needed(slots)

        solutions.append({
            "id": tc_id,
            "sortedMergedSlots": merged,
            "minBoatsNeeded": boats
        })

    return jsonify({"solutions": solutions}), 200

if __name__ == "__main__":
    # Local run: python app.py
    # Deploy: gunicorn -w 2 -b 0.0.0.0:$PORT app:app
    app.run(host="0.0.0.0", port=8000)
