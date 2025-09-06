from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.url_map.strict_slashes = False  # avoid redirect HTML

def merge_slots(slots):
    if not slots: return []
    slots = sorted(slots, key=lambda ab: (ab[0], ab[1]))
    merged, s, e = [], *slots[0]
    for ns, ne in slots[1:]:
        if ns <= e:
            if ne > e: e = ne
        else:
            merged.append([s, e])
            s, e = ns, ne
    merged.append([s, e])
    merged.sort(key=lambda ab: (ab[1], ab[0]))
    return merged

def min_boats_needed(slots):
    events = []
    for s, e in slots:
        if isinstance(s, int) and isinstance(e, int) and s < e:
            events += [(s, 1), (e, -1)]
    events.sort(key=lambda x: (x[0], x[1]))  # end(-1) before start(+1)
    cur = peak = 0
    for _, d in events:
        cur += d
        if cur > peak: peak = cur
    return peak

@app.post("/sailing-club/submission")
def submission():
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error":"Invalid JSON body"}), 400

    tcs = data.get("testCases")
    if not isinstance(tcs, list):
        return jsonify({"error":"Body must contain testCases: [...]"}), 400

    sols = []
    for tc in tcs:
        tc_id = tc.get("id")
        raw = tc.get("input", [])
        if tc_id is None:  # skip malformed
            continue
        slots = []
        for pair in raw if isinstance(raw, list) else []:
            if (isinstance(pair, (list,tuple)) and len(pair)==2 and
                isinstance(pair[0], int) and isinstance(pair[1], int) and
                0 <= pair[0] < pair[1] <= 4096):
                slots.append([pair[0], pair[1]])
        sols.append({
            "id": tc_id,
            "sortedMergedSlots": merge_slots(slots),
            "minBoatsNeeded": min_boats_needed(slots),
        })
    return jsonify({"solutions": sols}), 200

# Helpful JSON response for GET (so browsers don’t see HTML)
@app.get("/sailing-club/submission")
def submission_get():
    return jsonify({
        "message": "Use POST with Content-Type: application/json",
        "example": {"testCases":[{"id":"0001","input":[[1,8],[17,28],[5,8],[8,10]]}]}
    }), 200

# Ensure all errors return JSON (not HTML)
@app.errorhandler(HTTPException)
def handle_http_exc(e):
    return jsonify({"error": e.name, "status": e.code, "description": e.description}), e.code

@app.errorhandler(Exception)
def handle_exc(e):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

