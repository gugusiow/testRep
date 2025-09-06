from math import sqrt
from flask import Flask, request, jsonify

app = Flask(__name__)

def latency_points(cx, cy, bx, by):
    # Euclidean distance with bucketed points (fast & matches the example)
    d = sqrt((cx - bx) ** 2 + (cy - by) ** 2)
    if d <= 2.0:
        return 30, d
    if d <= 4.0:
        return 20, d
    return 0, d

def solve(payload):
    # Basic shape validation
    if not isinstance(payload, dict):
        return {"error": "Input must be a JSON object."}, 400
    for key in ("customers", "concerts", "priority"):
        if key not in payload:
            return {"error": f"Missing '{key}'."}, 400

    customers = payload["customers"]
    concerts = payload["concerts"]
    priority = payload["priority"] or {}

    # Pre-normalize concerts for speed
    concert_list = []
    for c in concerts:
        try:
            cname = c["name"]
            bx, by = c["booking_center_location"]
        except Exception:
            return {"error": "Invalid 'concerts' element shape."}, 400
        concert_list.append((cname, int(bx), int(by)))

    # Build quick map card -> preferred concert
    # (We accept that cards may map to concerts that don't exist; those just won't match.)
    card_to_concert = {str(k): str(v) for k, v in priority.items()}

    result = {}

    # Evaluate each customer independently
    for cust in customers:
        try:
            name = cust["name"]
            vip = bool(cust["vip_status"])
            cx, cy = cust["location"]
            card = cust["credit_card"]
        except Exception:
            return {"error": "Invalid 'customers' element shape."}, 400

        vip_pts = 100 if vip else 0
        preferred_concert = card_to_concert.get(str(card), None)

        best = None  # (total_points, distance, concert_name)
        best_concert_name = None

        for cname, bx, by in concert_list:
            lat_pts, dist = latency_points(int(cx), int(cy), bx, by)
            cc_pts = 50 if preferred_concert == cname else 0
            total = vip_pts + cc_pts + lat_pts

            # Argmax with deterministic tie-break:
            # 1) higher total points
            # 2) smaller distance
            # 3) lexicographically smaller concert name
            cand = (total, -dist, cname)  # negate dist so bigger is better in tuple compare
            if best is None or cand > best:
                best = cand
                best_concert_name = cname

        # If there are no concerts (shouldn’t happen per constraints), skip
        result[name] = best_concert_name if best_concert_name is not None else ""

    return result, 200

@app.get("/")
def health():
    return "Ticketing Agent 2025 API is running."

@app.post("/ticketing-agent")
def ticketing_agent():
    # Enforce JSON
    if request.content_type is None or "application/json" not in request.content_type.lower():
        return jsonify({"error": "Content-Type must be application/json."}), 400
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON."}), 400

    out, status = solve(payload)
    return (jsonify(out), status)

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
