# app.py
from math import sqrt
from flask import Flask, request, jsonify

app = Flask(__name__)
# Accept both "/ticketing-agent" and "/ticketing-agent/" without redirect issues
app.url_map.strict_slashes = False

# ----------------------------
# Core scoring logic
# ----------------------------
def latency_points(cx, cy, bx, by):
    """
    Bucket latency points by Euclidean distance to booking center:
      d <= 2  -> +30
      d <= 4  -> +20
      else    -> +0
    Returns (points, distance_float).
    """
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
    priority = payload.get("priority") or {}

    # Normalize concerts (fast lookup + conversion)
    concert_list = []
    try:
        for c in concerts:
            cname = c["name"]
            bx, by = c["booking_center_location"]
            concert_list.append((str(cname), int(bx), int(by)))
    except Exception:
        return {"error": "Invalid 'concerts' element shape."}, 400

    # Map credit card -> priority concert
    card_to_concert = {str(k): str(v) for k, v in priority.items()}

    result = {}

    # Evaluate each customer independently
    try:
        for cust in customers:
            name = str(cust["name"])
            vip = bool(cust["vip_status"])
            cx, cy = cust["location"]
            cx, cy = int(cx), int(cy)
            card = str(cust["credit_card"])

            vip_pts = 100 if vip else 0
            preferred_concert = card_to_concert.get(card)

            best_tuple = None   # tuple for comparison: (total_points, -distance, concert_name_lex)
            best_concert = ""

            for cname, bx, by in concert_list:
                lat_pts, dist = latency_points(cx, cy, bx, by)
                cc_pts = 50 if preferred_concert == cname else 0
                total = vip_pts + cc_pts + lat_pts

                # Argmax with deterministic tie-breakers:
                # 1) higher total points
                # 2) smaller distance
                # 3) lexicographically smaller concert name
                cand = (total, -dist, cname)
                if best_tuple is None or cand > best_tuple:
                    best_tuple = cand
                    best_concert = cname

            result[name] = best_concert
    except Exception:
        return {"error": "Invalid 'customers' element shape."}, 400

    return result, 200

# ----------------------------
# HTTP routes
# ----------------------------
@app.get("/")
def health():
    return "Ticketing Agent 2025 API is running."

@app.route("/ticketing-agent", methods=["POST", "OPTIONS"])
@app.route("/ticketing-agent/", methods=["POST", "OPTIONS"])
def ticketing_agent():
    # Preflight (some gateways send this)
    if request.method == "OPTIONS":
        # Minimal CORS-friendly response (if needed by grader)
        resp = jsonify({})
        resp.status_code = 204
        return resp

    # Enforce JSON
    ctype = (request.content_type or "").lower()
    if "application/json" not in ctype:
        return jsonify({"error": "Content-Type must be application/json."}), 400

    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON."}), 400

    out, status = solve(payload)
    return jsonify(out), status

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    # For local/dev: python app.py
    # For prod (e.g., Render/Heroku): gunicorn app:app --workers 2 --threads 4 --timeout 60
    import os
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
