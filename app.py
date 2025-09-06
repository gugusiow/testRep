# app.py
from math import sqrt
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)
app.url_map.strict_slashes = False  # accept /path and /path/

# ---- tiny logger (helps catch wrong paths/methods) ----
@app.before_request
def _log_req():
    print(f"[REQ] {request.method} {request.path}  ctype={request.content_type}")

# ----------------------------
# Core scoring logic
# ----------------------------
def latency_points(cx, cy, bx, by):
    d = sqrt((cx - bx) ** 2 + (cy - by) ** 2)
    if d <= 2.0:
        return 30, d
    if d <= 4.0:
        return 20, d
    return 0, d

def solve(payload):
    if not isinstance(payload, dict):
        return {"error": "Input must be a JSON object."}, 400
    for key in ("customers", "concerts", "priority"):
        if key not in payload:
            return {"error": f"Missing '{key}'."}, 400

    customers = payload["customers"]
    concerts = payload["concerts"]
    priority = payload.get("priority") or {}

    # normalize concerts
    concert_list = []
    try:
        for c in concerts:
            cname = c["name"]
            bx, by = c["booking_center_location"]
            concert_list.append((str(cname), int(bx), int(by)))
    except Exception:
        return {"error": "Invalid 'concerts' element shape."}, 400

    card_to_concert = {str(k): str(v) for k, v in priority.items()}
    result = {}

    try:
        for cust in customers:
            name = str(cust["name"])
            vip = bool(cust["vip_status"])
            cx, cy = cust["location"]
            cx, cy = int(cx), int(cy)
            card = str(cust["credit_card"])

            vip_pts = 100 if vip else 0
            preferred_concert = card_to_concert.get(card)

            best_tuple = None   # (total_points, -distance, concert_name)
            best_concert = ""

            for cname, bx, by in concert_list:
                lat_pts, dist = latency_points(cx, cy, bx, by)
                cc_pts = 50 if preferred_concert == cname else 0
                total = vip_pts + cc_pts + lat_pts
                cand = (total, -dist, cname)  # ties: higher total, nearer, name asc
                if best_tuple is None or cand > best_tuple:
                    best_tuple = cand
                    best_concert = cname

            result[name] = best_concert
    except Exception:
        return {"error": "Invalid 'customers' element shape."}, 400

    return result, 200

# ----------------------------
# HTTP handlers
# ----------------------------
def _handle_post():
    ctype = (request.content_type or "").lower()
    if "application/json" not in ctype:
        return jsonify({"error": "Content-Type must be application/json."}), 400
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON."}), 400
    out, status = solve(payload)
    return jsonify(out), status

def _handle_options():
    # Some graders send preflight; respond 204
    return ("", 204)

@app.get("/")
def health():
    print(app.url_map)
    return "Ticketing Agent 2025 API is running."

# Accept both base and trailing-slash; also an /api prefix to dodge path proxies
for base in ("/ticketing-agent", "/api/ticketing-agent"):
    app.add_url_rule(base,       methods=["POST"],   view_func=_handle_post)
    app.add_url_rule(base + "/", methods=["POST"],   view_func=_handle_post)
    app.add_url_rule(base,       methods=["OPTIONS"],view_func=_handle_options)
    app.add_url_rule(base + "/", methods=["OPTIONS"],view_func=_handle_options)

# Optional: catch-all POST that forwards if the last segment is 'ticketing-agent'
# This neutralizes unexpected path prefixes from proxies (e.g., /v1/service/ticketing-agent)
@app.route("/<path:anypath>", methods=["POST", "OPTIONS"])
def catch_all(anypath):
    if anypath.rstrip("/").endswith("ticketing-agent"):
        if request.method == "OPTIONS":
            return _handle_options()
        return _handle_post()
    # real 404 for other paths
    return jsonify({"error": "Not Found"}), 404

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "3000"))
    print("== Route map ==")
    print(app.url_map)
    app.run(host="0.0.0.0", port=port)
