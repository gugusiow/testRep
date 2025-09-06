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

