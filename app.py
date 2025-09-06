from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.url_map.strict_slashes = False  # avoid 308 redirects (which can yield HTML)

# ------------------------------
# Helpers & constants
# ------------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def json_ok(payload):
    return jsonify(payload), 200

def json_error(msg, code=400):
    return jsonify({"error": msg}), code

# Sensors: [-90°, -45°, 0°, +45°, +90°]
LEFT_IDX, FWDL_IDX, FWD_IDX, FWDR_IDX, RIGHT_IDX = 0, 1, 2, 3, 4

# Movement/strategy caps
MAX_FWD_SPEED = 2        # keep ≤2 to make safe moving-rotations easier
TARGET_TURN_SPEED = 1    # brake to |m|=1 for moving rotations (m_eff ≤ 1)

# Clearance thresholds (cm)
NEED_FWD = 8.0           # half-step forward = 8 cm
NEED_SIDE = 8.0          # conservative side clearance

# ------------------------------
# Token logic
# ------------------------------
def accel_forward(m):
    """Accelerate forward respecting caps and direction rules."""
    if m < 0:
        # illegal to accelerate opposite direction; brake toward 0 first
        return "BB" if abs(m) >= 2 else "V0"
    if m < MAX_FWD_SPEED:
        return "F2"
    return "F1"

def brake_to_one(m):
    """One token nudging |m| toward 1 (prep for moving rotation)."""
    if m > 1:
        return "BB" if m >= 3 else "F0"
    if m < -1:
        return "BB" if m <= -3 else "V0"
    if m == 2:
        return "F0"
    if m == -2:
        return "V0"
    return None  # already |m| ≤ 1

def full_stop(m):
    """One token to reduce |m| toward 0 quickly."""
    if m == 0:
        return None
    return "BB" if abs(m) >= 2 else ("F0" if m > 0 else "V0")

def moving_rotation_allowed(m_in, token):
    """
    Check m_eff ≤ 1 for moving rotations (translation then 45° rotation).
    Only permit bases that don't accelerate away from zero (F1/F0/V1/V0/BB).
    """
    if len(token) < 2 or token[-1] not in ("L", "R"):
        return False
    base = token[:-1]

    # Compute m_out for the translation part
    if base == "F1":
        m_out = m_in
    elif base == "F0":
        m_out = m_in - 1 if m_in > 0 else m_in + 1 if m_in < 0 else 0
    elif base == "V1":
        m_out = m_in
    elif base == "V0":
        m_out = m_in + 1 if m_in < 0 else m_in - 1 if m_in > 0 else 0
    elif base == "BB":
        if m_in > 0:
            m_out = m_in - 2
        elif m_in < 0:
            m_out = m_in + 2
        else:
            m_out = 0
    else:
        # disallow F2/V2 moving rotations
        return False

    m_eff = (abs(m_in) + abs(m_out)) / 2.0
    return m_eff <= 1.0

# ------------------------------
# Policy (right-hand follower, wall-safe)
# ------------------------------
def choose_instructions(sensor_data, momentum):
    """
    Right-hand wall follower with safety:
      • Treat sensors as cm; need >=8 cm to translate forward half-step.
      • Moving rotations only if FRONT is clear (>=8 cm) and m_eff ≤ 1.
      • In-place turns only at momentum == 0.
    Priority: Right open → Forward open → Left open → U-turn.
    """
    # Normalize sensors to floats
    if not isinstance(sensor_data, (list, tuple)):
        s = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        s = list(sensor_data)[:5] + [0.0] * (5 - len(sensor_data))
    try:
        s = [float(x) for x in s]
    except Exception:
        s = [0.0, 0.0, 0.0, 0.0, 0.0]

    front_clear = s[FWD_IDX] >= NEED_FWD
    right_open  = s[RIGHT_IDX] >= NEED_SIDE
    left_open   = s[LEFT_IDX]  >= NEED_SIDE

    m = clamp(int(momentum), -4, +4)

    # Helper: stop then perform an in-place rotation
    def stop_then(turn):
        if m != 0:
            t = full_stop(m)
            return [t if t else "BB"]
        return [turn]

    # 1) Prefer RIGHT if open
    if right_open:
        # If front is clear, we may use a moving rotation safely.
        if front_clear and m > 0 and abs(m) <= TARGET_TURN_SPEED:
            if moving_rotation_allowed(m, "F1R"):
                return ["F1R", accel_forward(m)]
            if moving_rotation_allowed(m, "F0R"):
                return ["F0R", accel_forward(max(m - 1, 0))]
        # Speed control or opposite direction before turning
        if m > TARGET_TURN_SPEED:
            return [brake_to_one(m) or "F0"]
        if m < 0:
            return [full_stop(m) or "BB"]
        # If front blocked, turn in place (no translation into wall)
        if not front_clear:
            return stop_then("R") + ["F2"]
        # Otherwise at low speed but moving rotation not chosen: gentle decel or in-place
        if m == 0:
            return ["R", "F2"]
        return ["F0"]

    # 2) Go FORWARD if clear
    if front_clear:
        if m >= 0:
            step1 = accel_forward(m)
            m2 = min(m + 1, MAX_FWD_SPEED) if step1 == "F2" else m
            step2 = accel_forward(m2)
            return [step1, step2]
        else:
            # moving backward—must brake toward 0 first
            return [full_stop(m) or "BB"]

    # 3) Try LEFT if open
    if left_open:
        if front_clear and m > 0 and abs(m) <= TARGET_TURN_SPEED:
            if moving_rotation_allowed(m, "F1L"):
                return ["F1L", accel_forward(m)]
            if moving_rotation_allowed(m, "F0L"):
                return ["F0L", accel_forward(max(m - 1, 0))]
        if m > TARGET_TURN_SPEED:
            return [brake_to_one(m) or "F0"]
        if m < 0:
            return [full_stop(m) or "BB"]
        if not front_clear:
            return stop_then("L") + ["F2"]
        if m == 0:
            return ["L", "F2"]
        return ["F0"]

    # 4) DEAD END → U-turn: stop, then in-place rotations
    if m != 0:
        return [full_stop(m) or "BB"]
    # Do 180° via 4 × 45°; keep batch short
    return ["R", "R", "R"]

# ------------------------------
# API
# ------------------------------
@app.post("/micro-mouse")
def micro_mouse():
    # Always JSON (prevents HTML doctype issues)
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return json_ok({"instructions": [], "end": True})

    if not isinstance(payload, dict):
        return json_ok({"instructions": [], "end": True})

    sensor_data = payload.get("sensor_data") or [0, 0, 0, 0, 0]
    momentum = payload.get("momentum", 0)
    goal_reached = payload.get("goal_reached", False)

    # If goal already reached, end without moving
    if goal_reached:
        return json_ok({"instructions": [], "end": True})

    try:
        instr = choose_instructions(sensor_data, momentum)
    except Exception:
        instr = []

    # Spec forbids empty/invalid instruction arrays → ensure at least one safe token
    if not instr:
        instr = ["BB"] if momentum != 0 else ["F1"]

    # Keep batches compact (amortize 50 ms thinking once per call)
    if len(instr) > 4:
        instr = instr[:4]

    return json_ok({"instructions": instr, "end": False})

@app.get("/micro-mouse")
def micro_mouse_get():
    return json_ok({
        "message": "POST JSON to this endpoint.",
        "example_request": {
            "game_uuid": "demo",
            "sensor_data": [12, 12, 12, 12, 12],
            "total_time_ms": 0,
            "goal_reached": False,
            "best_time_ms": None,
            "run_time_ms": 0,
            "run": 0,
            "momentum": 0
        },
        "example_response": {"instructions": ["F2", "F2"], "end": False}
    })

# JSON-only error handlers
@app.errorhandler(HTTPException)
def handle_http_exc(e):
    return jsonify({"error": e.name, "status": e.code, "description": e.description}), e.code

@app.errorhandler(Exception)
def handle_exc(_):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # Local run
    app.run(host="0.0.0.0", port=8000)
    # Deploy (Render/Heroku/etc.):
    # gunicorn -w 2 -b 0.0.0.0:$PORT app:app
