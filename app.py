from __future__ import annotations
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.url_map.strict_slashes = False  # avoid 308 redirects

# ------------------------------
# Utilities
# ------------------------------
def json_error(code: int, msg: str):
    return jsonify({"error": msg}), code

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

# Sensors are in order: [-90°, -45°, 0°, +45°, +90°]
# We'll interpret "open" as > 0 (any range means there is space).
LEFT_IDX, FWDL_IDX, FWD_IDX, FWDR_IDX, RIGHT_IDX = 0, 1, 2, 3, 4

# We’ll keep forward speed in {0,1,2} to keep moving-rotations legal.
MAX_FWD_SPEED = 2
TARGET_TURN_SPEED = 1  # we brake to |m|=1 for moving rotations (m_eff<=1)

# ------------------------------
# Movement primitives (tokens)
# ------------------------------
def accel_forward(m):
    """Accelerate forward respecting caps."""
    if m < 0:
        # Illegal to accelerate opposite direction. Must brake toward 0 first.
        return "BB" if abs(m) >= 2 else "V0"  # both move half-steps but reduce |m|
    if m < MAX_FWD_SPEED:
        return "F2"  # accel +1
    return "F1"      # hold

def brake_to_one(m):
    """Return one token that tends to bring |m| -> 1 (for moving rotations)."""
    if m > 1:
        # BB reduces by 2 toward 0; if m==2, BB -> 0; else F0 reduces by 1
        return "BB" if m >= 3 else "F0"
    elif m < -1:
        return "BB" if m <= -3 else "V0"
    elif m == 2 or m == -2:
        # Prefer F0/V0 over BB if just one step needed
        return "F0" if m == 2 else "V0"
    else:
        return None  # already |m| <= 1

def full_stop(m):
    """One token to move |m| toward 0 quickly."""
    if m == 0:
        return None
    # BB halves |m| fast: reduces by 2 toward 0 and still moves half-step
    return "BB" if abs(m) >= 2 else ("F0" if m > 0 else "V0")

def moving_rotation_allowed(m_in, token):
    """
    For tokens like F1R, F0L, etc., must have m_eff ≤ 1.
    m_out for F1/F0: same or ±1 toward 0 respectively.
    BB* moving rotations are allowed by spec if m_eff ≤ 1 as well.
    """
    base = token[:-1]    # e.g., "F1", "F0", "BB", "V1"
    rot = token[-1]      # 'L' or 'R'
    if rot not in ("L", "R"):
        return False

    if base == "F1":
        m_out = m_in
    elif base == "F0":
        m_out = m_in - 1 if m_in > 0 else m_in + 1 if m_in < 0 else 0
    elif base == "BB":
        # toward 0 by 2
        if m_in > 0:
            m_out = m_in - 2
        elif m_in < 0:
            m_out = m_in + 2
        else:
            m_out = 0  # “default action at rest” but still timed
    elif base == "V1":
        m_out = m_in
    elif base == "V0":
        m_out = m_in + 1 if m_in < 0 else m_in - 1 if m_in > 0 else 0
    elif base in ("F2", "V2"):  # accelerating in place with rotation is illegal here
        return False
    else:
        return False

    m_eff = (abs(m_in) + abs(m_out)) / 2.0
    return m_eff <= 1.0

def choose_instructions(sensor_data, momentum):
    """
    Right-hand wall follower (reactive):
    Priority: Right open → Forward open → Left open → U-turn.
    Momentum-aware:
      • Cap forward speed at +2.
      • For moving rotation, ensure |m| ≤ 1 (we brake to 1 first).
      • In-place rotations require m == 0.
    Returns a short list of 2–4 tokens.
    """
    s = sensor_data or [0, 0, 0, 0, 0]
    right_open = s[RIGHT_IDX] > 0
    forward_open = s[FWD_IDX] > 0
    left_open = s[LEFT_IDX] > 0

    m = clamp(int(momentum), -4, +4)
    plan = []

    # 1) If right is open, try to turn right using a moving rotation at |m|<=1
    if right_open:
        if abs(m) <= TARGET_TURN_SPEED and m > 0:
            # Moving rotation, time is only translation; safe if m_eff<=1
            if moving_rotation_allowed(m, "F1R"):
                plan.append("F1R")
                # After turning, try to accelerate
                plan.append(accel_forward(m))  # one more step to keep moving
                return plan
            # F0R (slight decel + rotate) might also be legal
            if moving_rotation_allowed(m, "F0R"):
                plan.append("F0R")
                plan.append(accel_forward(max(m-1, 0)))
                return plan
        # If moving too fast or stopped, prepare for a clean turn
        if m > TARGET_TURN_SPEED:
            plan.append(brake_to_one(m) or "F0")
            return plan
        if m < 0:
            # Must stop before reversing direction to forward
            plan.append(full_stop(m) or "BB")
            return plan
        if m == 0:
            # In-place turn allowed only at momentum 0
            plan.append("R")
            # After rotate, start moving
            plan.append("F2")
            return plan
        # We are at m==1 but moving rotation wasn’t permitted? Brake slightly.
        plan.append("F0")
        return plan

    # 2) If forward is open, go forward (accelerate up to +2).
    if forward_open:
        if m >= 0:
            plan.append(accel_forward(m))
            # Try to add a second forward to amortize thinking time
            m2 = min(m + 1, MAX_FWD_SPEED) if plan[-1] == "F2" else m
            plan.append(accel_forward(m2))
            return plan
        else:
            # Moving backward—must brake toward 0 first
            plan.append(full_stop(m) or "BB")
            return plan

    # 3) If left is open, mirror logic of right turn.
    if left_open:
        if abs(m) <= TARGET_TURN_SPEED and m > 0:
            if moving_rotation_allowed(m, "F1L"):
                plan.append("F1L")
                plan.append(accel_forward(m))
                return plan
            if moving_rotation_allowed(m, "F0L"):
                plan.append("F0L")
                plan.append(accel_forward(max(m-1, 0)))
                return plan
        if m > TARGET_TURN_SPEED:
            plan.append(brake_to_one(m) or "F0")
            return plan
        if m < 0:
            plan.append(full_stop(m) or "BB")
            return plan
        if m == 0:
            plan.append("L")
            plan.append("F2")
            return plan
        plan.append("F0")
        return plan

    # 4) Dead end → U-turn: brake to 0, then rotate twice.
    if m != 0:
        plan.append(full_stop(m) or "BB")
        return plan
    # momentum == 0 now: rotate 180 (two 45° turns twice)
    plan.extend(["R", "R", "R", "R"])  # 4 × 45° = 180°
    return plan[:3]  # keep it short per request to amortize thinking time

# ------------------------------
# API
# ------------------------------
@app.post("/micro-mouse")
def micro_mouse():
    # Always JSON only — never HTML (prevents '<!doctype html>' errors)
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"instructions": [], "end": True}), 200

    if not isinstance(payload, dict):
        return jsonify({"instructions": [], "end": True}), 200

    # Extract fields (robust defaults)
    sensor_data = payload.get("sensor_data") or [0, 0, 0, 0, 0]
    momentum = payload.get("momentum", 0)
    run_time_ms = payload.get("run_time_ms", 0)
    total_time_ms = payload.get("total_time_ms", 0)
    goal_reached = payload.get("goal_reached", False)

    # If we already reached the goal (and presumably stopped), end politely
    if goal_reached:
        return jsonify({"instructions": [], "end": True}), 200

    # Choose a small batch of safe instructions (2–4 tokens)
    try:
        instr = choose_instructions(sensor_data, momentum)
    except Exception:
        # Fail-safe: don’t move if we aren’t sure
        instr = []

    # Ensure a non-empty valid response (spec forbids empty/invalid instruction arrays)
    if not instr:
        # At rest → harmless default action (200ms); else brake
        instr = ["BB"] if momentum != 0 else ["F1"]

    # Cap batch size to reduce over-commit (and pay 50ms once per call)
    if len(instr) > 4:
        instr = instr[:4]

    return jsonify({
        "instructions": instr,
        "end": False
    }), 200

# Helpful JSON for accidental GETs (never HTML)
@app.get("/micro-mouse")
def micro_mouse_get():
    return jsonify({
        "message": "POST JSON to this endpoint.",
        "example_request": {
            "game_uuid": "demo",
            "sensor_data": [1, 1, 0, 1, 1],
            "total_time_ms": 0,
            "goal_reached": False,
            "best_time_ms": None,
            "run_time_ms": 0,
            "run": 0,
            "momentum": 0
        },
        "example_response": {"instructions": ["F2", "F2"], "end": False}
    }), 200

# JSON-only error handling
@app.errorhandler(HTTPException)
def handle_http_exc(e):
    return jsonify({"error": e.name, "status": e.code, "description": e.description}), e.code

@app.errorhandler(Exception)
def handle_exc(_):
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # Local run
    app.run(host="0.0.0.0", port=8000)
    # Deploy: gunicorn -w 2 -b 0.0.0.0:$PORT app:app
