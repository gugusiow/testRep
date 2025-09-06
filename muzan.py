# app.py
from flask import Flask, request, jsonify
from collections import defaultdict
import os

app = Flask(__name__)
app.url_map.strict_slashes = False

# ----------------------------
# In-memory game state
# ----------------------------
# games[(challenger_id, game_id)] = {
#   "grid_n": int,
#   "num_walls": int,
#   "crows": { crow_id: {"x": int, "y": int} },
#   "walls": set[(x,y)],
#   "scanned_centers": set[(x,y)],   # positions where we've scanned
#   "boustro_row": dict[crow_id -> int],   # current row assignment for raster scan
#   "boustro_dir": dict[crow_id -> int],   # +1 for east, -1 for west
# }
games = {}

# ----------------------------
# Helpers
# ----------------------------
def key_xy(x, y):
    return f"{x}-{y}"

def in_bounds(x, y, n):
    return 0 <= x < n and 0 <= y < n

def parse_initial_test_case(body):
    """Extract initial test_case payload."""
    test = body.get("test_case", {}) or {}
    crows = test.get("crows", []) or []
    grid_n = int(test.get("length_of_grid", 0))
    num_walls = int(test.get("num_of_walls", 0))
    return crows, grid_n, num_walls

def apply_move_result(state, prev):
    """Update crow position after a move."""
    cid = str(prev["crow_id"])
    x, y = prev["move_result"]
    # Clamp to bounds just in case input is noisy
    n = state["grid_n"]
    x = max(0, min(n-1, int(x)))
    y = max(0, min(n-1, int(y)))
    state["crows"][cid]["x"] = x
    state["crows"][cid]["y"] = y

def apply_scan_result(state, prev):
    """Ingest a 5x5 scan centered at the crow; record walls and mark scanned center."""
    cid = str(prev["crow_id"])
    scan = prev["scan_result"]
    cx = state["crows"][cid]["x"]
    cy = state["crows"][cid]["y"]
    n = state["grid_n"]

    # Mark that we scanned this center already
    state["scanned_centers"].add((cx, cy))

    # scan is a 5x5 grid; indices (0..4), center is (2,2)
    # Translate each cell to absolute (x,y): dx = j-2, dy = i-2
    for i in range(5):
        for j in range(5):
            sym = scan[i][j]
            ax = cx + (j - 2)
            ay = cy + (i - 2)
            if sym == "X":
                continue  # out-of-bounds
            if not in_bounds(ax, ay, n):
                continue
            if sym == "W":
                state["walls"].add((ax, ay))
            # "_" is empty; "C" is crow center; we don't need to store empties.

def ensure_game(body):
    """Initialize or fetch the state for (challenger_id, game_id)."""
    challenger_id = str(body.get("challenger_id"))
    game_id = str(body.get("game_id"))
    if not challenger_id or not game_id:
        return None, "Missing challenger_id or game_id"

    gkey = (challenger_id, game_id)
    if gkey not in games:
        # First message of this test case must contain test_case
        crows, grid_n, num_walls = parse_initial_test_case(body)
        if not grid_n or not isinstance(crows, list) or len(crows) == 0:
            return None, "Initial request must include test_case with crows and length_of_grid"

        state = {
            "grid_n": grid_n,
            "num_walls": num_walls,
            "crows": { str(c["id"]): {"x": int(c["x"]), "y": int(c["y"])} for c in crows },
            "walls": set(),
            "scanned_centers": set(),
            "boustro_row": {},
            "boustro_dir": {},
        }
        # Assign rows / directions for a simple multi-crow raster:
        # crow 1 starts at its current row, moves east; crow 2 starts next row, moves west; crow 3 next, east; etc.
        row_assign = 0
        dir_sign = +1
        for cid in sorted(state["crows"].keys()):
            state["boustro_row"][cid] = state["crows"][cid]["y"]
            state["boustro_dir"][cid] = dir_sign
            dir_sign *= -1  # alternate E/W for coverage
            row_assign += 1
        games[gkey] = state
    else:
        state = games[gkey]
    return state, None

def pick_crow_to_act(state):
    """
    Simple policy:
    1) If any crow is on a center we haven't scanned yet, scan with that crow.
    2) Otherwise, move a crow along its boustrophedon (lawnmower) path.
    """
    # 1) Prefer scanning at unscanned center
    for cid, pos in state["crows"].items():
        if (pos["x"], pos["y"]) not in state["scanned_centers"]:
            return cid, "scan", None

    # 2) Move following raster (per-crow row with E/W sweeps)
    # Try each crow and return a single move for the first feasible one
    for cid, pos in state["crows"].items():
        n = state["grid_n"]
        x, y = pos["x"], pos["y"]
        dir_sign = state["boustro_dir"][cid]

        # If not on its assigned row, move vertically to that row first
        target_row = state["boustro_row"][cid]
        if y < target_row:
            return cid, "move", "S"
        if y > target_row:
            return cid, "move", "N"

        # On the target row: sweep left->right or right->left
        if dir_sign == +1:
            # Move East until x == n-1, then drop one row (S) and flip direction
            if x < n - 1:
                return cid, "move", "E"
            else:
                # at east edge: drop a row if possible; else, flip row assignment upward
                if y < n - 1:
                    state["boustro_row"][cid] = y + 1
                    state["boustro_dir"][cid] = -1
                    return cid, "move", "S"
                else:
                    # bottom-right corner; flip to go up (optional)
                    state["boustro_row"][cid] = max(0, y - 1)
                    state["boustro_dir"][cid] = -1
                    return cid, "move", "N"
        else:
            # dir_sign == -1 => Move West until x == 0
            if x > 0:
                return cid, "move", "W"
            else:
                if y < n - 1:
                    state["boustro_row"][cid] = y + 1
                    state["boustro_dir"][cid] = +1
                    return cid, "move", "S"
                else:
                    state["boustro_row"][cid] = max(0, y - 1)
                    state["boustro_dir"][cid] = +1
                    return cid, "move", "N"

    # Fallback (shouldn't happen): just act with the first crow
    first_cid = next(iter(state["crows"]))
    return first_cid, "scan", None

def direction_from_delta(dx, dy):
    if dx == 0 and dy == -1: return "N"
    if dx == 0 and dy ==  1: return "S"
    if dx == 1 and dy ==  0: return "E"
    if dx == -1 and dy == 0: return "W"
    return None

# ----------------------------
# Core endpoint
# ----------------------------
@app.post("/fog-of-wall")
def fog_of_wall():
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "invalid body"}), 400

    # Create or fetch game state
    state, err = ensure_game(body)
    if err:
        return jsonify({"error": err}), 400

    challenger_id = str(body.get("challenger_id"))
    game_id = str(body.get("game_id"))
    prev = body.get("previous_action")

    # Ingest previous action result (if any)
    if prev:
        your_action = prev.get("your_action")
        if your_action == "move" and "move_result" in prev:
            apply_move_result(state, prev)
        elif your_action == "scan" and "scan_result" in prev:
            apply_scan_result(state, prev)

    # If we have found all walls, submit
    if len(state["walls"]) >= state["num_walls"] > 0:
        submission = [key_xy(x, y) for (x, y) in sorted(state["walls"])]
        return jsonify({
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": submission
        })

    # Policy: prefer scanning at any unscanned center, else move along raster
    crow_id, action_type, maybe_dir = pick_crow_to_act(state)

    if action_type == "scan":
        return jsonify({
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": crow_id,
            "action_type": "scan"
        })

    # action_type == "move"
    return jsonify({
        "challenger_id": challenger_id,
        "game_id": game_id,
        "crow_id": crow_id,
        "action_type": "move",
        "direction": maybe_dir
    })

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
