# app.py
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
app.url_map.strict_slashes = False

# -------------------------------------------------
# In-memory game state (single-process only)
# -------------------------------------------------
# games[game_id] = {
#   "grid_n": int,
#   "num_walls": int,
#   "crows": { crow_id: {"x": int, "y": int} },
#   "walls": set[(x,y)],
#   "scanned_centers": set[(x,y)],
#   "boustro_row": {crow_id: int},
#   "boustro_dir": {crow_id: int},   # +1 east, -1 west
#   "actions": int                    # number of actions taken so far
# }
games = {}

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def key_xy(x, y):
    return f"{x}-{y}"

def in_bounds(x, y, n):
    return 0 <= x < n and 0 <= y < n

def parse_initial_test_case(body):
    test = body.get("test_case", {}) or {}
    crows = test.get("crows", []) or []
    grid_n = int(test.get("length_of_grid", 0))
    num_walls = int(test.get("num_of_walls", 0))
    return crows, grid_n, num_walls

def ensure_game(body):
    """
    Initialize or fetch the state for game_id (keyed ONLY by game_id to avoid challenger_id churn).
    """
    game_id = str(body.get("game_id"))
    if not game_id:
        return None, "Missing game_id"

    if game_id not in games:
        # First message must include test_case
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
            "actions": 0,
        }
        # Simple multi-crow raster: alternate east/west start directions
        dir_sign = +1
        for cid in sorted(state["crows"].keys()):
            state["boustro_row"][cid] = state["crows"][cid]["y"]
            state["boustro_dir"][cid] = dir_sign
            dir_sign *= -1
        games[game_id] = state
    else:
        state = games[game_id]
    return state, None

def apply_move_result(state, prev):
    """
    Update crow position after a move.
    If the crow didn't move, infer a wall at the attempted target cell and adjust raster.
    """
    cid = str(prev["crow_id"])
    direction = prev.get("direction")
    oldx, oldy = state["crows"][cid]["x"], state["crows"][cid]["y"]
    nx, ny = prev["move_result"]
    n = state["grid_n"]
    nx = max(0, min(n-1, int(nx)))
    ny = max(0, min(n-1, int(ny)))

    # No movement -> bumped into a wall: infer target cell
    if (nx, ny) == (oldx, oldy) and direction:
        tx, ty = oldx, oldy
        if direction == "N": ty -= 1
        elif direction == "S": ty += 1
        elif direction == "E": tx += 1
        elif direction == "W": tx -= 1
        if in_bounds(tx, ty, n):
            state["walls"].add((tx, ty))
            # Adjust raster to avoid infinite bumping
            if direction in ("E", "W"):
                y = state["crows"][cid]["y"]
                if y < n - 1:
                    state["boustro_row"][cid] = y + 1
                elif y > 0:
                    state["boustro_row"][cid] = y - 1
                state["boustro_dir"][cid] *= -1
            else:
                # Hit a wall north/south; flip horizontal sweep as a nudge
                state["boustro_dir"][cid] *= -1

    # Update crow position
    state["crows"][cid]["x"] = nx
    state["crows"][cid]["y"] = ny

def apply_scan_result(state, prev):
    """
    Ingest a 5x5 scan centered at the crow; record walls and mark scanned center.
    """
    cid = str(prev["crow_id"])
    scan = prev["scan_result"]
    cx = state["crows"][cid]["x"]
    cy = state["crows"][cid]["y"]
    n = state["grid_n"]

    state["scanned_centers"].add((cx, cy))

    # scan[i][j], i=row (y), j=col (x); center is (2,2)
    for i in range(5):
        for j in range(5):
            sym = scan[i][j]
            ax = cx + (j - 2)
            ay = cy + (i - 2)
            if sym == "X":
                continue
            if not in_bounds(ax, ay, n):
                continue
            if sym == "W":
                state["walls"].add((ax, ay))
            # "_" and "C" don't need to be recorded

def should_scan_here(x, y):
    """
    Tile scans to cover the grid efficiently.
    Since scan covers Chebyshev radius 2, scanning every 2 cells gives heavy overlap but is simple.
    You can change to modulo 3 for sparser coverage if needed.
    """
    return (x % 2 == 0) and (y % 2 == 0)

def next_step_avoiding_known_walls(state, cid):
    """
    Determine next move for crow cid in boustrophedon pattern,
    while avoiding cells already known to be walls.
    """
    n = state["grid_n"]
    x = state["crows"][cid]["x"]
    y = state["crows"][cid]["y"]
    dir_sign = state["boustro_dir"][cid]
    target_row = state["boustro_row"][cid]

    def is_wall(xy):
        return xy in state["walls"]

    # Move vertically to assigned row first (avoid walls if known)
    if y < target_row:
        if not is_wall((x, y + 1)) and in_bounds(x, y + 1, n):
            return "S"
        # If blocked, try horizontal detour
        if dir_sign == +1 and x + 1 < n and not is_wall((x + 1, y)):
            return "E"
        if dir_sign == -1 and x - 1 >= 0 and not is_wall((x - 1, y)):
            return "W"
        # Fallback: opposite horizontal
        if dir_sign == +1 and x - 1 >= 0 and not is_wall((x - 1, y)):
            return "W"
        if dir_sign == -1 and x + 1 < n and not is_wall((x + 1, y)):
            return "E"
        # Last resort: try N if not out of bounds
        if y - 1 >= 0 and not is_wall((x, y - 1)):
            return "N"
        return "S"  # let server tell us if blocked; we will learn wall

    if y > target_row:
        if not is_wall((x, y - 1)) and in_bounds(x, y - 1, n):
            return "N"
        if dir_sign == +1 and x + 1 < n and not is_wall((x + 1, y)):
            return "E"
        if dir_sign == -1 and x - 1 >= 0 and not is_wall((x - 1, y)):
            return "W"
        if dir_sign == +1 and x - 1 >= 0 and not is_wall((x - 1, y)):
            return "W"
        if dir_sign == -1 and x + 1 < n and not is_wall((x + 1, y)):
            return "E"
        if y + 1 < n and not is_wall((x, y + 1)):
            return "S"
        return "N"

    # On the target row: sweep E/W
    if dir_sign == +1:
        # Prefer E while inside bounds and not known wall
        if x + 1 < n and not is_wall((x + 1, y)):
            return "E"
        # At east edge or blocked → go down a row if possible and flip
        if y + 1 < n and not is_wall((x, y + 1)):
            state["boustro_row"][cid] = y + 1
            state["boustro_dir"][cid] = -1
            return "S"
        # Otherwise try up and flip
        if y - 1 >= 0 and not is_wall((x, y - 1)):
            state["boustro_row"][cid] = y - 1
            state["boustro_dir"][cid] = -1
            return "N"
        # Fallback: still attempt E (we'll learn the wall if blocked)
        return "E"
    else:
        # dir_sign == -1: Prefer W
        if x - 1 >= 0 and not is_wall((x - 1, y)):
            return "W"
        if y + 1 < n and not is_wall((x, y + 1)):
            state["boustro_row"][cid] = y + 1
            state["boustro_dir"][cid] = +1
            return "S"
        if y - 1 >= 0 and not is_wall((x, y - 1)):
            state["boustro_row"][cid] = y - 1
            state["boustro_dir"][cid] = +1
            return "N"
        return "W"

def pick_crow_to_act(state):
    """
    Policy:
    1) If any crow is at a strategic center not yet scanned, scan there.
    2) Otherwise, move one crow following boustrophedon while avoiding known walls.
    """
    # 1) Prefer scan at spaced centers
    for cid, pos in state["crows"].items():
        x, y = pos["x"], pos["y"]
        if should_scan_here(x, y) and (x, y) not in state["scanned_centers"]:
            return cid, "scan", None

    # 2) Move one crow along its raster
    for cid in state["crows"].keys():
        direction = next_step_avoiding_known_walls(state, cid)
        return cid, "move", direction

    # Fallback: scan with first crow
    first_cid = next(iter(state["crows"]))
    return first_cid, "scan", None

# -------------------------------------------------
# Core endpoint
# -------------------------------------------------
@app.post("/fog-of-wall")
def fog_of_wall():
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "invalid body"}), 400

    challenger_id = str(body.get("challenger_id", ""))
    game_id = str(body.get("game_id", ""))
    if not game_id:
        return jsonify({"error": "missing game_id"}), 400

    # Create or fetch state
    state, err = ensure_game(body)
    if err:
        return jsonify({"error": err}), 400

    prev = body.get("previous_action")

    # Ingest previous action result (if any)
    if prev:
        your_action = prev.get("your_action")
        if your_action == "move" and "move_result" in prev:
            apply_move_result(state, prev)
        elif your_action == "scan" and "scan_result" in prev:
            apply_scan_result(state, prev)

    # Increment action counter and consider a conservative submit guard
    state["actions"] += 1
    n = state["grid_n"]
    move_cap = n * n  # judge may stop at size_of_grid^2
    # Submit early if we know the exact count or are near cap
    if state["num_walls"] == 0 or state["actions"] >= max(1, move_cap - 1):
        submission = [key_xy(x, y) for (x, y) in sorted(state["walls"])]
        return jsonify({
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": submission
        })

    # If we have found all declared walls, submit now
    if state["num_walls"] > 0 and len(state["walls"]) >= state["num_walls"]:
        submission = [key_xy(x, y) for (x, y) in sorted(state["walls"])]
        return jsonify({
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": submission
        })

    # Decide next action
    crow_id, action_type, direction = pick_crow_to_act(state)

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
        "direction": direction
    })

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    # For Render/Heroku-style envs
    port = int(os.environ.get("PORT", 5000))
    # Single process recommended for in-memory state
    app.run(host="0.0.0.0", port=port)
