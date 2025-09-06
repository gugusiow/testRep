from flask import Flask, request, jsonify
import os
import math
from collections import defaultdict
import random
from typing import List, Dict, Tuple


app = Flask(__name__)

@app.route('/trivia', methods=['GET'])
# def home():
#     return "Welcome to the Flask app!"

# @app.route('/data', methods=['POST'])
# def receive_data():
#     data = request.get_json()
#     return jsonify({"received": data}), 201

def get_trivia():
    # Answers to the trivia questions
    #result = {"answers": [2, 1, 2, 2, 3, 4, 3, 5, 4]}
    #result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 2, 1, 2, 1, 1]}
    #result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 2, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3, 5, 1]}
    # result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 2, 1, 2, 1, 1, 2, 3, 1, 3, 2, 3, 5, 1]}
    # result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 2, 1, 2, 1, 1, 2, 2, 1, 3, 2, 3, 5, 1]}
    result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 4, 1, 2, 1, 1, 2, 2, 1, 3, 2, 3, 4, 2]}

    return jsonify(result)

# Task1
# def calculate_distance(point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2
#     return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# def calculate_latency_points(distance):
#     points = max(0, 30 - distance)
#     return round(points)

# @app.route('/ticketing-agent', methods=['POST'])
# def ticketing_agent():
#     try:
#         data = request.get_json()
        
#         customers = data.get('customers', [])
#         concerts = data.get('concerts', [])
#         priority = data.get('priority', {})
        
#         recommendations = {}
        
#         for customer in customers:
#             customer_name = customer['name']
#             vip_status = customer['vip_status']
#             customer_location = (customer['location'][0], customer['location'][1])
#             credit_card = customer['credit_card']
            
#             max_points = -1
#             best_concert = None
            
#             for concert in concerts:
#                 concert_name = concert['name']
#                 booking_center = (concert['booking_center_location'][0], 
#                                  concert['booking_center_location'][1])
                
#                 points = 0 # Calculate points for this concert
                
#                 if vip_status:
#                     points += 100
                
#                 if credit_card in priority and priority[credit_card] == concert_name:
#                     points += 50
                
#                 distance = calculate_distance(customer_location, booking_center)
#                 latency_points = calculate_latency_points(distance)
#                 points += latency_points
                

#                 if points > max_points:
#                     max_points = points
#                     best_concert = concert_name
                
#             recommendations[customer_name] = best_concert
        
#         return jsonify(recommendations)
    
#     except Exception as e:
#        return jsonify({'error': str(e)}), 400

# @app.route('/sailing-club/submission', methods=['POST'])
# def sail_club():
#     try:
#         data = request.get_json()
        
#         for test_case in data['testCases']:
#             case_id = test_case.get('id')
#             input = test_case.get('input', [])

#             merged_result = []

#             intervals = sorted(input, key=lambda x: x[0])
#             merged = []
#             for interval in intervals:
#                 if not merged or merged[-1][1] < interval[0]:
#                     merged.append(interval[:])
#                 else:
#                     merged[-1][1] = max(merged[-1][1], interval[1])

#             # Sweep line to count overlaps
#             events = []
#             for start, end in input:
#                 events.append((start, 1))  # boat arrives
#                 events.append((end, -1))   # boat leaves
#             events.sort()

#             max_boats = 0
#             current_boats = 0
#             for _, change in events:
#                 current_boats += change
#                 max_boats = max(max_boats, current_boats)
            
#             merged_result.append({
#                 'id':case_id,
#                 'sortedMergedSlots': merged,
#                 'minBoatsNeeded': max_boats
#             })

#         result = {'solutions': merged_result}
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# Task 2
# from typing import List, Dict, Any, Optional, Tuple
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# import uvicorn

# app = FastAPI()

# def normalize_slots(slots: Any) -> Tuple[List[List[int]], Optional[str]]:
#     """
#     Validate and normalize the 'input' field to a list of [start, end] integer pairs.
#     Returns (normalized_slots, error_message_if_any).
#     """
#     if slots is None:
#         return [], None
#     if not isinstance(slots, list):
#         return [], "input must be a list"
    
#     norm: List[List[int]] = []
#     for idx, it in enumerate(slots):
#         if not (isinstance(it, (list, tuple)) and len(it) == 2):
#             return [], f"input[{idx}] must be a pair [start, end]"
        
#         s, e = it[0], it[1]
#         if not (isinstance(s, int) and isinstance(e, int)):
#             return [], f"input[{idx}] values must be integers"
        
#         # Handle cases where start > end (like [11, 8] in the sample)
#         if s > e:
#             # This represents a booking that wraps around midnight
#             # We'll split it into two intervals: [s, 4096] and [0, e]
#             # But for now, we'll keep it as is and handle during merging
#             pass
        
#         # Validate constraints: 0 <= hours <= 4096
#         if not (0 <= s <= 4096 and 0 <= e <= 4096):
#             # Skip invalid interval but keep processing others
#             continue
        
#         norm.append([s, e])
    
#     return norm, None

# def merge_slots(slots: List[List[int]]) -> List[List[int]]:
#     if not slots:
#         return []
    
#     # First, handle wrap-around intervals by splitting them
#     normalized_slots = []
#     for start, end in slots:
#         if start <= end:
#             normalized_slots.append([start, end])
#         else:
#             # Split wrap-around interval: [start, 4096] and [0, end]
#             normalized_slots.append([start, 4096])
#             normalized_slots.append([0, end])
    
#     # Sort by start time
#     normalized_slots.sort(key=lambda x: x[0])
    
#     merged: List[List[int]] = []
#     if not normalized_slots:
#         return merged
    
#     current_start, current_end = normalized_slots[0]
    
#     for i in range(1, len(normalized_slots)):
#         start, end = normalized_slots[i]
        
#         if start <= current_end + 1:  # +1 because intervals are inclusive
#             current_end = max(current_end, end)
#         else:
#             merged.append([current_start, current_end])
#             current_start, current_end = start, end
    
#     merged.append([current_start, current_end])
    
#     # Sort the final result
#     merged.sort(key=lambda x: x[0])
#     return merged

# def min_boats_needed(slots: List[List[int]]) -> int:
#     """
#     Calculate minimum boats needed by finding maximum overlapping intervals.
#     Uses a sweep-line algorithm.
#     """
#     if not slots:
#         return 0
    
#     # Handle wrap-around intervals by splitting them
#     events = []
#     for start, end in slots:
#         if start <= end:
#             events.append((start, 1))    # start event
#             events.append((end + 1, -1)) # end event (exclusive)
#         else:
#             # Split wrap-around interval
#             events.append((start, 1))
#             events.append((4097, -1))    # end of first part
#             events.append((0, 1))        # start of second part
#             events.append((end + 1, -1)) # end of second part
    
#     # Sort events: by time, and for same time, process ends first
#     events.sort(key=lambda x: (x[0], x[1]))
    
#     current_boats = 0
#     max_boats = 0
    
#     for time, event_type in events:
#         current_boats += event_type
#         max_boats = max(max_boats, current_boats)
    
#     return max_boats

# def solve_one(tc: Dict[str, Any]) -> Dict[str, Any]:
#     tc_id = tc.get("id", "")
#     slots_raw = tc.get("input", [])
    
#     # Normalize and validate slots
#     slots, error = normalize_slots(slots_raw)
    
#     # Always include both fields
#     merged = merge_slots(slots)
#     boats = min_boats_needed(slots)
    
#     return {
#         "id": tc_id,
#         "sortedMergedSlots": merged,
#         "minBoatsNeeded": boats
#     }

# @app.post("/sailing-club/submission")
# async def submission(request: Request):
#     try:
#         body = await request.json()
#     except Exception:
#         return JSONResponse(status_code=400, content={"error": "Invalid JSON in request body"})
    
#     test_cases = body.get("testCases", [])
#     if not isinstance(test_cases, list):
#         test_cases = []
    
#     solutions = []
#     for tc in test_cases:
#         if isinstance(tc, dict):
#             solutions.append(solve_one(tc))
#         else:
#             solutions.append({
#                 "id": "",
#                 "sortedMergedSlots": [],
#                 "minBoatsNeeded": 0
#             })
    
#     return JSONResponse(content={"solutions": solutions})

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


class SnakesLaddersSmokeMirrors:
    def __init__(self, board_size: int, players: int, jumps: List[str]):
        self.board_size = board_size
        self.players = players
        self.positions = [0] * players  # 0-based positions (0 to board_size-1)
        self.current_player = 0
        self.die_rolls = []
        
        # Parse jumps
        self.snakes = {}
        self.ladders = {}
        self.smoke = {}  # key: position, value: None (just need to know location)
        self.mirror = {}  # key: position, value: None
        
        for jump in jumps:
            parts = jump.split(':')
            from_pos = int(parts[0])
            to_pos = int(parts[1]) if parts[1] != '0' else '0'
            
            if to_pos == 0:  # Smoke (0 in second position)
                self.smoke[from_pos - 1] = None  # Convert to 0-based
            elif from_pos == 0:  # Mirror (0 in first position)
                self.mirror[to_pos - 1] = None  # Convert to 0-based
            elif from_pos > to_pos:  # Snake
                self.snakes[from_pos - 1] = to_pos - 1  # Convert to 0-based
            else:  # Ladder
                self.ladders[from_pos - 1] = to_pos - 1  # Convert to 0-based
    
    def move_player(self, roll: int) -> bool:
        """Move current player with given roll, return True if game ended"""
        player_pos = self.positions[self.current_player]
        
        # Calculate new position
        new_pos = player_pos + roll
        
        # Check for overshoot
        if new_pos >= self.board_size:
            overshoot = new_pos - (self.board_size - 1)
            new_pos = (self.board_size - 1) - overshoot
        
        # Check for special squares
        while True:
            moved = False
            
            # Check snakes
            if new_pos in self.snakes:
                new_pos = self.snakes[new_pos]
                moved = True
            
            # Check ladders
            if new_pos in self.ladders:
                new_pos = self.ladders[new_pos]
                moved = True
            
            # Check smoke (needs extra die roll)
            if new_pos in self.smoke:
                if not self.die_rolls:  # No more die rolls available
                    return False
                extra_roll = self.die_rolls.pop(0)
                new_pos = max(0, new_pos - extra_roll)  # Move backwards
                moved = True
            
            # Check mirror (needs extra die roll)
            if new_pos in self.mirror:
                if not self.die_rolls:  # No more die rolls available
                    return False
                extra_roll = self.die_rolls.pop(0)
                new_pos = min(self.board_size - 1, new_pos + extra_roll)  # Move forwards
                moved = True
            
            if not moved:
                break
        
        # Update position
        self.positions[self.current_player] = new_pos
        
        # Check if player won
        if new_pos == self.board_size - 1:
            return True
        
        # Move to next player
        self.current_player = (self.current_player + 1) % self.players
        return False
    
    def simulate_with_rolls(self, rolls: List[int]) -> Tuple[bool, int]:
        """Simulate game with given rolls, return (success, squares_visited)"""
        self.positions = [0] * self.players
        self.current_player = 0
        self.die_rolls = rolls.copy()
        
        total_squares = 0
        game_ended = False
        
        while self.die_rolls and not game_ended:
            roll = self.die_rolls.pop(0)
            game_ended = self.move_player(roll)
            total_squares += 1
        
        # Check if last player won
        success = game_ended and self.current_player == self.players - 1
        return success, total_squares
    
    def solve_optimized(self) -> List[int]:
        """Generate optimal die rolls using a smarter strategy"""
        # This is a more sophisticated approach that tries to minimize squares visited
        best_rolls = []
        best_score = 0
        
        # Try multiple strategies
        strategies = [
            self._generate_direct_rolls,
            self._generate_safe_rolls,
            self._generate_random_rolls
        ]
        
        for strategy in strategies:
            for _ in range(100):  # Try each strategy multiple times
                rolls = strategy()
                success, squares_visited = self.simulate_with_rolls(rolls)
                
                if success:
                    score = self.board_size / squares_visited
                    if score > best_score:
                        best_score = score
                        best_rolls = rolls
        
        return best_rolls if best_rolls else [1, 2, 3, 4, 5, 6] * 10
    
    def _generate_direct_rolls(self) -> List[int]:
        """Generate rolls that move directly toward the end"""
        rolls = []
        target = self.board_size - 1
        
        for player in range(self.players):
            # For each player, generate rolls to reach the end
            current_pos = 0
            while current_pos < target:
                needed = target - current_pos
                roll = min(6, needed)
                rolls.append(roll)
                current_pos += roll
        
        return rolls
    
    def _generate_safe_rolls(self) -> List[int]:
        """Generate safe rolls avoiding problematic squares"""
        rolls = []
        target = self.board_size - 1
        
        # Identify problematic squares to avoid
        problematic_squares = set(self.snakes.keys()) | set(self.smoke.keys())
        
        for player in range(self.players):
            current_pos = 0
            while current_pos < target:
                # Try to avoid problematic squares
                best_roll = 1
                best_next_pos = current_pos + 1
                
                for roll in range(1, 7):
                    next_pos = current_pos + roll
                    if next_pos > target:
                        # Handle overshoot
                        overshoot = next_pos - target
                        next_pos = target - overshoot
                    
                    # Check if this position is better
                    if next_pos not in problematic_squares:
                        best_roll = roll
                        best_next_pos = next_pos
                        break
                
                rolls.append(best_roll)
                current_pos = best_next_pos
        
        return rolls
    
    def _generate_random_rolls(self) -> List[int]:
        """Generate random rolls"""
        return [random.randint(1, 6) for _ in range(self.players * 20)]

@app.route('/slpu', methods=['POST'])   # was slsm...
def handle_slsm():
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'boardSize' not in data or 'players' not in data or 'jumps' not in data:
            return jsonify({'error': 'Invalid input format'}), 400
        
        board_size = data['boardSize']
        players = data['players']
        jumps = data['jumps']
        
        # Validate constraints
        if not (8*8 <= board_size <= 20*20) or not (2 <= players <= 8):
            return jsonify({'error': 'Constraints not satisfied'}), 400
        
        # Solve the game
        game = SnakesLaddersSmokeMirrors(board_size, players, jumps)
        die_rolls = game.solve_optimized()
        
        return jsonify(die_rolls)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Persistent per-game state (in-memory)
# In production, replace with external store keyed by game_uuid
games = defaultdict(lambda: {
    "heading_deg": 0,         # 0=N, 90=E, 180=S, 270=W, with 45° steps permitted
    "last_run": 0,
    "last_best": None,
})

# Constants from spec
MAX_MOMENTUM = 4
MIN_MOMENTUM = -4

# Helper functions

def clamp_heading(h):
    # Normalize to [0, 360) in 45° increments
    h = ((h % 360) + 360) % 360
    # Round to nearest 45 to stay on the grid of allowed rotations
    return (round(h / 45) * 45) % 360

def sensor_to_walls(sensor):
    # sensor indices: [-90, -45, 0, +45, +90]; 1 means wall within 12 cm, 0 means clear
    # We’ll interpret 1 as blocked, 0 as open.
    return {
        -90: sensor[0] == 1,
        -45: sensor[1] == 1,
        0:   sensor[2] == 1,
        45:  sensor[3] == 1,
        90:  sensor[4] == 1,
    }

def is_cardinal(h):
    return h % 90 == 0

def left_hand_preference(walls):
    # Return desired relative direction priority: left, forward-left, forward, forward-right, right
    # Only use forward for straight moves; turns handled explicitly.
    # We’ll pick the first open direction.
    order = [-90, -45, 0, 45, 90]
    for rel in order:
        if not walls[rel]:
            return rel
    # Dead-end: everything blocked -> we must brake/stop and rotate
    return None

def plan_tokens(momentum, heading_deg, sensor):
    """
    Return a list of safe tokens as a small batch.
    Strategy:
    - Prefer left if open. Otherwise go forward if open. Otherwise try right. Otherwise stop and rotate.
    - Use in-place rotation only at momentum 0.
    - Use F2 to accelerate up to +2, then hold with F1; brake with BB as needed.
    - Avoid illegal moving rotations by checking m_eff <= 1.
    """
    walls = sensor_to_walls(sensor)
    tokens = []

    # If goal already reached or we’re at rest and front blocked with all around blocked -> rotate to find open
    # Decide relative direction
    rel = left_hand_preference(walls)

    # If moving backward (negative momentum), first brake to 0
    if momentum < 0:
        # BB reduces by 2 toward 0; legal and adds half-step translation in the reverse direction
        tokens.append("BB")
        return tokens

    # If we want to turn left/right from a cardinal heading using corner turns where safe
    def try_corner(turn_dir):
        # turn_dir: 'L' or 'R'
        # We will attempt a tight corner with F1 or F0 so that m_eff <= 1.
        # Tight requires m_eff <= 1.
        # Choose F1 if already at +1, else F2/F1 sequence to get to +1 at entry then corner,
        # but we must keep batch small. We'll do:
        # - If momentum == 0: F2 to +1 straight half-step, then corner F1T{L/R}.
        #   However corner token format is (F?|V?)(L|R)(T|W)[(L|R)], applied immediately,
        #   so we can do: "F1" to get/hold +1, then "(F1)(L|R)T".
        # Simpler: at momentum 0: do in-place rotation then go straight; but that costs extra 200 ms.
        # We'll opt for safe and simple: if momentum == 0 -> in-place rotate 45°, then accelerate straight.
        nonlocal tokens
        if momentum == 0:
            tokens.append(turn_dir)  # in-place 45°
            return True
        # momentum > 0: consider moving rotation with F0L/F0R or F1L/F1R:
        # Moving rotations allowed only if m_eff <= 1.
        # If momentum == 1 and we use F1L, m_eff = (1 + 1)/2 = 1 -> allowed.
        # If momentum == 2 and we use F0L, exit momentum 1 -> m_eff = (2+1)/2 = 1.5 -> illegal.
        if momentum == 1:
            tokens.append(f"F1{turn_dir}")
            return True
        if momentum == 0:
            tokens.append(f"F1{turn_dir}")
            return True
        # Otherwise, we need to reduce momentum with BB until <=1 before moving rotation
        tokens.append("BB")
        return True

    # Decide action based on rel preference
    if rel is None:
        # Dead-end: brake to 0; if already 0, rotate 90° left via two L to seek exit
        if momentum > 0:
            tokens.append("BB")  # will move half-step forward while braking
        else:
            # momentum == 0
            tokens.extend(["L", "L"])
        return tokens

    if rel == 0:
        # Forward is open. Accelerate up to +2 then hold.
        if momentum < 2:
            tokens.append("F2")
        else:
            tokens.append("F1")
        return tokens

    if rel in (-45, -90):
        # Prefer to turn left, tighter first
        if rel == -45:
            turn = "L"
            did = try_corner(turn)
            if did:
                return tokens
        else:
            # -90 means harder left; do two steps of left: either two moving rotations or two in-place if at 0
            # Keep it simple and safe:
            if momentum == 0:
                tokens.extend(["L", "L"])
            elif momentum == 1:
                tokens.append("F1L")  # now heading -45
            else:
                tokens.append("BB")
            return tokens

    if rel in (45, 90):
        # Need to turn right
        if rel == 45:
            turn = "R"
            did = try_corner(turn)
            if did:
                return tokens
        else:
            # +90
            if momentum == 0:
                tokens.extend(["R", "R"])
            elif momentum == 1:
                tokens.append("F1R")
            else:
                tokens.append("BB")
            return tokens

    # Fallback: hold if forward looks fine, else brake
    if not walls[0]:
        tokens.append("F1" if momentum >= 1 else "F2")
    else:
        tokens.append("BB")
    return tokens


@app.route("/micro-mouse", methods=["POST"])
def micro_mouse():
    data = request.get_json(force=True)

    # Basic validation per spec
    required = ["game_uuid", "sensor_data", "total_time_ms", "goal_reached",
                "best_time_ms", "run_time_ms", "run", "momentum"]
    if not isinstance(data, dict) or any(k not in data for k in required):
        # Returning an empty or invalid instructions array would crash the simulator per spec,
        # but as a controller we should still return a valid response. We'll send an 'end': true.
        return jsonify({"instructions": [], "end": True})

    game_id = data["game_uuid"]
    sensor = data["sensor_data"]
    momentum = int(data["momentum"])
    run = int(data["run"])
    goal_reached = bool(data["goal_reached"])
    best_time_ms = data["best_time_ms"]

    # Track per game state for simple heading bookkeeping (not strictly required)
    st = games[game_id]
    if st["last_run"] != run:
        # New run started
        st["last_run"] = run
    if best_time_ms is not None:
        st["last_best"] = best_time_ms

    # If challenge should end (e.g., after reaching goal and getting a best), you could decide to end.
    # Here we keep going until the simulator ends us or caller decides.
    if goal_reached:
        # Stop issuing commands for safety; end the challenge to record the score.
        return jsonify({"instructions": [], "end": True})

    # Plan a small batch of tokens to amortize the 50 ms thinking time.
    # Keep batch size modest to remain responsive.
    try:
        tokens = plan_tokens(momentum, st["heading_deg"], sensor)
    except Exception:
        # On any planner error, end safely
        return jsonify({"instructions": [], "end": True})

    # Ensure we never send an empty instructions array, which would crash per spec.
    if not tokens:
        tokens = ["F1"] if momentum >= 1 else ["F2"]

    # Update our internal heading approximation based on rotations we requested.
    # Note: We only track in-place 45° turns here; moving rotations and corners are not
    # simulated locally because we don't know if they succeed until sensors update next call.
    for t in tokens:
        if t == "L":
            st["heading_deg"] = clamp_heading(st["heading_deg"] - 45)
        elif t == "R":
            st["heading_deg"] = clamp_heading(st["heading_deg"] + 45)
        # We skip heading updates for moving rotations and corners; the next sensor input will reflect reality.

    return jsonify({
        "instructions": tokens,
        "end": False
    })

    
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000)) # Get the PORT env var, default to 5000 for local run
    app.run(host='0.0.0.0', port=port) # You MUST set host to '0.0.0.0'