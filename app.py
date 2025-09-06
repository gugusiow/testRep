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



games = defaultdict(lambda: {
    "heading_deg": 0,
    "last_run": 0,
    "last_best": None,
})

# Helpers

def clamp_heading(h):
    h = ((h % 360) + 360) % 360
    return (round(h / 45) * 45) % 360

def walls_from_sensor(sensor):
    # sensor: [-90, -45, 0, +45, +90] with 1 = blocked (<=12 cm), 0 = clear
    return {
        "L90": sensor[0] == 1,
        "L45": sensor[1] == 1,
        "F":   sensor[2] == 1,
        "R45": sensor[3] == 1,
        "R90": sensor[4] == 1,
    }

def left_hand_choice(w):
    # Priority: L90, L45, F, R45, R90 (first open)
    if not w["L90"]: return -90
    if not w["L45"]: return -45
    if not w["F"]:   return 0
    if not w["R45"]: return 45
    if not w["R90"]: return 90
    return None

def plan_safe_tokens(momentum, sensor):
    w = walls_from_sensor(sensor)
    tokens = []

    # 1) If moving backward, always brake toward 0 first
    if momentum < 0:
        tokens.append("BB")
        return tokens

    # 2) Dead-end: everything near blocked
    rel = left_hand_choice(w)
    if rel is None:
        # If moving forward (momentum > 0), BB once (half-step forward) can collide if F is blocked.
        # To avoid collision: if F is blocked and momentum > 0, prefer reducing momentum via BB
        # but note BB still advances half-step forward. If front is blocked, first try rotate in place at 0.
        if momentum > 0:
            # Safer approach: if momentum >= 2, BB will still move forward half-step.
            # We cannot avoid the half-step; so reduce momentum as soon as possible.
            tokens.append("BB")
        else:
            # momentum == 0: rotate 90° left to search
            tokens.extend(["L", "L"])
        return tokens

    # 3) If we need to turn, avoid advancing into a blocked front
    def can_moving_rotation(turn):
        # Only allow when momentum == 1 and the respective diagonal is open,
        # and also front is not blocked to reduce corner clipping risk.
        if momentum != 1:
            return False
        if w["F"]:
            return False
        if turn == "L":
            return not w["L45"]
        else:
            return not w["R45"]

    # 4) Actions by relative choice
    if rel == 0:
        # Straight
        if w["F"]:
            # Front blocked -> do not advance. Handle turn instead: prefer left.
            if momentum > 0:
                tokens.append("BB")  # slow down first
            else:
                # rotate in place to seek opening; prefer left
                if not w["L90"] or not w["L45"]:
                    tokens.append("L")
                else:
                    tokens.append("R")
            return tokens
        # Front clear: accelerate up to +2, then hold
        if momentum < 2:
            tokens.append("F2")
        else:
            tokens.append("F1")
        return tokens

    if rel in (-45, -90):
        # Want to go left
        if rel == -90:
            # 90° left: do in-place if at 0; otherwise reduce speed
            if momentum == 0:
                tokens.extend(["L", "L"])
            elif can_moving_rotation("L"):
                tokens.append("F1L")  # -45 first; next cycle we can do another F1L or L
            else:
                tokens.append("BB")
            return tokens
        else:  # -45
            if can_moving_rotation("L"):
                tokens.append("F1L")
            else:
                if momentum == 0:
                    tokens.append("L")
                else:
                    tokens.append("BB")
            return tokens

    if rel in (45, 90):
        # Want to go right
        if rel == 90:
            if momentum == 0:
                tokens.extend(["R", "R"])
            elif can_moving_rotation("R"):
                tokens.append("F1R")
            else:
                tokens.append("BB")
            return tokens
        else:  # 45
            if can_moving_rotation("R"):
                tokens.append("F1R")
            else:
                if momentum == 0:
                    tokens.append("R")
                else:
                    tokens.append("BB")
            return tokens

    # Fallbacks
    if not w["F"]:
        tokens.append("F1" if momentum >= 2 else "F2")
    else:
        # Front blocked; reduce speed or rotate
        if momentum > 0:
            tokens.append("BB")
        else:
            tokens.append("L")
    return tokens

@app.route("/micro-mouse", methods=["POST"])
def micro_mouse():
    data = request.get_json(force=True)
    required = ["game_uuid", "sensor_data", "total_time_ms", "goal_reached",
                "best_time_ms", "run_time_ms", "run", "momentum"]
    if not isinstance(data, dict) or any(k not in data for k in required):
        return jsonify({"instructions": [], "end": True})

    gid = data["game_uuid"]
    sensor = data["sensor_data"]
    momentum = int(data["momentum"])
    run = int(data["run"])
    goal_reached = bool(data["goal_reached"])
    best_time_ms = data["best_time_ms"]

    st = games[gid]
    if st["last_run"] != run:
        st["last_run"] = run
    if best_time_ms is not None:
        st["last_best"] = best_time_ms

    if goal_reached:
        return jsonify({"instructions": [], "end": True})

    try:
        tokens = plan_safe_tokens(momentum, sensor)
    except Exception:
        return jsonify({"instructions": [], "end": True})

    if not tokens:
        tokens = ["F1"] if momentum >= 1 and sensor[2] == 0 else ["L"]

    # Update heading for in-place L/R only
    for t in tokens:
        if t == "L":
            st["heading_deg"] = clamp_heading(st["heading_deg"] - 45)
        elif t == "R":
            st["heading_deg"] = clamp_heading(st["heading_deg"] + 45)

    return jsonify({"instructions": tokens, "end": False})


    
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000)) # Get the PORT env var, default to 5000 for local run
    app.run(host='0.0.0.0', port=port) # You MUST set host to '0.0.0.0'