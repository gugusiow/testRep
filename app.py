# from flask import Flask, request, jsonify
# import os
# import math
# from collections import defaultdict

# app = Flask(__name__)

# @app.route('/trivia', methods=['GET'])
# def home():
#     return "Welcome to the Flask app!"

# @app.route('/data', methods=['POST'])
# def receive_data():
#     data = request.get_json()
#     return jsonify({"received": data}), 201

# def get_trivia():
#     # Answers to the trivia questions
#     #result = {"answers": [2, 1, 2, 2, 3, 4, 3, 5, 4]}
#     #result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 2, 1, 2, 1, 1]}
#     result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 2, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3, 5, 1]}
    
#     return jsonify(result)

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

from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

def normalize_slots(slots: Any) -> Tuple[List[List[int]], Optional[str]]:
    """
    Validate and normalize the 'input' field to a list of [start, end] integer pairs.
    Returns (normalized_slots, error_message_if_any).
    On validation issues, it tries to coerce safely; if impossible, returns empty list with an error.
    """
    if slots is None:
        return [], None
    if not isinstance(slots, list):
        return [], "input must be a list"
    norm: List[List[int]] = []
    for idx, it in enumerate(slots):
        if not (isinstance(it, (list, tuple)) and len(it) == 2):
            return [], f"input[{idx}] must be a pair [start, end]"
        s, e = it[0], it[1]
        if not (isinstance(s, int) and isinstance(e, int)):
            return [], f"input[{idx}] values must be integers"
        # Basic constraints from problem; if invalid, skip or clamp?
        # We'll skip invalid intervals but still return a solution entry.
        if not (0 <= s <= 4096 and 0 <= e <= 4096 and s < e):
            # Skip invalid interval but keep processing others
            continue
        norm.append([s, e])
    # Sort here to ensure determinism even if empty or partially invalid
    norm.sort(key=lambda x: (x[0], x[1]))
    return norm, None

def merge_slots(slots: List[List[int]]) -> List[List[int]]:
    if not slots:
        return []
    merged: List[List[int]] = []
    cur_start, cur_end = slots[0]
    for s, e in slots[1:]:
        # Merge overlapping or touching intervals
        if s <= cur_end:
            if e > cur_end:
                cur_end = e
        else:
            merged.append([cur_start, cur_end])
            cur_start, cur_end = s, e
    merged.append([cur_start, cur_end])
    return merged

def min_boats_needed(slots: List[List[int]]) -> int:
    if not slots:
        return 0
    starts = sorted(s for s, _ in slots)
    ends = sorted(e for _, e in slots)
    i = j = 0
    boats = max_boats = 0
    n = len(slots)
    # Use strict inequality so [a,b] and [b,c] are not overlapping (end is exclusive)
    while i < n and j < n:
        if starts[i] < ends[j]:
            boats += 1
            if boats > max_boats:
                max_boats = boats
            i += 1
        else:
            boats -= 1
            j += 1
    return max_boats

def solve_one(tc: Dict[str, Any]) -> Dict[str, Any]:
    tc_id = tc.get("id")
    # Always include id; if missing, put a safe fallback string so response is valid JSON
    if tc_id is None:
        tc_id = ""

    slots_raw = tc.get("input")
    slots, _err = normalize_slots(slots_raw)
    merged = merge_slots(slots)
    boats = min_boats_needed(slots)

    # Always include sortedMergedSlots; include minBoatsNeeded to aim for full score
    return {
        "id": tc_id,
        "sortedMergedSlots": merged,
        "minBoatsNeeded": boats,
    }

@app.post("/sailing-club/submission")
async def submission(request: Request):
    # Always respond JSON to avoid HTML errors at clients
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON in request body"})
    test_cases = body.get("testCases")
    if test_cases is None:
        # Still return the required envelope
        return JSONResponse(content={"solutions": []})
    if not isinstance(test_cases, list):
        # Coerce to empty if malformed
        test_cases = []

    solutions = [solve_one(tc if isinstance(tc, dict) else {}) for tc in test_cases]
    return JSONResponse(content={"solutions": solutions})

if __name__ == "__main__":
    # Run locally: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)

    
# if __name__ == '__main__':
#     #app.run(host='0.0.0.0', port=5000)
#     port = int(os.environ.get("PORT", 5000)) # Get the PORT env var, default to 5000 for local run
#     app.run(host='0.0.0.0', port=port) # You MUST set host to '0.0.0.0'