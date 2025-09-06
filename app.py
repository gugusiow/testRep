from flask import Flask, request, jsonify
import os
import math
from collections import defaultdict

app = Flask(__name__)

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

from typing import List, Tuple
import json

def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """Merge overlapping intervals and sort them"""
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    current_start, current_end = intervals[0]
    
    for interval in intervals[1:]:
        start, end = interval
        
        # If current interval overlaps with next interval, merge them
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            # No overlap, add current interval to result
            merged.append([current_start, current_end])
            current_start, current_end = start, end
    
    # Add the last interval
    merged.append([current_start, current_end])
    
    return merged

def min_boats_needed(intervals: List[List[int]]) -> int:
    """Find minimum number of boats needed using sweep-line algorithm"""
    if not intervals:
        return 0
    
    # Create events: (time, +1 for start, -1 for end)
    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    
    # Sort events by time, and for same time, process ends first
    events.sort(key=lambda x: (x[0], x[1]))
    
    max_boats = 0
    current_boats = 0
    
    for time, event_type in events:
        current_boats += event_type
        max_boats = max(max_boats, current_boats)
    
    return max_boats

def solve_sailing_club(test_cases):
    solutions = []
    
    for test_case in test_cases:
        intervals = test_case["input"]
        
        # Part 1: Merge intervals
        merged_slots = merge_intervals(intervals)
        
        # Part 2: Find minimum boats needed
        min_boats = min_boats_needed(intervals)
        
        solutions.append({
            "id": test_case["id"],
            "sortedMergedSlots": merged_slots,
            "minBoatsNeeded": min_boats
        })
    
    return {"solutions": solutions}

# Flask endpoint would look like this:

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/sailing-club/submission', methods=['POST'])
def sailing_club_endpoint():
    try:
        data = request.get_json()
        test_cases = data.get('testCases', [])
        result = solve_sailing_club(test_cases)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)


    
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000)) # Get the PORT env var, default to 5000 for local run
    app.run(host='0.0.0.0', port=port) # You MUST set host to '0.0.0.0'