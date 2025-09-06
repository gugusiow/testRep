from flask import Flask, request, jsonify
import os
import math
from collections import defaultdict

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
    result = {"answers": [3, 1, 2, 2, 3, 4, 4, 5, 4, 3, 3, 3, 2, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3, 5, 1]}
    
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

# @app.route('/sailing-club', methods=['POST'])
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

#             result = {'solutions': merged_result}
#             return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

    
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000)) # Get the PORT env var, default to 5000 for local run
    app.run(host='0.0.0.0', port=port) # You MUST set host to '0.0.0.0'