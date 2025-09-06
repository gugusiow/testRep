from flask import Flask, request, jsonify
import os
import math

app = Flask(__name__)

#@app.route('/trivia', methods=['GET'])
# def home():
#     return "Welcome to the Flask app!"

# @app.route('/data', methods=['POST'])
# def receive_data():
#     data = request.get_json()
#     return jsonify({"received": data}), 201

# def get_trivia():
#     # Answers to the trivia questions
#     result = {"answers": [2, 1, 2, 2, 3, 4, 3, 5, 4]}
    
#     return jsonify(result)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_latency_points(distance):
    points = max(0, 30 - distance)
    return round(points)

@app.route('/ticketing-agent', methods=['POST'])
def ticketing_agent():
    try:
        data = request.get_json()
        
        customers = data.get('customers', [])
        concerts = data.get('concerts', [])
        priority = data.get('priority', {})
        
        recommendations = {}
        
        for customer in customers:
            customer_name = customer['name']
            vip_status = customer['vip_status']
            customer_location = (customer['location'][0], customer['location'][1])
            credit_card = customer['credit_card']
            
            max_points = -1
            best_concert = None
            
            for concert in concerts:
                concert_name = concert['name']
                booking_center = (concert['booking_center_location'][0], 
                                 concert['booking_center_location'][1])
                
                points = 0 # Calculate points for this concert
                
                if vip_status:
                    points += 100
                
                if credit_card in priority and priority[credit_card] == concert_name:
                    points += 50
                
                distance = calculate_distance(customer_location, booking_center)
                latency_points = calculate_latency_points(distance)
                points += latency_points
                

                if points > max_points:
                    max_points = points
                    best_concert = concert_name
                
            recommendations[customer_name] = best_concert
        
        return jsonify(recommendations)
    
    except Exception as e:
       return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000)
    port = int(os.environ.get("PORT", 5000)) # Get the PORT env var, default to 5000 for local run
    app.run(host='0.0.0.0', port=port) # You MUST set host to '0.0.0.0'