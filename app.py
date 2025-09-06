from flask import Flask, request, jsonify

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
    result = {"answers": [2, 1, 2, 2, 3, 4, 3, 5, 4]}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)