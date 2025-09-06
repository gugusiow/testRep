from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Flask app!"

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    return jsonify({"received": data}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)