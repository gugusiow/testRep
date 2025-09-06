from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/payload_crackme', methods=['GET'])
def get_payload():
    return "111-1111111"

# @app.route('/chasetheflag', methods=['POST'])
# def receive_data():
#     dict = {
#       "challenge1": "your_flag_1",
#       "challenge2": "your_flag_2",
#       "challenge3": "your_flag_3",
#       "challenge4": "your_flag_4",
#       "challenge5": "your_flag_5"
#     }
#     return jsonify(dict), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)      