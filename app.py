from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/detectExercises', methods=['POST'])
def detectExercises():
    videoURL = request.json.get('videoURL')
    response_data = [{'exercise': "exmaple"}]
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')