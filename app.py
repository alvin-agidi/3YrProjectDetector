from flask import Flask, request, jsonify
from flask_cors import CORS

from poseEstimation import *

app = Flask(__name__)
CORS(app)

poseEstimator = None


def startup():
    global poseEstimator
    poseEstimator = loadPoseEstimator()


with app.app_context():
    startup()


@app.route("/detectExercises", methods=["POST"])
def detectExercises():
    videoURL = request.json.get("videoURL")
    # print(extractPoses(videoURL))
    response_data = [{"exercise": "exmaple"}]
    return jsonify(response_data)


@app.route("/")
def home():
    return "Wow this is a basic output!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
