from flask import Flask, request, jsonify
from flask_cors import CORS

from main import *

app = Flask(__name__)
CORS(app)


def startup():
    loadPoseEstimator()


with app.app_context():
    startup()


# @app.route("/detectExercises", methods=["POST"])
@app.route("/")
def detectExercise():
    poses = extractPoses(
        "https://firebasestorage.googleapis.com/v0/b/yrproject-64b5e.appspot.com/o/dhQCVLTSVZMNzDwNVDa0pumDhhm2%2Fposts%2F0.4126l9pqhdp?alt=media&token=f0d1e235-d2ce-41ca-b80e-8e683361be19"
    )
    print(poses)
    createLoader(poses)
    return jsonify(poses.tolist())


if __name__ == "__main__":
    app.run(host="0.0.0.0")
