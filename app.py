from flask import Flask, request, jsonify
import cv2
import numpy as np
import insightface
import os

app = Flask(__name__)

# -----------------------------
# Load InsightFace
# -----------------------------
face_app = insightface.app.FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0, det_size=(320, 320))

# -----------------------------
# Load registered face
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_FILE = os.path.join(BASE_DIR, "dinesh_face.npy")

registered_face = np.load(FACE_FILE).astype(np.float32)
registered_face = registered_face / np.linalg.norm(registered_face)


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "Face Recognition API"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


@app.route("/recognize", methods=["POST"])
def recognize():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "Image missing"
        }), 400

    file = request.files["image"]

    image_bytes = file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({
            "success": False,
            "message": "Invalid image"
        }), 400

    faces = face_app.get(frame)

    if len(faces) == 0:
        return jsonify({
            "success": True,
            "match": False,
            "message": "No face detected"
        })

    # Largest face
    face = max(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) *
                      (x.bbox[3] - x.bbox[1])
    )

    embedding = face.embedding.astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)

    similarity = float(np.dot(registered_face, embedding))

    # Threshold
    threshold = 0.45

    matched = similarity >= threshold

    return jsonify({
        "success": True,
        "match": matched,
        "name": "DINESH" if matched else "UNKNOWN",
        "similarity": round(similarity, 4)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)