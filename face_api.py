import os
import threading
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import requests as http_requests
import cv2
import tempfile
from functools import wraps
from dotenv import load_dotenv
import insightface
import pytz

load_dotenv()
app = Flask(__name__)

AUTH_KEY = os.getenv('auth_key')
PORT = int(os.getenv('port', 8080))
DEBUG = os.getenv('debug', 'false').lower() == 'true'
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

SIMILARITY_THRESHOLD = float(os.getenv('similarity_threshold', 0.70))
DET_QUALITY_MIN = float(os.getenv('det_quality_min', 0.3))

# ArcFace model - shared across Flask and worker
USE_GPU = os.getenv('use_gpu', 'false').lower() == 'true'
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if USE_GPU else ['CPUExecutionProvider']
face_model = insightface.app.FaceAnalysis(providers=providers)
face_model.prepare(ctx_id=0 if USE_GPU else -1, det_size=(int(os.getenv('det_size', 480)),) * 2, det_thresh=float(os.getenv('det_thresh', 0.5)))
face_model_lock = threading.Lock()

def log_to_file(message):
    if DEBUG:
        os.makedirs(LOG_DIR, exist_ok=True)
        kolkata_tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(kolkata_tz)
        with open(os.path.join(LOG_DIR, f"{now.strftime('%Y-%m-%d')}.log"), 'a') as f:
            f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')} - {message}\n")

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer ') or auth_header.split(' ')[1] != AUTH_KEY:
            return jsonify({'error': 'Unauthenticated'}), 401
        return f(*args, **kwargs)
    return decorated_function

def download_image(url):
    response = http_requests.get(url, timeout=15)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(response.content)
        return tmp.name

@app.route('/detect-face', methods=['GET'])
def detect_face():
    try:
        image_url = request.args.get('url')
        if not image_url:
            return jsonify({'error': 'url parameter required'}), 400

        img_path = download_image(image_url)
        img = cv2.imread(img_path)
        with face_model_lock:
            faces = face_model.get(img)
        os.unlink(img_path)

        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            color = (0, 255, 0) if face.det_score >= DET_QUALITY_MIN else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{face.det_score:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(out.name, img)
        return send_file(out.name, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/match', methods=['POST'])
@require_auth
def match():
    try:
        data = request.json
        photo_url = data['photo']
        selfie_url = data['selfie']

        selfie_path = download_image(selfie_url)
        selfie_img = cv2.imread(selfie_path)
        with face_model_lock:
            selfie_faces = face_model.get(selfie_img)
        os.unlink(selfie_path)

        if not selfie_faces:
            return jsonify({'match': False, 'error': 'No face detected in selfie'})

        selfie_embedding = selfie_faces[0].embedding / np.linalg.norm(selfie_faces[0].embedding)

        photo_path = download_image(photo_url)
        photo_img = cv2.imread(photo_path)
        with face_model_lock:
            photo_faces = face_model.get(photo_img)
        os.unlink(photo_path)

        if not photo_faces:
            return jsonify({'match': False, 'error': 'No face detected in photo'})

        best_cosine = max(
            float(np.dot(face.embedding / np.linalg.norm(face.embedding), selfie_embedding) /
                  (np.linalg.norm(face.embedding / np.linalg.norm(face.embedding)) * np.linalg.norm(selfie_embedding)))
            for face in photo_faces
        )
        best_similarity = (1 + best_cosine) / 2

        is_match = bool(best_similarity > SIMILARITY_THRESHOLD)
        log_to_file(f"/match result: {is_match}, similarity: {best_similarity:.4f}, photo_faces: {len(photo_faces)}, selfie_faces: {len(selfie_faces)}")

        return jsonify({
            'match': is_match,
            'similarity': best_similarity,
            'threshold': SIMILARITY_THRESHOLD,
            'photo_faces': len(photo_faces),
            'selfie_faces': len(selfie_faces)
        })

    except Exception as e:
        log_to_file(f"ERROR /match - error: {str(e)}")
        return jsonify({'match': False, 'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'active': face_model is not None,
        'is_debug': DEBUG,
        'use_gpu': USE_GPU,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'working', 'timestamp': datetime.now().isoformat()})

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Face Recognition API Server is running',
        'endpoints': ['/detect-face', '/match', '/status', '/test']
    })

if __name__ == '__main__':
    print(f"Starting on port {PORT}, debug={DEBUG}, gpu={USE_GPU}")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
