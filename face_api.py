import os
import threading
import numpy as np
from datetime import datetime, date, timedelta
from flask import Flask, request, jsonify, send_file
import requests
import cv2
import tempfile
import shutil
from functools import wraps
from dotenv import load_dotenv
import insightface
import pytz
from upstash_vector import Index

load_dotenv()
app = Flask(__name__)

AUTH_KEY = os.getenv('auth_key')
PORT = int(os.getenv('port', 8080))
DEBUG = os.getenv('debug', 'false').lower() == 'true'
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

# Upstash Vector
upstash_index = Index(url=os.getenv('upstash_url'), token=os.getenv('upstash_token'))

SIMILARITY_THRESHOLD = float(os.getenv('similarity_threshold', 0.70))
DET_QUALITY_MIN = float(os.getenv('det_quality_min', 0.3))

# ArcFace model
face_model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0, det_size=(int(os.getenv('det_size', 480)),) * 2, det_thresh=float(os.getenv('det_thresh', 0.5)))
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
    response = requests.get(url, timeout=15)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(response.content)
        return tmp.name

@app.before_request
def log_request_info():
    if request.method in ['POST', 'PUT', 'PATCH']:
        log_to_file(f"Request: {request.method} {request.path} - Payload: {request.get_json()}")

@app.route('/submit', methods=['POST'])
@require_auth
def submit():
    data = None
    try:
        data = request.json
        image_url = data['image_url']
        file_id = data['id']
        album_id = data['album_id']
        date_deletion = data['date_deletion']

        log_to_file(f"Processing /submit - id: {file_id}, image_url: {image_url}")

        # Check if already exists
        existing = upstash_index.fetch([f"{album_id}:{date_deletion}:{file_id}"], include_metadata=False)
        if existing and existing[0] is not None:
            log_to_file(f"SKIPPED /submit - id: {file_id} already exists")
            return jsonify({'status': 'done', 'id': file_id, 'total_faces': 1})

        # Download and detect faces
        img_path = download_image(image_url)
        img = cv2.imread(img_path)
        with face_model_lock:
            faces = face_model.get(img)
        os.unlink(img_path)

        if not faces:
            log_to_file(f"NO FACES DETECTED /submit - id: {file_id}")
            return jsonify({'status': 'noface', 'id': file_id, 'total_faces': 0})

        # Build vectors for all quality faces
        vectors = []
        for i, face in enumerate(faces):
            if face.det_score < DET_QUALITY_MIN:
                continue
            face_id = f"{file_id}_face{i+1}" if len(faces) > 1 else file_id
            embedding = face.embedding / np.linalg.norm(face.embedding)
            vectors.append((
                f"{album_id}:{date_deletion}:{face_id}",
                embedding.tolist(),
                {'id': face_id, 'image_url': image_url, 'album_id': album_id, 'date_deletion': date_deletion}
            ))

        if not vectors:
            log_to_file(f"NO QUALITY FACES /submit - id: {file_id}, all {len(faces)} faces filtered")
            return jsonify({'status': 'noface', 'id': file_id, 'total_faces': 0})

        # Single batch upsert
        upstash_index.upsert(vectors=vectors)
        log_to_file(f"SUCCESS /submit - id: {file_id}, faces: {len(vectors)}, album: {album_id}")

        return jsonify({'status': 'done', 'id': file_id, 'total_faces': len(vectors)})

    except Exception as e:
        error_id = data.get('id', 'N/A') if data else 'N/A'
        log_to_file(f"ERROR /submit - id: {error_id}, error: {str(e)}")
        return jsonify({'status': 'error', 'id': error_id, 'total_faces': 0, 'error': str(e)}), 500

@app.route('/search', methods=['POST'])
@require_auth
def search():
    try:
        data = request.json
        image_url = data['image_url']
        album_id = data['album_id']
        date_deletion = data['date_deletion']

        img_path = download_image(image_url)
        img = cv2.imread(img_path)
        with face_model_lock:
            faces = face_model.get(img)
        os.unlink(img_path)

        if not faces:
            return jsonify({'matches': [], 'details': [], 'error': 'No face detected'})

        all_results = {}
        for face in faces:
            query_norm = face.embedding / np.linalg.norm(face.embedding)
            search_results = upstash_index.query(
                vector=query_norm.tolist(),
                top_k=100,
                include_metadata=True,
                filter=f"album_id = '{album_id}' AND date_deletion = '{date_deletion}'"
            )
            for match in search_results:
                if match.score < SIMILARITY_THRESHOLD:
                    continue
                meta = match.metadata
                face_id = meta.get('id', '')
                photo_id = face_id.split('_face')[0]
                if photo_id not in all_results or match.score > all_results[photo_id]['similarity']:
                    all_results[photo_id] = {
                        'image_url': meta.get('image_url'),
                        'similarity': float(match.score),
                        'id': face_id,
                        'original_id': photo_id
                    }

        final_results = sorted(all_results.values(), key=lambda x: x['similarity'], reverse=True)
        return jsonify({'matches': [r['original_id'] for r in final_results], 'details': final_results})

    except Exception as e:
        log_to_file(f"ERROR /search - error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/match', methods=['POST'])
@require_auth
def match():
    try:
        data = request.json
        photo_url = data['photo']
        selfie_url = data['selfie']

        log_to_file(f"/match - selfie: {selfie_url}, photo: {photo_url}")

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

        best_similarity = max(
            float(np.dot(face.embedding / np.linalg.norm(face.embedding), selfie_embedding))
            for face in photo_faces
        )

        threshold = SIMILARITY_THRESHOLD
        is_match = bool(best_similarity > threshold)
        log_to_file(f"/match result: {is_match}, similarity: {best_similarity:.4f}, photo_faces: {len(photo_faces)}, selfie_faces: {len(selfie_faces)}")

        return jsonify({
            'match': is_match,
            'similarity': best_similarity,
            'threshold': threshold,
            'photo_faces': len(photo_faces),
            'selfie_faces': len(selfie_faces)
        })

    except Exception as e:
        log_to_file(f"ERROR /match - error: {str(e)}")
        return jsonify({'match': False, 'error': str(e)}), 500

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

@app.route('/delete-file', methods=['POST'])
@require_auth
def delete_file():
    try:
        data = request.json
        album_id = data['album_id']
        file_id = data['id']
        date_deletion = data['date_deletion']

        # Delete base ID and up to 50 sub-faces
        ids = [f"{album_id}:{date_deletion}:{file_id}"] + \
              [f"{album_id}:{date_deletion}:{file_id}_face{i}" for i in range(1, 51)]
        upstash_index.delete(ids)

        return jsonify({'status': 'success', 'id': file_id, 'album_id': album_id})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/delete-album', methods=['POST'])
@require_auth
def delete_album():
    try:
        data = request.json
        album_id = data['album_id']
        date_deletion = data['date_deletion']

        # Delete all vectors matching this album via range delete
        deleted = 0
        cursor = None
        while True:
            result = upstash_index.range(cursor=cursor, limit=100, include_metadata=True)
            ids_to_delete = [
                v.id for v in result.vectors
                if v.metadata and v.metadata.get('album_id') == album_id
                and v.metadata.get('date_deletion') == date_deletion
            ]
            if ids_to_delete:
                upstash_index.delete(ids_to_delete)
                deleted += len(ids_to_delete)
            if not result.next_cursor:
                break
            cursor = result.next_cursor

        return jsonify({'status': 'success', 'album_id': album_id, 'deleted': deleted})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/clean', methods=['GET'])
@require_auth
def clean():
    try:
        deleted_logs = []
        today = date.today()
        three_days_ago = today - timedelta(days=3)

        # Delete expired vectors from Upstash
        deleted_vectors = 0
        cursor = None
        while True:
            result = upstash_index.range(cursor=cursor, limit=100, include_metadata=True)
            ids_to_delete = []
            for v in result.vectors:
                if v.metadata:
                    try:
                        exp = datetime.strptime(v.metadata.get('date_deletion', ''), '%Y-%m-%d').date()
                        if exp < today:
                            ids_to_delete.append(v.id)
                    except ValueError:
                        pass
            if ids_to_delete:
                upstash_index.delete(ids_to_delete)
                deleted_vectors += len(ids_to_delete)
            if not result.next_cursor:
                break
            cursor = result.next_cursor

        # Clean old log files
        if os.path.exists(LOG_DIR):
            for log_file in os.listdir(LOG_DIR):
                if log_file.endswith('.log'):
                    try:
                        log_date = datetime.strptime(log_file.replace('.log', ''), '%Y-%m-%d').date()
                        if log_date < three_days_ago:
                            os.remove(os.path.join(LOG_DIR, log_file))
                            deleted_logs.append(log_file)
                    except ValueError:
                        pass

        return jsonify({'status': 'success', 'deleted_vectors': deleted_vectors, 'deleted_logs': deleted_logs})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status-album', methods=['POST'])
@require_auth
def status_album():
    try:
        data = request.json
        album_id = data['album_id']
        date_deletion = data['date_deletion']

        files = {}
        cursor = None
        while True:
            result = upstash_index.range(cursor=cursor, limit=100, include_metadata=True)
            for v in result.vectors:
                if v.metadata and v.metadata.get('album_id') == album_id \
                        and v.metadata.get('date_deletion') == date_deletion:
                    fid = v.metadata.get('id', '')
                    photo_id = fid.split('_face')[0]
                    files[photo_id] = v.metadata.get('image_url', '')
            if not result.next_cursor:
                break
            cursor = result.next_cursor

        file_list = [{'id': k, 'filepath': v} for k, v in files.items()]
        return jsonify({'album_id': album_id, 'date_deletion': date_deletion,
                        'total_files': len(file_list), 'files': file_list})

    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'active': face_model is not None,
        'is_debug': DEBUG,
        'storage': 'upstash_vector',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Face Recognition API Server is running',
        'endpoints': ['/submit', '/search', '/match', '/delete-file', '/delete-album', '/clean', '/status', '/status-album', '/test']
    })

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'working', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print(f"Starting on port {PORT}, debug={DEBUG}")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
