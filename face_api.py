import os
import pickle
import numpy as np
from datetime import datetime, date, timedelta
from flask import Flask, request, jsonify
import requests
import cv2
from PIL import Image
import tempfile
import shutil
from functools import wraps
from dotenv import load_dotenv
import insightface
import pytz

load_dotenv()
app = Flask(__name__)

@app.before_request
def log_request_info():
    if request.method in ['POST', 'PUT', 'PATCH']:
        log_to_file(f"Request: {request.method} {request.path} - Payload: {request.get_json()}")

# Load configuration from environment
AUTH_KEY = os.getenv('auth_key')
PORT = int(os.getenv('port', 8080))
DEBUG = os.getenv('debug', 'false').lower() == 'true'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

def log_to_file(message):
    """Log message to daily log file"""
    if DEBUG:
        os.makedirs(LOG_DIR, exist_ok=True)
        kolkata_tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(kolkata_tz)
        log_file = os.path.join(LOG_DIR, f"{now.strftime('%Y-%m-%d')}.log")
        with open(log_file, 'a') as f:
            f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S %Z')} - {message}\n")

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Unauthenticated'}), 401
        
        token = auth_header.split(' ')[1]
        if token != AUTH_KEY:
            return jsonify({'error': 'Unauthenticated'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

# Load ArcFace model with thread safety
import threading
face_model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)  # Very low detection threshold for challenging images
face_model_lock = threading.Lock()
file_locks = {}  # Dictionary to store locks per album file
file_locks_lock = threading.Lock()  # Lock for the locks dictionary

def download_image(url):
    """Download image from URL"""
    response = requests.get(url, timeout=15)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(response.content)
        return tmp.name

def get_face_embedding(image_path):
    """Extract ArcFace 512d embedding with quality filtering"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None, 0, 0.0
    
    with face_model_lock:
        faces = face_model.get(img)
    
    if faces:
        # Use best quality face (highest detection score)
        best_face = max(faces, key=lambda f: f.det_score)
        print(f"Detected {len(faces)} faces, best score: {best_face.det_score}")
        # Accept any detected face (model already filtered with det_thresh=0.1)
        embedding = best_face.embedding / np.linalg.norm(best_face.embedding)  # Normalize
        return embedding, len(faces), float(best_face.det_score)
    else:
        print(f"No faces detected in image: {image_path}")
    return None, 0, 0.0

def save_embedding(embedding, image_url, file_id, date_deletion, album_id, quality_score=1.0):
    """Save embedding to album file with quality metadata"""
    folder_path = os.path.join(DATA_DIR, date_deletion)
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"{album_id}.pkl")
    
    # Get or create lock for this specific file
    with file_locks_lock:
        if file_path not in file_locks:
            file_locks[file_path] = threading.Lock()
        file_lock = file_locks[file_path]
    
    # Use file-specific lock to prevent concurrent writes
    with file_lock:
        # Load existing album data or create new
        album_data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    album_data = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Corrupted pickle file {file_path}, recreating: {e}")
                log_to_file(f"Corrupted pickle file {file_path}, recreating: {e}")
                album_data = []
        
        # Add new face data with quality score
        face_data = {
            'id': file_id,
            'image_url': image_url,
            'embedding': embedding / np.linalg.norm(embedding),  # Normalize
            'quality_score': quality_score,
            'album_id': album_id,
            'date_deletion': date_deletion
        }
        
        # Remove existing entry with same id if exists
        album_data = [item for item in album_data if item['id'] != file_id]
        album_data.append(face_data)
        
        # Save updated album atomically
        temp_path = file_path + '.tmp'
        with open(temp_path, 'wb') as f:
            pickle.dump(album_data, f)
        os.replace(temp_path, file_path)  # Atomic rename
    
    return len(album_data)

def search_similar_faces(query_embedding, album_id, date_deletion, query_quality=1.0):
    """Search for similar faces with adaptive threshold"""
    results = []
    file_path = os.path.join(DATA_DIR, date_deletion, f"{album_id}.pkl")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                album_data = pickle.load(f)
            
            # Normalize query embedding
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            
            for face_data in album_data:
                stored_embedding = face_data['embedding']
                stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
                
                # Optimized cosine similarity
                similarity = float(np.dot(query_norm, stored_norm))
                
                # Lower threshold for better recall
                threshold = 0.25
                
                if similarity > threshold:
                    results.append({
                        'image_url': face_data['image_url'],
                        'similarity': similarity,
                        'id': face_data['id'],
                        'quality': face_data.get('quality_score', 0.5)
                    })
        except Exception as e:
            print(f"Search error: {e}")
    
    # Return all matches sorted by similarity (no limit)
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

@app.route('/submit', methods=['POST'])
@require_auth
def submit():
    """Submit all faces in image for storage"""
    data = None
    try:
        data = request.json
        image_url = data['image_url']
        file_id = data['id']
        album_id = data['album_id']
        date_deletion = data['date_deletion']
        
        # Log at start of processing
        print(f"Processing /submit - image_url: {image_url}, id: {file_id}")
        log_to_file(f"Processing /submit - image_url: {image_url}, id: {file_id}")
        
        # Check if ID already exists
        folder_path = os.path.join(DATA_DIR, date_deletion)
        file_path = os.path.join(folder_path, f"{album_id}.pkl")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    album_data = pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Corrupted pickle file during check: {e}")
                album_data = []
            
            # Check if any face with this ID exists
            existing_faces = [item for item in album_data if item['id'].startswith(file_id)]
            if existing_faces:
                return jsonify({'status': 'done', 'id': file_id, 'total_faces': len(existing_faces)})
        
        # Download and process image - get all faces
        img_path = download_image(image_url)
        img = cv2.imread(img_path)
        with face_model_lock:
            faces = face_model.get(img) if face_model else []
        os.unlink(img_path)
        
        if not faces:
            return jsonify({'status': 'noface', 'id': file_id, 'total_faces': 0})
        
        # Save all faces with sub-IDs and quality scores
        saved_count = 0
        for i, face in enumerate(faces):
            # Accept all detected faces (model already filtered with det_thresh=0.1)
            face_id = f"{file_id}_face{i+1}" if len(faces) > 1 else file_id
            embedding = face.embedding
            save_embedding(embedding, image_url, face_id, date_deletion, album_id, float(face.det_score))
            saved_count += 1
        
        return jsonify({'status': 'done', 'id': file_id, 'total_faces': saved_count})
    
    except Exception as e:
        error_url = data.get('image_url', 'N/A') if data else 'N/A'
        error_id = data.get('id', 'N/A') if data else 'N/A'
        error_msg = f"ERROR /submit - image_url: {error_url}, id: {error_id}, error: {str(e)}"
        log_to_file(error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'id': error_id, 'image_url': error_url, 'total_faces': 0, 'error': str(e)}), 500

@app.route('/search', methods=['POST'])
@require_auth
def search():
    """Search for similar faces using all faces in query image"""
    try:
        data = request.json
        image_url = data['image_url']
        album_id = data['album_id']
        date_deletion = data['date_deletion']
        
        # Download and process query image - get all faces
        img_path = download_image(image_url)
        img = cv2.imread(img_path)
        with face_model_lock:
            faces = face_model.get(img) if face_model else []
        os.unlink(img_path)
        
        if not faces:
            return jsonify({'matches': [], 'details': [], 'error': 'No face detected'})
        
        # Search using all faces in query image
        all_results = {}
        for face in faces:
            # Accept all detected faces (model already filtered with det_thresh=0.1)
            query_embedding = face.embedding
            results = search_similar_faces(query_embedding, album_id, date_deletion, float(face.det_score))
            
            # Merge results, keeping best similarity for each ID
            for result in results:
                face_id = result['id']
                # Extract original photo ID (remove _face suffix)
                photo_id = face_id.split('_face')[0]
                result['original_id'] = photo_id
                
                if photo_id not in all_results or result['similarity'] > all_results[photo_id]['similarity']:
                    all_results[photo_id] = result
        
        # Convert to list and sort by similarity
        final_results = sorted(all_results.values(), key=lambda x: x['similarity'], reverse=True)
        
        # Debug info
        file_path = os.path.join(DATA_DIR, date_deletion, f"{album_id}.pkl")
        stored_faces_count = 0
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                album_data = pickle.load(f)
            stored_faces_count = len(album_data)
        
        return jsonify({
            'matches': [r['original_id'] for r in final_results],
            'details': final_results
        })
    
    except Exception as e:
        log_to_file(f"ERROR /search - image_url: {data.get('image_url', 'N/A')}, album_id: {data.get('album_id', 'N/A')}, error: {str(e)}")
        return jsonify({'error': str(e), 'image_url': data.get('image_url', '')}), 500

@app.route('/match', methods=['POST'])
@require_auth
def match():
    """Match selfie face with all faces in photo"""
    try:
        data = request.json
        photo_url = data['photo']
        selfie_url = data['selfie']
        
        print(f"Processing /match - selfie: {selfie_url}, photo: {photo_url}")
        
        # Download and process selfie
        selfie_path = download_image(selfie_url)
        print(f"Processing selfie: {selfie_url}")
        selfie_embedding, selfie_faces, selfie_quality = get_face_embedding(selfie_path)
        os.unlink(selfie_path)
        
        if selfie_embedding is None:
            return jsonify({'match': False, 'error': 'No face detected in selfie', 'selfie_url': selfie_url})
        
        # Download and process photo - get all faces
        photo_path = download_image(photo_url)
        print(f"Processing photo: {photo_url}")
        photo_img = cv2.imread(photo_path)
        if photo_img is None:
            os.unlink(photo_path)
            return jsonify({'match': False, 'error': 'Failed to read photo', 'photo_url': photo_url})
        
        with face_model_lock:
            faces = face_model.get(photo_img)
        os.unlink(photo_path)
        
        print(f"Photo faces detected: {len(faces) if faces else 0}")
        if not faces:
            return jsonify({'match': False, 'error': 'No face detected in photo', 'photo_url': photo_url})
        
        # Normalize selfie embedding
        selfie_norm = selfie_embedding / np.linalg.norm(selfie_embedding)
        
        # Compare selfie against all faces in photo
        best_similarity = -1
        for face in faces:
            # Accept all detected faces (model already filtered with det_thresh=0.1)
            photo_embedding = face.embedding / np.linalg.norm(face.embedding)
            similarity = float(np.dot(photo_embedding, selfie_norm))
            
            if similarity > best_similarity:
                best_similarity = similarity
        
        # Lower threshold for better recall
        threshold = 0.25
        is_match = bool(best_similarity > threshold)
        
        print(f"Match result: {is_match}, similarity: {best_similarity}, threshold: {threshold}")
        
        return jsonify({
            'match': is_match,
            'similarity': best_similarity,
            'threshold': threshold,
            'photo_faces': len(faces),
            'selfie_faces': selfie_faces
        })
    
    except Exception as e:
        log_to_file(f"ERROR /match - photo: {data.get('photo', 'N/A')}, selfie: {data.get('selfie', 'N/A')}, error: {str(e)}")
        print(f"ERROR /match: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'match': False, 'error': str(e), 'photo': data.get('photo', ''), 'selfie': data.get('selfie', '')}), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Face Recognition API Server is running',
        'status': 'active',
        'endpoints': {
            'POST /submit': 'Store face embedding',
            'POST /search': 'Search similar faces',
            'POST /match': 'Match two faces',
            'POST /delete-album': 'Delete specific album',
            'POST /delete-file': 'Delete specific file by ID',
            'GET /clean': 'Delete expired folders',
            'GET /status': 'Data folder statistics',
            'POST /status-album': 'Album file details',
            'GET /test': 'Health check'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/delete-file', methods=['POST'])
@require_auth
def delete_file():
    """Delete specific file by ID from album"""
    try:
        data = request.json
        album_id = data['album_id']
        file_id = data['id']
        date_deletion = data['date_deletion']
        
        file_path = os.path.join(DATA_DIR, date_deletion, f"{album_id}.pkl")
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                album_data = pickle.load(f)
            
            # Find and remove all faces with this ID (including sub-faces)
            original_count = len(album_data)
            album_data = [item for item in album_data if not item['id'].startswith(file_id)]
            
            if len(album_data) < original_count:
                # Save updated album
                with open(file_path, 'wb') as f:
                    pickle.dump(album_data, f)
                
                return jsonify({
                    'status': 'success',
                    'message': f'File {file_id} deleted from album {album_id}',
                    'id': file_id,
                    'album_id': album_id,
                    'remaining_files': len(album_data)
                })
            else:
                return jsonify({
                    'status': 'not_found',
                    'message': 'File ID not found in album',
                    'id': file_id,
                    'album_id': album_id
                })
        else:
            return jsonify({
                'status': 'not_found',
                'message': 'Album not found',
                'album_id': album_id
            })
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/delete-album', methods=['POST'])
@require_auth
def delete_album():
    """Delete specific album file"""
    try:
        data = request.json
        album_id = data['album_id']
        date_deletion = data['date_deletion']
        
        file_path = os.path.join(DATA_DIR, date_deletion, f"{album_id}.pkl")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({
                'status': 'success',
                'message': f'Album {album_id} deleted',
                'album_id': album_id,
                'date_deletion': date_deletion
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': 'Album not found',
                'album_id': album_id,
                'date_deletion': date_deletion
            })
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/clean', methods=['GET'])
@require_auth
def clean():
    """Clean expired folders and old log files"""
    try:
        deleted_folders = []
        deleted_logs = []
        today = date.today()
        three_days_ago = today - timedelta(days=3)
        data_path = DATA_DIR
        
        # Clean expired data folders
        if os.path.exists(data_path):
            for folder in os.listdir(data_path):
                folder_path = os.path.join(data_path, folder)
                if os.path.isdir(folder_path):
                    try:
                        folder_date = datetime.strptime(folder, '%Y-%m-%d').date()
                        if folder_date < today:
                            shutil.rmtree(folder_path)
                            deleted_folders.append(folder)
                    except ValueError:
                        continue
        
        # Clean old log files
        if os.path.exists(LOG_DIR):
            for log_file in os.listdir(LOG_DIR):
                if log_file.endswith('.log'):
                    try:
                        log_date_str = log_file.replace('.log', '')
                        log_date = datetime.strptime(log_date_str, '%Y-%m-%d').date()
                        if log_date < three_days_ago:
                            log_path = os.path.join(LOG_DIR, log_file)
                            os.remove(log_path)
                            deleted_logs.append(log_file)
                    except ValueError:
                        continue
        
        return jsonify({
            'status': 'success',
            'deleted_folders': deleted_folders,
            'deleted_logs': deleted_logs,
            'count': len(deleted_folders) + len(deleted_logs)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status-album', methods=['POST'])
@require_auth
def status_album():
    """Get album file details"""
    try:
        data = request.json
        album_id = data['album_id']
        date_deletion = data['date_deletion']
        
        file_path = os.path.join(DATA_DIR, date_deletion, f"{album_id}.pkl")
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                album_data = pickle.load(f)
            
            # Extract id and filepath (image_url) for each entry
            files = [{
                'id': item['id'],
                'filepath': item['image_url']
            } for item in album_data]
            
            return jsonify({
                'album_id': album_id,
                'date_deletion': date_deletion,
                'total_files': len(files),
                'files': files
            })
        else:
            return jsonify({
                'status': 'not_found',
                'message': 'Album not found',
                'album_id': album_id,
                'date_deletion': date_deletion
            })
    
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Data folder statistics"""
    try:
        data_path = DATA_DIR
        total_size = 0
        last_modified = None
        
        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    file_mtime = os.path.getmtime(file_path)
                    if last_modified is None or file_mtime > last_modified:
                        last_modified = file_mtime
        
        # Convert bytes to MB
        total_size_mb = round(total_size / (1024 * 1024), 2)
        
        # Convert timestamp to readable format
        last_updated = datetime.fromtimestamp(last_modified).isoformat() if last_modified else None
        
        # Check if face model exists (don't reload it!)
        active = face_model is not None
        
        return jsonify({
            'total_disk_space_mb': total_size_mb,
            'total_disk_space_bytes': total_size,
            'last_updated': last_updated,
            'data_folder_exists': os.path.exists(data_path),
            'is_debug': DEBUG,
            'active': active
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """Test API endpoint"""
    return jsonify({
        'status': 'working',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)