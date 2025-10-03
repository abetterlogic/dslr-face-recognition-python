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

# Load ArcFace model
face_model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0, det_size=(640, 640))

def download_image(url):
    """Download image from URL"""
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(response.content)
        return tmp.name

def get_face_embedding(image_path):
    """Extract ArcFace 512d embedding"""
    img = cv2.imread(image_path)
    faces = face_model.get(img)
    
    if faces:
        # Use first detected face (highest confidence)
        face = faces[0]
        embedding = face.embedding  # 512-dimensional ArcFace embedding
        return embedding, len(faces)
    return None, 0

def save_embedding(embedding, image_url, file_id, date_deletion, album_id):
    """Save embedding to album file"""
    folder_path = os.path.join(DATA_DIR, date_deletion)
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"{album_id}.pkl")
    
    # Load existing album data or create new
    album_data = []
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            album_data = pickle.load(f)
    
    # Add new face data
    face_data = {
        'id': file_id,
        'image_url': image_url,
        'embedding': embedding.tolist(),
        'album_id': album_id,
        'date_deletion': date_deletion
    }
    
    # Remove existing entry with same id if exists
    album_data = [item for item in album_data if item['id'] != file_id]
    album_data.append(face_data)
    
    # Save updated album
    with open(file_path, 'wb') as f:
        pickle.dump(album_data, f)
    
    return len(album_data)

def search_similar_faces(query_embedding, album_id, date_deletion):
    """Search for similar faces in album file"""
    results = []
    file_path = os.path.join(DATA_DIR, date_deletion, f"{album_id}.pkl")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                album_data = pickle.load(f)
            
            for face_data in album_data:
                stored_embedding = np.array(face_data['embedding'])
                # Calculate similarity using cosine similarity
                dot_product = np.dot(query_embedding, stored_embedding)
                norm_a = np.linalg.norm(query_embedding)
                norm_b = np.linalg.norm(stored_embedding)
                similarity = dot_product / (norm_a * norm_b)
                
                if similarity > 0.4:  # ArcFace threshold
                    results.append({
                        'image_url': face_data['image_url'],
                        'similarity': float(similarity),
                        'id': face_data['id']
                    })
        except:
            pass
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

@app.route('/submit', methods=['POST'])
@require_auth
def submit():
    """Submit face for storage"""
    try:
        data = request.json
        image_url = data['image_url']
        file_id = data['id']
        album_id = data['album_id']
        date_deletion = data['date_deletion']
        
        # Check if ID already exists
        folder_path = os.path.join(DATA_DIR, date_deletion)
        file_path = os.path.join(folder_path, f"{album_id}.pkl")
        
        current_total = 0
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                album_data = pickle.load(f)
            current_total = len(album_data)
            
            for item in album_data:
                if item['id'] == file_id:
                    return jsonify({'status': 'done', 'id': file_id, 'total_faces': 1})
        
        # Download and process image
        img_path = download_image(image_url)
        embedding, face_count = get_face_embedding(img_path)
        os.unlink(img_path)
        
        if embedding is None:
            return jsonify({'status': 'noface', 'id': file_id, 'total_faces': face_count})
        
        # Save embedding
        save_embedding(embedding, image_url, file_id, date_deletion, album_id)
        
        return jsonify({'status': 'done', 'id': file_id, 'total_faces': face_count})
    
    except Exception as e:
        return jsonify({'status': 'error', 'id': data.get('id', ''), 'total_faces': 0, 'error': str(e)}), 500

@app.route('/search', methods=['POST'])
@require_auth
def search():
    """Search for similar faces"""
    try:
        data = request.json
        image_url = data['image_url']
        album_id = data['album_id']
        date_deletion = data['date_deletion']
        
        # Download and process query image
        img_path = download_image(image_url)
        query_embedding, _ = get_face_embedding(img_path)
        os.unlink(img_path)
        
        if query_embedding is None:
            return jsonify({'matches': [], 'details': [], 'error': 'No face detected'})
        
        # Search for matches
        results = search_similar_faces(query_embedding, album_id, date_deletion)
        
        return jsonify({
            'matches': [r['id'] for r in results],
            'details': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Face Recognition API Server is running',
        'status': 'active',
        'endpoints': {
            'POST /submit': 'Store face embedding',
            'POST /search': 'Search similar faces',
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
            
            # Find and remove the specific ID
            original_count = len(album_data)
            album_data = [item for item in album_data if item['id'] != file_id]
            
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
        
        # Test if face model is working
        active = True
        try:
            # Quick test of face model functionality
            face_model.prepare(ctx_id=0, det_size=(640, 640))
        except:
            active = False
        
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