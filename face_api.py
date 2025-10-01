import os
import pickle
import numpy as np
from datetime import datetime, date
from flask import Flask, request, jsonify
import requests
import cv2
from PIL import Image
import tempfile
import shutil
from functools import wraps
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

@app.before_request
def log_request_info():
    if request.method in ['POST', 'PUT', 'PATCH']:
        print(f"Request: {request.method} {request.path} - Payload: {request.get_json()}")

# Load configuration from environment
AUTH_KEY = os.getenv('auth_key')
PORT = int(os.getenv('port', 8080))

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

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def download_image(url):
    """Download image from URL"""
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(response.content)
        return tmp.name

def get_face_embedding(image_path):
    """Extract simple face features using OpenCV"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
    
    if len(faces) > 0:
        # Use largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (100, 100))
        # Normalize pixel values
        face_normalized = face_resized / 255.0
        return face_normalized.flatten().astype(np.float32), len(faces)
    return None, len(faces)

def save_embedding(embedding, image_url, file_id, date_deletion, album_id):
    """Save embedding to album file"""
    folder_path = os.path.join('data', date_deletion)
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
    file_path = os.path.join('data', date_deletion, f"{album_id}.pkl")
    
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
                
                if similarity > 0.1:  # Very low threshold for basic features
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
        print(f"Submit request payload: {request.json}")
        data = request.json
        image_url = data['image_url']
        file_id = data['id']
        album_id = data['album_id']
        date_deletion = data['date_deletion']
        
        # Check if ID already exists
        folder_path = os.path.join('data', date_deletion)
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
        print(f"Search request payload: {request.json}")
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
            'GET /clean': 'Delete expired folders',
            'GET /test': 'Health check'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/clean', methods=['GET'])
@require_auth
def clean():
    """Clean expired folders"""
    try:
        deleted_folders = []
        today = date.today()
        data_path = 'data'
        
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
        
        return jsonify({
            'status': 'success',
            'deleted_folders': deleted_folders,
            'count': len(deleted_folders)
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
    app.run(host='0.0.0.0', port=PORT, debug=True)