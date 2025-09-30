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

# Load auth key from environment
AUTH_KEY = os.getenv('auth_key')

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
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        # Create simple feature vector from resized face
        return face_resized.flatten().astype(np.float32)
    return None

def save_embedding(embedding, image_url, file_id, expiry_date, album_id):
    """Save embedding to album file"""
    folder_path = os.path.join('data', expiry_date)
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
        'expiry_date': expiry_date
    }
    
    # Remove existing entry with same id if exists
    album_data = [item for item in album_data if item['id'] != file_id]
    album_data.append(face_data)
    
    # Save updated album
    with open(file_path, 'wb') as f:
        pickle.dump(album_data, f)

def search_similar_faces(query_embedding, album_id, expiry_date):
    """Search for similar faces in album file"""
    results = []
    file_path = os.path.join('data', expiry_date, f"{album_id}.pkl")
    
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
                
                if similarity > 0.3:  # Lower threshold
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
        expiry_date = data['expiry_date']
        
        # Download and process image
        img_path = download_image(image_url)
        embedding = get_face_embedding(img_path)
        os.unlink(img_path)
        
        if embedding is None:
            return jsonify({'error': 'No face detected'}), 400
        
        # Save embedding
        save_embedding(embedding, image_url, file_id, expiry_date, album_id)
        
        return jsonify({'status': 'success', 'id': file_id})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
@require_auth
def search():
    """Search for similar faces"""
    try:
        data = request.json
        image_url = data['image_url']
        album_id = data['album_id']
        expiry_date = data['expiry_date']
        
        # Download and process query image
        img_path = download_image(image_url)
        query_embedding = get_face_embedding(img_path)
        os.unlink(img_path)
        
        if query_embedding is None:
            return jsonify({'error': 'No face detected'}), 400
        
        # Search for matches
        results = search_similar_faces(query_embedding, album_id, expiry_date)
        
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
    app.run(host='0.0.0.0', port=8080, debug=True)