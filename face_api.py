import os
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import requests
import cv2
from PIL import Image
import tempfile

app = Flask(__name__)

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
    """Save embedding to file system"""
    folder_path = os.path.join('data', expiry_date, album_id)
    os.makedirs(folder_path, exist_ok=True)
    
    data = {
        'id': file_id,
        'image_url': image_url,
        'embedding': embedding.tolist(),
        'album_id': album_id,
        'expiry_date': expiry_date
    }
    
    file_path = os.path.join(folder_path, f"{file_id}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def search_similar_faces(query_embedding, album_id, expiry_date):
    """Search for similar faces in specific expiry/album folder"""
    results = []
    folder_path = os.path.join('data', expiry_date, album_id)
    
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.pkl'):
                file_path = os.path.join(folder_path, file)
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    stored_embedding = np.array(data['embedding'])
                    # Calculate similarity using cosine similarity
                    dot_product = np.dot(query_embedding, stored_embedding)
                    norm_a = np.linalg.norm(query_embedding)
                    norm_b = np.linalg.norm(stored_embedding)
                    similarity = dot_product / (norm_a * norm_b)
                    
                    if similarity > 0.3:  # Lower threshold
                        results.append({
                            'image_url': data['image_url'],
                            'similarity': float(similarity),
                            'id': data['id']
                        })
                except:
                    continue
    
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

@app.route('/submit', methods=['POST'])
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
            'GET /test': 'Health check'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test', methods=['GET'])
def test():
    """Test API endpoint"""
    return jsonify({
        'status': 'working',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)