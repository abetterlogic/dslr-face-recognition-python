import os
import time
import pickle
import numpy as np
import cv2
import requests
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from face_api import face_model, face_model_lock, log_to_file, DET_QUALITY_MIN, SIMILARITY_THRESHOLD

load_dotenv()

LARAVEL_URL = os.getenv('laravel_url').rstrip('/')
FACE_AUTH = os.getenv('face_auth')
POLL_INTERVAL = int(os.getenv('poll_interval', 10))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

HEADERS = {
    'X-Face-Auth': FACE_AUTH,
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

def download_image(url):
    response = requests.get(url, timeout=15)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(response.content)
        return tmp.name

def get_pkl_path(date_deletion, album_id):
    # Normalize date_deletion to date only (strip time if present)
    date_str = date_deletion.split(' ')[0]
    folder = os.path.join(DATA_DIR, date_str)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{album_id}.pkl")

def load_embeddings(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(pkl_path, data):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def process_embedding(photo):
    photo_id = photo['photo_id']
    album_id = photo['album_id']
    date_deletion = photo['date_deletion']
    url = photo['url']

    try:
        img_path = download_image(url)
        img = cv2.imread(img_path)
        with face_model_lock:
            faces = face_model.get(img)
        os.unlink(img_path)

        if not faces:
            log_to_file(f"NO FACES - photo_id: {photo_id}")
            return {'photo_id': photo_id, 'embedded_status': 'no_face', 'total_faces': 0}

        # Filter quality faces and build embeddings list
        face_embeddings = []
        for face in faces:
            if face.det_score < DET_QUALITY_MIN:
                continue
            embedding = face.embedding / np.linalg.norm(face.embedding)
            face_embeddings.append(embedding)

        if not face_embeddings:
            log_to_file(f"NO QUALITY FACES - photo_id: {photo_id}")
            return {'photo_id': photo_id, 'embedded_status': 'no_face', 'total_faces': 0}

        # Save to pkl
        pkl_path = get_pkl_path(date_deletion, album_id)
        data = load_embeddings(pkl_path)
        data[photo_id] = face_embeddings
        save_embeddings(pkl_path, data)

        log_to_file(f"EMBEDDED - photo_id: {photo_id}, faces: {len(face_embeddings)}, album: {album_id}")
        return {'photo_id': photo_id, 'embedded_status': 'done', 'total_faces': len(face_embeddings)}

    except Exception as e:
        log_to_file(f"FAILED - photo_id: {photo_id}, error: {str(e)}")
        return {'photo_id': photo_id, 'embedded_status': 'failed', 'total_faces': 0}

def process_matching(job):
    request_id = job['request_id']
    album_id = job['album_id']
    date_deletion = job['date_deletion']
    selfie_url = job['selfie_url']

    try:
        # Download and get selfie embedding
        selfie_path = download_image(selfie_url)
        selfie_img = cv2.imread(selfie_path)
        with face_model_lock:
            selfie_faces = face_model.get(selfie_img)
        os.unlink(selfie_path)

        if not selfie_faces:
            log_to_file(f"NO SELFIE FACE - request_id: {request_id}")
            return {'request_id': request_id, 'photo_ids': ''}

        selfie_embedding = selfie_faces[0].embedding / np.linalg.norm(selfie_faces[0].embedding)

        # Load pkl directly using date_deletion and album_id
        pkl_path = get_pkl_path(date_deletion, album_id)
        if not os.path.exists(pkl_path):
            log_to_file(f"NO PKL FOUND - request_id: {request_id}, album: {album_id}, path: {pkl_path}")
            return None  # Skip — don't mark as scanned so it retries later

        data = load_embeddings(pkl_path)
        matched_ids = []
        for photo_id, face_embeddings in data.items():
            best_cosine = max(
                float(np.dot(emb, selfie_embedding) / (np.linalg.norm(emb) * np.linalg.norm(selfie_embedding)))
                for emb in face_embeddings
            )
            score = (1 + best_cosine) / 2
            if score >= SIMILARITY_THRESHOLD:
                matched_ids.append(photo_id)

        log_to_file(f"MATCHED - request_id: {request_id}, album: {album_id}, matches: {len(matched_ids)}")
        return {'request_id': request_id, 'photo_ids': matched_ids}

    except Exception as e:
        log_to_file(f"MATCH FAILED - request_id: {request_id}, error: {str(e)}")
        return {'request_id': request_id, 'photo_ids': ''}

def run_embedding_loop():
    try:
        res = requests.get(f"{LARAVEL_URL}/api/facerecognition/pending-face-embedding", headers=HEADERS, timeout=15)
        if res.status_code != 200:
            log_to_file(f"Embedding fetch failed: {res.status_code}")
            return

        photos = res.json().get('data', [])
        if not photos:
            return

        log_to_file(f"Embedding batch: {len(photos)} photos")
        results = [process_embedding(photo) for photo in photos]

        post_res = requests.post(
            f"{LARAVEL_URL}/api/facerecognition/pending-face-embedding",
            headers=HEADERS,
            json={'results': results},
            timeout=15
        )
        log_to_file(f"Embedding submitted: status={post_res.status_code} body={post_res.text[:200]}")

    except Exception as e:
        log_to_file(f"Embedding loop error: {str(e)}")

def run_matching_loop():
    try:
        res = requests.get(f"{LARAVEL_URL}/api/facerecognition/pending-face-matching", headers=HEADERS, timeout=30)
        if res.status_code != 200:
            log_to_file(f"Matching fetch failed: {res.status_code}")
            return

        jobs = res.json().get('data', [])
        if not jobs:
            return

        log_to_file(f"Matching batch: {len(jobs)} requests")
        results = [r for r in (process_matching(job) for job in jobs) if r is not None]

        if not results:
            log_to_file("Matching: no results to submit (all skipped)")
            return

        post_res = requests.post(
            f"{LARAVEL_URL}/api/facerecognition/pending-face-matching",
            headers=HEADERS,
            json={'results': results},
            timeout=15
        )
        log_to_file(f"Matching submitted: status={post_res.status_code} body={post_res.text[:200]}")

    except Exception as e:
        log_to_file(f"Matching loop error: {str(e)}")

if __name__ == '__main__':
    print(f"Worker started. Poll interval: {POLL_INTERVAL}s")
    log_to_file("Worker started")

    while True:
        run_embedding_loop()
        run_matching_loop()
        time.sleep(POLL_INTERVAL)
