# Face Recognition API

ArcFace 512-d based face recognition system using Upstash Vector for storage.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 face_api.py
```

## VENV ON SERVER 
/www/wwwroot/dslr-selfiesearch-python/af6e46fea22026e09b2518cefff15de3_venv/bin/python3 -m pip install -r /www/wwwroot/dslr-selfiesearch-python/requirements.txt

## Configuration

Create a `.env` file:
```
auth_key=your_secret_auth_key_here
port=8080
debug=true

# Upstash Vector
upstash_url=your_upstash_url
upstash_token=your_upstash_token

# Face Detection Settings
similarity_threshold=0.70
det_thresh=0.5
det_quality_min=0.3
det_size=480
```

| Key | Description | Default |
|-----|-------------|---------|
| `auth_key` | Bearer token for API authentication | required |
| `port` | Server port | 8080 |
| `debug` | Enable daily log files | false |
| `upstash_url` | Upstash Vector index URL | required |
| `upstash_token` | Upstash Vector token | required |
| `similarity_threshold` | Min cosine similarity for search/match | 0.70 |
| `det_thresh` | Min confidence to detect a face | 0.5 |
| `det_quality_min` | Min det_score to save a face | 0.3 |
| `det_size` | Detection resolution (affects CPU, not embedding quality) | 480 |

## Authentication

All endpoints except `/`, `/test`, and `/detect-face` require Bearer token:

```
Authorization: Bearer your_secret_auth_key_here
```

## API Endpoints

### GET /detect-face
Visualize face detection on an image. Returns the image with rectangles drawn:
- 🟢 Green = face will be saved (det_score ≥ det_quality_min)
- 🔴 Red = face will be skipped (det_score < det_quality_min)

```
GET /detect-face?url=https://example.com/image.jpg
```

No authentication required. Useful for tuning detection settings.

### POST /submit
Store all face embeddings from image (requires auth):
```json
{
  "id": "unique_id",
  "image_url": "https://example.com/image.jpg",
  "album_id": "album1",
  "date_deletion": "2024-12-31"
}
```

Response:
```json
{
  "status": "done",
  "id": "unique_id",
  "total_faces": 5
}
```

Status values:
- `done` - Successfully processed or ID already exists
- `noface` - No face detected in image
- `error` - Processing error occurred

All faces from a group photo are batch-uploaded to Upstash in a single API call with sub-IDs (e.g., `unique_id_face1`, `unique_id_face2`).

### POST /search
Search similar faces using all faces in query image (requires auth):
```json
{
  "image_url": "https://example.com/query.jpg",
  "album_id": "album1",
  "date_deletion": "2024-12-31"
}
```

Response:
```json
{
  "matches": ["person1", "person2"],
  "details": [
    {
      "image_url": "https://example.com/image1.jpg",
      "similarity": 0.85,
      "id": "person1_face2",
      "original_id": "person1"
    }
  ]
}
```

### POST /match
Match selfie face with all faces in photo (requires auth):
```json
{
  "photo": "https://example.com/group-photo.jpg",
  "selfie": "https://example.com/selfie.jpg"
}
```

Response:
```json
{
  "match": true,
  "similarity": 0.75,
  "threshold": 0.70,
  "photo_faces": 10,
  "selfie_faces": 1
}
```

### POST /delete-album
Delete all vectors for a specific album (requires auth):
```json
{
  "album_id": "album1",
  "date_deletion": "2024-12-31"
}
```

### POST /delete-file
Delete specific file by ID (requires auth):
```json
{
  "album_id": "album1",
  "id": "unique_id",
  "date_deletion": "2024-12-31"
}
```

### GET /clean
Delete expired vectors from Upstash and log files older than 3 days (requires auth).

Response:
```json
{
  "status": "success",
  "deleted_vectors": 120,
  "deleted_logs": ["2024-01-12.log"]
}
```

### GET /status
Server status.

### POST /status-album
List all files in an album (requires auth):
```json
{
  "album_id": "album1",
  "date_deletion": "2024-12-31"
}
```

Response:
```json
{
  "album_id": "album1",
  "date_deletion": "2024-12-31",
  "total_files": 2,
  "files": [
    {"id": "123", "filepath": "https://example.com/image1.jpg"},
    {"id": "456", "filepath": "https://example.com/image2.jpg"}
  ]
}
```

### GET /test
Health check endpoint.

## Logging

When `debug=true`, daily log files are created:
```
logs/
├── 2024-01-15.log
├── 2024-01-16.log
```

Log files use Asia/Kolkata timezone and are cleaned up after 3 days via `/clean`.
