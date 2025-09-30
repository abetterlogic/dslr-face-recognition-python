# Face Recognition API

Minimal OpenCV-based face recognition system optimized for 2 CPU 8GB RAM.

## Setup

```bash
pip install -r requirements.txt
python3 face_api.py
```

Server runs on `http://localhost:8080`

## API Endpoints

### GET /
Server status and available endpoints.

### POST /submit
Store face embedding:
```json
{
  "id": "unique_id",
  "image_url": "https://example.com/image.jpg",
  "album_id": "album1",
  "expiry_date": "2024-12-31"
}
```

### POST /search
Search similar faces:
```json
{
  "image_url": "https://example.com/query.jpg",
  "album_id": "album1",
  "expiry_date": "2024-12-31"
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
      "id": "person1"
    }
  ]
}
```

### GET /test
Health check endpoint.

## File Structure
```
data/
├── {expiry_date}/
│   └── {album_id}/
│       └── {id}.pkl
```

Each .pkl file contains face embedding and metadata. The `data/` folder is git-ignored.