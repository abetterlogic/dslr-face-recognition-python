# Face Recognition API

Minimal OpenCV-based face recognition system optimized for 2 CPU 8GB RAM.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 face_api.py
```

Server runs on `http://localhost:8080`

## Authentication

All API endpoints (except `/` and `/test`) require Bearer token authentication:

```
Authorization: Bearer your_secret_auth_key_here
```

Update the `auth_key` in `.env` file before starting the server.

## API Endpoints

### GET /
Server status and available endpoints.

### POST /submit
Store face embedding (requires auth):
```json
{
  "id": "unique_id",
  "image_url": "https://example.com/image.jpg",
  "album_id": "album1",
  "expiry_date": "2024-12-31"
}
```

### POST /search
Search similar faces (requires auth):
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

### GET /clean
Delete expired folders (requires auth).

### GET /test
Health check endpoint.

## File Structure
```
data/
├── {expiry_date}/
│   ├── {album_id}.pkl
│   └── {album_id2}.pkl
```

Each .pkl file contains all face embeddings for that album. The `data/` folder is git-ignored.