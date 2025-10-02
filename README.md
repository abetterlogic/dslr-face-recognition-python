# Face Recognition API

Minimal ArcFace 512-d based face recognition system optimized for 2 CPU 8GB RAM.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 face_api.py
```

Server runs on configurable port (default: 8080)

## Configuration

Create a `.env` file with the following structure:
```
auth_key=your_secret_auth_key_here
port=8080
debug=true
```

- `auth_key` - Bearer token for API authentication
- `port` - Server port (default: 8080)
- `debug` - Enable/disable debug mode (true/false)

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
  "date_deletion": "2024-12-31"
}
```

Response:
```json
{
  "status": "done",
  "id": "unique_id",
  "total_faces": 1
}
```

Status values:
- `done` - Successfully processed or ID already exists
- `noface` - No face detected in image
- `error` - Processing error occurred

`total_faces` indicates number of faces detected in the submitted image.

### POST /search
Search similar faces (requires auth):
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
      "id": "person1"
    }
  ]
}
```

### POST /delete-album
Delete specific album (requires auth):
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
  "album_id": "1bb85b1c-84ac-4a3b-9811-35d798bc3bb0",
  "id": "123",
  "date_deletion": "2025-11-27"
}
```

### GET /clean
Delete expired folders (requires auth).

### GET /status
Data folder statistics:

Response:
```json
{
  "total_disk_space_mb": 15.67,
  "total_disk_space_bytes": 16435200,
  "last_updated": "2024-01-15T14:30:45.123456",
  "data_folder_exists": true,
  "is_debug": true,
  "active": true
}
```

### POST /status-album
Album file details (requires auth):
```json
{
  "album_id": "1bb85b1c-84ac-4a3b-9811-35d798bc3bb0",
  "date_deletion": "2025-11-27"
}
```

Response:
```json
{
  "album_id": "1bb85b1c-84ac-4a3b-9811-35d798bc3bb0",
  "date_deletion": "2025-11-27",
  "total_files": 2,
  "files": [
    {
      "id": "123",
      "filepath": "https://example.com/image1.jpg"
    },
    {
      "id": "456",
      "filepath": "https://example.com/image2.jpg"
    }
  ]
}
```

### GET /test
Health check endpoint.

## File Structure
```
data/
├── {date_deletion}/
│   ├── {album_id}.pkl
│   └── {album_id2}.pkl
```

Each .pkl file contains all face embeddings for that album. The `data/` folder is git-ignored.