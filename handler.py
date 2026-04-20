import runpod
from face_api import app

def handler(event):
    """RunPod serverless handler - routes jobs to Flask endpoints"""
    try:
        input_data = event.get('input', {})
        endpoint = input_data.get('endpoint', '')
        payload = input_data.get('payload', {})
        auth_key = input_data.get('auth_key', '')

        with app.test_client() as client:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {auth_key}'
            }

            if endpoint == 'submit':
                res = client.post('/submit', json=payload, headers=headers)
            elif endpoint == 'search':
                res = client.post('/search', json=payload, headers=headers)
            elif endpoint == 'match':
                res = client.post('/match', json=payload, headers=headers)
            elif endpoint == 'delete-file':
                res = client.post('/delete-file', json=payload, headers=headers)
            elif endpoint == 'delete-album':
                res = client.post('/delete-album', json=payload, headers=headers)
            elif endpoint == 'clean':
                res = client.get('/clean', headers=headers)
            elif endpoint == 'status-album':
                res = client.post('/status-album', json=payload, headers=headers)
            elif endpoint == 'status':
                res = client.get('/status')
            elif endpoint == 'detect-face':
                url = payload.get('url', '')
                res = client.get(f'/detect-face?url={url}')
            else:
                return {'error': f'Unknown endpoint: {endpoint}'}

            return res.get_json()

    except Exception as e:
        return {'error': str(e)}

runpod.serverless.start({'handler': handler})
