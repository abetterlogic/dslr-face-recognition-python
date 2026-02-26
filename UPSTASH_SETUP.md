# Upstash Vector Setup Guide

## Step 1: Get Upstash Credentials

1. Go to your Upstash Vector dashboard
2. Copy your **Endpoint URL** (looks like: `https://xxxxx.upstash.io`)
3. Copy your **Token**

## Step 2: Update .env File

Add these lines to your `.env` file:

```
upstash_url=https://your-endpoint.upstash.io
upstash_token=your_token_here
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Restart Server

```bash
python3 face_api.py
```

## How It Works

### Automatic Fallback System

The system now uses **Upstash Vector** as primary storage with **SQLite as fallback**:

- ✅ If Upstash credentials are configured → Uses Upstash Vector (cloud)
- ✅ If Upstash fails or not configured → Falls back to SQLite (local)
- ✅ No code changes needed to switch between them

### Benefits of Upstash

1. **Cloud Storage** - Data accessible from anywhere
2. **Fast Vector Search** - Native cosine similarity indexing
3. **Scalable** - Handles millions of embeddings
4. **Cost-Effective** - ~$4/month for 1M embeddings
5. **No Infrastructure** - Fully managed service

### Testing

1. **Test with Upstash:**
   - Configure credentials in `.env`
   - Submit a face via `/submit`
   - Search via `/search`
   - Check Upstash dashboard for stored vectors

2. **Test Fallback:**
   - Remove Upstash credentials from `.env`
   - System automatically uses SQLite
   - All APIs work the same way

## Monitoring

Check logs for:
- `Using Upstash Vector` - Upstash is active
- `Upstash error, falling back to SQLite` - Fallback triggered
- `Upstash search error` - Search fallback

## Cost Estimate

For **1 million embeddings + 5000 searches/month**:
- Storage: $4/month
- Queries: $0.01/month
- **Total: ~$4/month**

## Migration

### From SQLite to Upstash

If you have existing SQLite data and want to migrate:

1. Keep both systems running
2. New data goes to Upstash automatically
3. Old data remains in SQLite (fallback)
4. Gradually migrate old data if needed

### From Upstash to SQLite

Simply remove Upstash credentials from `.env` - system falls back to SQLite immediately.
