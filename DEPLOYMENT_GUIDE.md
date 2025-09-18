# Vercel Deployment Guide for Pneumonia Detection API

## Overview

This guide will help you deploy your PyTorch-based pneumonia detection Flask API to Vercel.

## What I've Set Up

### 1. Updated Requirements

- Fixed `requirements.txt` to include only Flask dependencies (removed FastAPI)
- Added specific versions for compatibility

### 2. Created Vercel-Compatible Structure

- Created `api/index.py` as the main entry point
- Updated `vercel.json` configuration
- Added proper CORS headers
- Fixed model loading paths for Vercel deployment

### 3. Key Changes Made

- **Entry Point**: Moved from `app.py` to `api/index.py` (Vercel standard)
- **Device**: Changed to CPU-only (Vercel doesn't provide GPU)
- **Model Paths**: Updated to use absolute paths for Vercel
- **Error Handling**: Improved error handling and logging
- **Endpoints**: Added health check and root endpoints

## Deployment Steps

### 1. Install Vercel CLI (if not already installed)

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Deploy to Vercel

From your project root directory:

```bash
vercel
```

Follow the prompts:

- Set up and deploy? **Yes**
- Which scope? Choose your account
- Link to existing project? **No** (for first deployment)
- Project name: `pneumonia-detection-api` (or your preferred name)
- Directory: **./** (current directory)

### 4. Production Deployment

```bash
vercel --prod
```

## API Endpoints

After deployment, your API will be available at:

- **Base URL**: `https://your-project-name.vercel.app`
- **Process Image**: `POST /api/process_image`
- **Health Check**: `GET /api/health`
- **Root**: `GET /`

## Testing Your API

### Using curl:

```bash
# Health check
curl https://your-project-name.vercel.app/api/health

# Process image
curl -X POST \
  -F "image=@path/to/your/image.jpg" \
  -F "model=1" \
  https://your-project-name.vercel.app/api/process_image
```

### Using Python requests:

```python
import requests

# Health check
response = requests.get('https://your-project-name.vercel.app/api/health')
print(response.json())

# Process image
with open('path/to/image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'model': '1'}
    response = requests.post(
        'https://your-project-name.vercel.app/api/process_image',
        files=files,
        data=data
    )
    print(response.json())
```

## Important Notes

### Model Files

- Your model files (`best_cnn_model.pth`, `best_model.pth`) are included in the deployment
- The API loads both models but currently uses the same model for both endpoints
- If you want to use different models, update the model loading in `api/index.py`

### Performance Considerations

- Vercel has a 10-second timeout for serverless functions
- Model loading happens on cold starts, which may take a few seconds
- Consider using Vercel Pro for longer timeouts if needed

### Environment Variables

If you need environment variables, create a `.env.local` file:

```
# Add any environment variables here
```

Then update `vercel.json` to include:

```json
{
  "env": {
    "YOUR_VARIABLE": "@your-variable"
  }
}
```

## Troubleshooting

### Common Issues:

1. **Model not found**: Ensure model files are in the `model/` directory
2. **Timeout errors**: Consider optimizing model size or using Vercel Pro
3. **Memory issues**: PyTorch models can be memory-intensive; monitor usage

### Logs:

Check deployment logs in Vercel dashboard or use:

```bash
vercel logs
```

## File Structure After Setup

```
your-project/
├── api/
│   ├── __init__.py
│   └── index.py          # Main Flask app
├── model/
│   ├── best_cnn_model.pth
│   └── best_model.pth
├── requirements.txt      # Updated dependencies
├── vercel.json          # Vercel configuration
└── app.py              # Original file (kept for reference)
```

## Next Steps

1. Deploy using the steps above
2. Test your API endpoints
3. Integrate with your frontend application
4. Monitor performance and usage in Vercel dashboard

Your pneumonia detection API is now ready for deployment on Vercel! 🚀
