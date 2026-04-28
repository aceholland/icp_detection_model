# Hugging Face + Vercel Deployment Strategy

This guide explains how to host the NeuraScan backend on Hugging Face Spaces and the frontend on Vercel.

## 1. Backend (Hugging Face Spaces)
1. Go to [Hugging Face Spaces](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK.
3. Upload all project files.
4. **Important**: Add a Secret named `GEMINI_API_KEY` in the Space settings.
5. Hugging Face will build the image using the provided `Dockerfile` and expose it on port 7860.
6. Your backend URL will look like: `https://your-user-name-space-name.hf.space`

## 2. Frontend (Vercel)
1. In your local `index.html`, find the line:
   ```javascript
   const API = window.location.protocol === 'file:' ? 'http://127.0.0.1:8000' : window.location.origin;
   ```
2. Replace it with your Hugging Face URL:
   ```javascript
   const API = 'https://your-user-name-space-name.hf.space';
   ```
3. Deploy the project folder to Vercel (you can use the Vercel dashboard or CLI).

## 3. Why this works?
- **FastAPI CORS**: The backend is configured to allow requests from any origin (`allow_origins=["*"]`), so the Vercel frontend can securely send data to the Hugging Face backend.
- **Compute**: Hugging Face provides the CPU/GPU power needed for the signal processing stages, while Vercel handles the static delivery of the UI.
