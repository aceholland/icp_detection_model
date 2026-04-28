# NeuraScan Deployment Guide

This document outlines how to deploy the NeuraScan ICP Detection Model using Docker.

## 1. Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop/) installed on your machine.
- A Gemini API Key (already set in your `.env` file).

## 2. Local Deployment with Docker Compose (Recommended)
The easiest way to run the application is using Docker Compose, as it automatically handles volumes (for saving results) and environment variables.

### Steps:
1. Open your terminal in the project directory.
2. Run the following command:
   ```bash
   docker compose up --build
   ```
3. Once the build is finished and the server starts, open your browser to:
   **[http://localhost:8000](http://localhost:8000)**

### Why use Compose?
- **Persistence**: It maps the `artifacts/` and `models/` folders to your host machine, so your scan data isn't lost if the container stops.
- **Simplicity**: You don't need to manually pass environment variables every time.

---

## 3. Manual Docker Commands
If you prefer using raw Docker commands:

### Build the image:
```bash
docker build -t neurascan .
```

### Run the container:
```bash
docker run -p 8000:8000 --env-file .env neurascan
```

---

## 4. Cloud Deployment
To host this application online, follow these general steps:

### A. Managed Platforms (Render, Railway, Fly.io)
1. Push your code to a GitHub repository.
2. Connect the repository to the platform.
3. The platform will detect the `Dockerfile` and build it automatically.
4. **Important**: Add your `GEMINI_API_KEY` as an "Environment Variable" in the platform's dashboard.

### B. Cloud Providers (AWS, Google Cloud, Azure)
1. Use **Google Cloud Run** or **AWS App Runner**.
2. These services are designed for Docker containers. You simply point them to your container image, and they provide a public URL with HTTPS automatically.

## 5. Troubleshooting
- **Camera Access**: Ensure you are accessing the site over `localhost` or `https`. Most browsers block camera access on non-secure `http` connections unless it is `localhost`.
- **Model Files**: If the app fails to start, ensure the `models/` directory contains the `face_landmarker.task` and other `.pkl` files.
