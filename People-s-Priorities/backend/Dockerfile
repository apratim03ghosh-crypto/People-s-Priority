FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# Copy frontend so FastAPI can serve index.html from ../frontend
COPY frontend/ ../frontend/

EXPOSE 8080

# Render sets $PORT at runtime; main.py reads os.getenv("PORT") automatically
CMD ["python", "main.py"]
