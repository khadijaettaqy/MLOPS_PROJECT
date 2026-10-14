FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY data/ data/

# Set environment variables
ENV MODEL_PATH=/app/src/models/heart_model.pkl
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "src.api.model_app:app", "--host", "0.0.0.0", "--port", "8000"]
