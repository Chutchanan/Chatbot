#!/bin/bash
# Activate virtual environment (if needed)
# source venv/bin/activate

# Run the FastAPI app
uvicorn api_app:app --host 0.0.0.0 --port 80
