
# Start from a lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the files needed for installation
COPY pyproject.toml .
COPY README.md .

# Install the dependencies (no need for cached packages)
RUN pip install --no-cache-dir .

# Copy the source code
COPY src/ src/
COPY outputs/model outputs/model
COPY configs/ configs/

# The FastAPI application listens on port 8000
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]