# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory inside the container
WORKDIR /app

# Install required system dependencies for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code and requirements
COPY . .

# Clone the LightRAG repository and install it
# RUN git clone https://github.com/HKUDS/LightRAG.git
RUN rm -rf LightRAG && git clone https://github.com/HKUDS/LightRAG.git

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD [ "sleep", "infinity" ]