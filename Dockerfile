# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory inside the container
WORKDIR /app

# Install required system dependencies for Python and Chrome
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        gnupg \
        fonts-liberation \
        libappindicator3-1 \
        libnss3 \
        libatk-bridge2.0-0 \
        libgbm1 \
        libx11-xcb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxrandr2 \
        libgtk-3-0 \
        libasound2 \
        xdg-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/google-chrome-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-chrome-stable && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies and Playwright
RUN pip install --upgrade pip && \
    pip install playwright && \
    playwright install && \
    playwright install-deps

# Copy application code and requirements
COPY . .

# Clone the LightRAG repository and install it
RUN rm -rf LightRAG && git clone https://github.com/HKUDS/LightRAG.git

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
