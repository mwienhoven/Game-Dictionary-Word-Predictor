# Optimized Dockerfile for python 3.11
FROM python:3.11-slim

# Set working directory to app
WORKDIR /app

# Install system dependencies
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal build tools for compiling packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install curl
RUN apt-get update && apt-get install -y curl

# Install uv
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

# Make uv accessible
ENV PATH="/root/.local/bin:${PATH}"

# Copy the dependency files and project structure
COPY pyproject.toml uv.lock ./
COPY README.md ./README.md
COPY src/slanggen ./src/slanggen
COPY backend ./backend
COPY frontend ./frontend
COPY artefacts ./artefacts

# Install the project (slanggen) from pyproject.toml via uv pip (system install, no-cache)
# Install backend requirements via uv pip (system install, no-cache)
# Install Torch CPU via uv pip from the PyTorch index
RUN /root/.local/bin/uv pip install --system --no-cache . \
    && /root/.local/bin/uv pip install --system --no-cache -r backend/requirements.txt \
    && /root/.local/bin/uv pip install --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch==2.9.1

# Remove uv to reduce image size
RUN rm -rf /root/.local/bin/uv /root/.local/lib/python3.11/site-packages/uv*

# Expose port 8000 for the application
EXPOSE 8000

# Set working directory to app root so relative paths work
WORKDIR /app

# CMD to run the backend application with uvicorn
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]