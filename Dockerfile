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
RUN sh /install.sh && rm /install.sh

# Make uv accessible
ENV PATH="/root/.local/bin:${PATH}"

# Copy the dependency files and project structure
COPY pyproject.toml uv.lock ./
COPY README.md ./README.md
COPY src/slanggen ./src/slanggen
COPY backend ./backend
COPY frontend ./frontend
COPY artefacts ./artefacts

# Install python dependencies without Torch (use pytorch index for torch exclusion)
RUN uv sync --frozen --no-dev --no-editable

# Install Torch from CPU PyTorch index (as specified in pyproject.toml)
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# Install backend requirements
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose port 8000 for the application
EXPOSE 8000

# Set working directory to app root so relative paths work
WORKDIR /app

# CMD to run the backend application with uvicorn
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
