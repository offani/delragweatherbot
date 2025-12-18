# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install curl for downloading Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first
COPY pyproject.toml uv.lock .python-version* ./

# Install dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start Ollama in background, wait for it, pull model, and run app
CMD ["/bin/bash", "-c", "ollama serve & \
    echo 'Waiting for Ollama...' && \
    while ! curl -s http://localhost:11434 >/dev/null; do sleep 1; done && \
    echo 'Ollama ready!' && \
    ollama pull nomic-embed-text && \
    echo 'Starting app...' && \
    uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
