# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

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

# Run the app
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
