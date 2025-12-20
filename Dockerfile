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

# Pre-load HuggingFace models to prevent runtime timeouts
RUN uv run python preload_models.py

# Expose port (default 8501, but Render will override)
EXPOSE 8501

# Run the app binding to Render's $PORT
# "sh -c" is required for variable expansion
CMD ["sh", "-c", "uv run streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
