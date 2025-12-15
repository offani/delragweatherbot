# LangGraph Weather & RAG Agent

A simple, elegant AI agent pipeline built with LangGraph, LangChain, and Streamlit. This agent can fetch real-time weather data and answer questions from your PDF documents.

## Features

- **Agentic Workflow**: Uses LangGraph to intelligently route queries between Weather and RAG tools.
- **RAG Capability**: Ingests PDFs, creates embeddings (using HuggingFace), and retrieves relevant answers using Qdrant.
- **Real-time Weather**: Fetches live weather data from OpenWeatherMap.
- **Visualization**: Streamlit UI shows the internal thought process (nodes visited, data retrieved).
- **Evaluation**: Includes LangSmith evaluation script.

## Setup

1.  **Prerequisites**:
    - Python 3.10+
    - [uv](https://github.com/astral-sh/uv) (Package Manager)

2.  **Installation**:
    ```bash
    uv sync
    ```

3.  **Configuration**:
    - Rename `.env` (if not already set) or create one.
    - Add your API keys:
      ```env
      GROQ_API_KEY=your_groq_key
      OPENWEATHERMAP_API_KEY=your_weather_key
      LANGCHAIN_API_KEY=your_langsmith_key
      ```

## Usage

### Run the UI
Start the Streamlit application:
```bash
uv run streamlit run app.py
```

1.  **Weather**: Type "Weather in [City]" (e.g., "Weather in Mumbai").
2.  **RAG**: Upload a PDF in the sidebar and ask questions about it.
3.  **Visuals**: Expand "Processing Details" to see the intermediate steps.

### Run Tests
Execute unit tests:
```bash
uv run pytest
```

### Run Evaluation
Run LangSmith evaluation (requires configured dataset):
```bash
uv run python eval.py
```

## Structure
- `src/graph.py`: Main LangGraph workflow definition.
- `src/nodes.py`: Implementation of graph nodes (Router, Weather, RAG).
- `src/weather.py`: OpenWeatherMap API wrapper.
- `src/rag.py`: RAG system with Qdrant and HuggingFace embeddings.
- `app.py`: Streamlit frontend.
- `eval.py`: Evaluation script.
