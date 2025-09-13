# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the backend for a HackMIT 2025 research paper comparison tool. The system aims to help researchers compare academic papers by automatically building citation graphs, extracting metrics, and enabling quantitative comparisons between papers.

## Architecture

The backend is a FastAPI application that uses:
- **Ray** for distributed computing and parallel processing
- **Datadog** for monitoring, tracing, and profiling
- **FastAPI** as the web framework with CORS middleware
- **Pydantic** for data validation

The system is designed to handle the following workflow:
1. Frontend agent determines search queries for seed papers
2. Backend spins up web crawlers to explore citation networks
3. Massively parallel processing with domain expert agents per paper
4. Metric extraction and vectorization across all papers
5. Graph construction with citation edges and benchmark comparisons

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync
```

### Running the Application
```bash
# Set up Datadog monitoring (optional)
./setup_datadog.sh
source datadog_env.sh

# Run the development server
python entrypoint.py
# OR
uvicorn entrypoint:app --host 0.0.0.0 --port 8000 --reload
```

### Key Environment Variables
- `DD_ENV`: Datadog environment (default: "development")
- `DD_AGENT_HOST`: Datadog agent host (default: "localhost")
- `RAY_ADDRESS`: Ray cluster address (default: "auto")
- `DEMO_SLOW_PROCESSING`: Enable slow processing demo (default: "false")

## Key Endpoints

- `GET /`: Health check
- `POST /items/`: Create and process items with Ray
- `GET /health`: Comprehensive health check for Ray and Datadog
- `GET /debug/profile`: Trigger profiling sample for performance testing
- `GET /debug/config`: View current configuration

## Technology Stack

- **Python 3.12+**: Core language
- **FastAPI**: Web framework with automatic OpenAPI documentation
- **Ray**: Distributed computing for parallel paper processing
- **Datadog**: Comprehensive monitoring with tracing and profiling
- **Pydantic**: Data validation and serialization
- **uvicorn**: ASGI server

## Development Notes

- The main application code is in `entrypoint.py`
- Ray is initialized with namespace "backend"
- All HTTP requests are automatically traced with Datadog
- Health checks verify both Ray cluster connectivity and Datadog integration
- The application includes middleware for request timing and metrics collection