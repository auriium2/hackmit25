# HackMIT 2025 Backend

A FastAPI + Ray application for distributed computing.

## Setup

1. Make sure you have Python 3.12+ installed

2. Set up the virtual environment:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .  # Installs from pyproject.toml
   # Or you can install dependencies explicitly:
   # pip install fastapi ray uvicorn pydantic asyncio
   ```

## Datadog Setup

1. Install the Datadog Agent:
   - Follow the [official installation instructions](https://docs.datadoghq.com/agent/basic_agent_usage/) for your operating system
   - For macOS: `brew install --cask datadog-agent`
   - For Linux: Follow distribution-specific instructions from Datadog's site
   - For Windows: Download and run the installer from Datadog's site

2. Use the provided setup script to configure Datadog environment variables:
   ```bash
   cd backend
   ./setup_datadog.sh
   source datadog_env.sh
   ```

   Or configure environment variables manually:
   ```bash
   # Required variables
   export DD_ENV=development  # or 'production', 'staging', etc.
   export DD_SERVICE=hackmit-backend
   export DD_VERSION=0.1.0
   
   # Optional variables
   export DD_AGENT_HOST=localhost  # Change if your agent is running elsewhere
   export DD_TRACE_AGENT_PORT=8126  # Default port
   export DD_PROFILING_ENABLED=true
   export DD_TRACE_SAMPLE_RATE=1.0  # 1.0 = 100% of requests
   export DD_LOGS_INJECTION=true
   export DD_RUNTIME_METRICS_ENABLED=true
   
   # For demonstrating slow processing in examples
   export DEMO_SLOW_PROCESSING=false  # Set to 'true' to simulate slow processing
   ```

3. Verify the Datadog Agent is running:
   - macOS/Linux: `sudo datadog-agent status`
   - Windows: Check the Datadog Agent Manager in the system tray

## Running the Application

Start the application with:

```bash
source datadog_env.sh  # Load Datadog environment variables
python entrypoint.py
```

This will start the FastAPI server at http://localhost:8000.

## Testing the Application with Demo Client

The project includes a demo client script that can generate various types of traffic to demonstrate the monitoring and profiling capabilities:

```bash
# Run basic checks to verify API functionality
./demo_client.py --mode check

# Generate mixed load to see diverse trace patterns
./demo_client.py --mode mixed --requests 200 --concurrency 10

# Generate spike load to test performance under heavy traffic
./demo_client.py --mode spike --spike-requests 500 --spike-concurrency 30

# Run all test modes in sequence
./demo_client.py --mode all
```

## API Endpoints

- `GET /`: Check if the API is running
- `GET /health`: Health check endpoint to verify Ray cluster and Datadog connection
- `POST /items/`: Create a new item and process it with Ray
- `GET /items/{item_id}`: Get an item by ID (placeholder implementation)
- `GET /debug/profile`: Trigger a CPU-intensive operation to demonstrate profiling
- `GET /debug/config`: View Datadog and Ray configuration information

## Ray Cluster

By default, the application connects to a local Ray cluster. To connect to a remote Ray cluster, set the `RAY_ADDRESS` environment variable:

```bash
export RAY_ADDRESS="ray://<host>:<port>"
python entrypoint.py
```

## Development

To enable hot-reloading during development, the application is configured to run with `uvicorn` and `reload=True` when executed directly.

## Monitoring and Profiling

### Datadog Dashboard

Once the application is running with Datadog properly configured:

1. Open the [Datadog APM Dashboard](https://app.datadoghq.com/apm/home)
2. Select the service `hackmit-backend` to view trace data
3. Look for the following information:
   - Request latency and throughput
   - Error rates
   - Service dependencies
   - Resource usage (CPU, memory)

### Understanding the Datadog Integration

Our application uses several Datadog features:

1. **Automatic Instrumentation**: The `patch_all()` function instruments libraries automatically
2. **Custom Middleware**: Tracks HTTP request timing and adds custom tags
3. **Ray Worker Tracing**: Traces Ray distributed tasks with parent-child relationship
4. **Continuous Profiling**: Collects CPU, memory, and I/O profiles at regular intervals

### Profiling Data

To view profiling data:

1. Go to the [Datadog Profiling Dashboard](https://app.datadoghq.com/profiling)
2. Select the service `hackmit-backend`
3. Analyze CPU, memory, and I/O profiles
4. Use the demo client to generate load and see the resulting profiles:
   ```bash
   ./demo_client.py --mode spike
   ```

### Creating Custom Dashboards

To create a custom dashboard in Datadog:

1. Go to Dashboards → New Dashboard
2. Add widgets for:
   - Request rate by endpoint
   - Error rate
   - Request latency (p95, p99)
   - Ray task execution time
   - CPU and memory profiles

### Performance Tuning

Based on profiling data, you can optimize your application:

1. Look for hotspots in the CPU profile
2. Identify memory leaks in memory profiles
3. Check for I/O-bound operations that could be optimized
4. Analyze distributed task performance in Ray

### Tracing Request Flow

The application is configured to trace:
- HTTP requests through FastAPI (middleware)
- Ray task execution (remote function wrappers)
- Custom business logic (explicit trace spans)
- Background processing (async tasks)

All traces are tagged with relevant metadata for easier filtering and analysis.