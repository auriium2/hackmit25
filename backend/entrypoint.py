import os
import ray
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Dict, Any, Optional
import time

# Import Datadog libraries for monitoring and profiling
from ddtrace import patch_all, tracer
from ddtrace.profiling import Profiler

# Initialize Datadog tracing and profiling
patch_all()  # Patch all supported libraries for automatic instrumentation
profiler = Profiler(
    service="hackmit-backend",
    env=os.environ.get("DD_ENV", "development"),
    version=os.environ.get("DD_VERSION", "0.1.0"),
)
profiler.start()

# Initialize Ray
ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), namespace="backend")

# Create FastAPI app
app = FastAPI(title="HackMIT 2025 Backend", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing and Datadog metrics
@app.middleware("http")
async def add_timing_and_metrics(request: Request, call_next):
    # Start timer
    start_time = time.time()
    
    # Get the route path for more accurate metrics grouping
    route = request.url.path
    method = request.method
    
    # Add custom span for this request
    with tracer.trace("http.request", service="hackmit-backend", resource=f"{method} {route}"):
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add timing header to response
        response.headers["X-Process-Time"] = str(duration)
        
        # Log request info with Datadog
        tracer.current_span().set_tag("http.status_code", response.status_code)
        tracer.current_span().set_tag("http.method", method)
        tracer.current_span().set_tag("http.url", str(request.url))
        
        return response

# Define data models
class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# Define Ray remote functions
@ray.remote
def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process an item with Ray (example function)"""
    # Add tracing to Ray task using Datadog
    with tracer.trace("ray.process_item", service="hackmit-ray-worker"):
        # Simulate some CPU-intensive processing
        result = item.copy()
        
        # Add a calculated field if tax is present
        if result.get("tax") is not None:
            result["price_with_tax"] = result["price"] * (1 + result["tax"])
        
        # Add artificial delay to demonstrate profiling
        if os.environ.get("DEMO_SLOW_PROCESSING", "false").lower() == "true":
            time.sleep(0.5)  # Simulate slow processing
            
        # Add trace information
        tracer.current_span().set_tag("item.id", result.get("id", "unknown"))
        tracer.current_span().set_tag("item.price", result.get("price", 0))
        
        return result

# API routes
@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "HackMIT 2025 Backend API is running"}

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    """Create a new item and process it with Ray"""
    # If no id is provided, assign a random one (in a real app, use a database)
    if item.id is None:
        import random
        item.id = random.randint(1, 10000)
    
    # Process the item using Ray
    with tracer.trace("api.create_item.process", service="hackmit-backend"):
        item_dict = item.model_dump()
        tracer.current_span().set_tag("item.name", item.name)
        result = await asyncio.to_thread(lambda: ray.get(process_item.remote(item_dict)))
    
    # Convert back to Item model
    return Item(**result)

@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    """Placeholder for reading an item (would use a database in a real app)"""
    # In a real application, this would fetch from a database
    # For demo purposes, just return a dummy item
    return Item(
        id=item_id,
        name=f"Example Item {item_id}",
        description="This is a placeholder item",
        price=100.0,
        tax=0.1
    )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify Ray cluster connection and Datadog status"""
    status = {
        "status": "initializing",
        "components": {
            "ray": "unknown",
            "datadog": "unknown"
        },
        "timestamp": time.time()
    }
    
    # Check Ray connection
    try:
        with tracer.trace("health.check.ray", service="hackmit-backend"):
            result = await asyncio.to_thread(lambda: ray.get(ray.remote(lambda: "OK").remote()))
            if result == "OK":
                status["components"]["ray"] = "connected"
            else:
                status["components"]["ray"] = "error"
    except Exception as e:
        status["components"]["ray"] = "error"
        status["ray_error"] = str(e)
    
    # Check Datadog connection
    try:
        with tracer.trace("health.check.datadog", service="hackmit-backend"):
            # This span will be reported to Datadog if connection is working
            status["components"]["datadog"] = "connected"
    except Exception as e:
        status["components"]["datadog"] = "error"
        status["datadog_error"] = str(e)
    
    # Overall status
    if all(v == "connected" for v in status["components"].values()):
        status["status"] = "healthy"
    elif any(v == "error" for v in status["components"].values()):
        status["status"] = "degraded"
        if status["components"]["ray"] == "error":
            # Ray is critical, so raise an exception
            raise HTTPException(status_code=500, detail=f"Ray cluster connection issue: {status.get('ray_error', 'unknown error')}")
    
    return status

# Add endpoint to trigger a profiling sample for demonstration
@app.get("/debug/profile")
async def trigger_profile_sample():
    """Trigger a CPU-intensive operation to demonstrate profiling"""
    with tracer.trace("debug.profile_sample", service="hackmit-backend"):
        # CPU-intensive operation
        result = 0
        for i in range(1000000):
            result += i
        return {"status": "profile_sample_completed", "result": result}

# Add Datadog environment information endpoint
@app.get("/debug/config")
async def debug_config():
    """Return configuration information for debugging"""
    return {
        "datadog": {
            "env": os.environ.get("DD_ENV", "development"),
            "service": "hackmit-backend",
            "version": os.environ.get("DD_VERSION", "0.1.0"),
            "agent_host": os.environ.get("DD_AGENT_HOST", "localhost"),
            "trace_agent_port": os.environ.get("DD_TRACE_AGENT_PORT", 8126),
            "profiling_enabled": True
        },
        "ray": {
            "address": os.environ.get("RAY_ADDRESS", "auto"),
            "namespace": "backend"
        }
    }

# Run the application with uvicorn when script is executed directly
if __name__ == "__main__":
    # For development only - don't use this in production
    try:
        # Configure Datadog settings from environment variables
        # These can be set before running the application
        # export DD_AGENT_HOST=localhost
        # export DD_ENV=development
        # export DD_SERVICE=hackmit-backend
        # export DD_TRACE_SAMPLE_RATE=1.0
        
        import uvicorn
        print("Starting server with Datadog tracing and profiling enabled")
        print(f"Datadog Environment: {os.environ.get('DD_ENV', 'development')}")
        uvicorn.run("entrypoint:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("Error: uvicorn is required to run the application directly.")
        print("Install it with: pip install uvicorn")
        import sys
        sys.exit(1)