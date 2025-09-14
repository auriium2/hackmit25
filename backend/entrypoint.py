import os
import ray
from fastapi import FastAPI, HTTPException, Request, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Dict, Any, Optional, List
import time
import uuid
import tempfile
from pathlib import Path

# Import Datadog libraries for monitoring and profiling
from ddtrace import patch_all, tracer
from ddtrace.profiling import Profiler

# Import benchmark extraction components
from app.workloads.benchmark_extractor import (
    EnhancedBenchmarkExtractor,
    BenchmarkExtractionRequest,
    BenchmarkExtractionResponse,
    process_benchmark_extraction_batch
)

# Initialize Datadog tracing and profiling
patch_all()  # Patch all supported libraries for automatic instrumentation
profiler = Profiler(
    service="hackmit-backend",
    env=os.environ.get("DD_ENV", "development"),
    version=os.environ.get("DD_VERSION", "0.1.0"),
)
profiler.start()

# Initialize Ray (will be done lazily when needed)
_ray_initialized = False

def ensure_ray_initialized():
    global _ray_initialized
    if not _ray_initialized:
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), namespace="backend")
        _ray_initialized = True

# Import workloads for paper processing
from app.workloads.front_agent import SeedPaperRetriever
from app.workloads.crawler import search_openalex_papers, build_citation_graph, PaperCrawler
from app.workloads.llm_processor import process_papers_parallel, generate_research_insights, vectorize_papers

# Create FastAPI app
app = FastAPI(title="RefGraph API", version="0.1.0")

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

# Mock in-memory store for task state (use Redis in production)
tasks: Dict[str, Dict] = {}

# ----------- MODELS -----------

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    systemid: str
    status: str

class StatusResponse(BaseModel):
    systemid: str
    status: str

class GraphResponse(BaseModel):
    systemid: str
    graph: Dict

class PaperDetailsResponse(BaseModel):
    openalex_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    citations: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    authors: Optional[List[str]] = []
    venue: Optional[str] = None

# Legacy Item model for existing endpoints
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

# ----------- REFGRAPH API ENDPOINTS -----------

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "RefGraph API is running", "version": "0.1.0"}

@app.post("/query", response_model=QueryResponse)
async def query_papers(req: QueryRequest):
    """
    Takes in initial user string, initializes a backend job,
    returns systemid and ok/fail.
    """
    with tracer.trace("api.query_papers", service="refgraph-api"):
        tracer.current_span().set_tag("query", req.query)

        if not req.query.strip():
            return {"systemid": "", "status": "fail"}

        systemid = str(uuid.uuid4())
        tasks[systemid] = {
            "status": "started",
            "graph": None,
            "query": req.query,
            "start_time": time.time()
        }

        # Start background processing with your Ray workloads
        asyncio.create_task(process_research_query(systemid, req.query))

        tracer.current_span().set_tag("systemid", systemid)
        return {"systemid": systemid, "status": "ok"}

@app.get("/check_status", response_model=StatusResponse)
async def check_status(systemid: str = Query(...)):
    """
    Checks the current status of a job by systemid.
    """
    with tracer.trace("api.check_status", service="refgraph-api"):
        tracer.current_span().set_tag("systemid", systemid)

        if systemid not in tasks:
            return {"systemid": systemid, "status": "fail"}

        return {"systemid": systemid, "status": tasks[systemid]["status"]}

@app.get("/pull_final_graph", response_model=GraphResponse)
async def pull_final_graph(systemid: str = Query(...)):
    """
    Returns the final graph once processing is complete.
    """
    with tracer.trace("api.pull_final_graph", service="refgraph-api"):
        tracer.current_span().set_tag("systemid", systemid)

        if systemid not in tasks:
            return {"systemid": systemid, "graph": {}}

        job = tasks[systemid]
        if job["status"] != "done":
            return {"systemid": systemid, "graph": {}}

        return {"systemid": systemid, "graph": job["graph"]}

# ----------- BACKGROUND PROCESSING -----------

async def process_research_query(systemid: str, query: str):
    """
    Real backend processing using your Ray workloads.
    Replaces the mock simulate_processing function.
    """
    try:
        with tracer.trace("api.process_research_query", service="refgraph-api"):
            tracer.current_span().set_tag("systemid", systemid)
            tracer.current_span().set_tag("query", query)

            # Ensure Ray is initialized
            ensure_ray_initialized()

            # Step 1: Get seed papers using front agent
            tasks[systemid]["status"] = "finding seed papers"
            seed_retriever = SeedPaperRetriever()
            seed_papers = await asyncio.to_thread(
                seed_retriever.retrieve_seed_papers, query
            )

            if not seed_papers:
                tasks[systemid]["status"] = "fail: no seed papers found"
                return

            # Step 2: Build citation graph
            tasks[systemid]["status"] = "building citation graph"
            graph_result = await asyncio.to_thread(
                lambda: ray.get(build_citation_graph.remote(seed_papers, max_radius=2))
            )

            # Step 3: Process papers with domain experts
            tasks[systemid]["status"] = "analyzing papers with AI agents"
            papers_for_analysis = []
            for paper_id, paper_info in graph_result["nodes"].items():
                papers_for_analysis.append({
                    "arxiv_id": paper_id,
                    "title": paper_info.get("title", ""),
                    "abstract": paper_info.get("abstract", ""),
                    "authors": paper_info.get("authors", []),
                    "venue": paper_info.get("venue", "")
                })

            analysis_result = await asyncio.to_thread(
                lambda: ray.get(process_papers_parallel.remote(papers_for_analysis, num_agents=4))
            )

            # Step 4: Generate insights and vectorize
            tasks[systemid]["status"] = "generating insights"
            insights, vectors = await asyncio.gather(
                asyncio.to_thread(
                    lambda: ray.get(generate_research_insights.remote(analysis_result, query))
                ),
                asyncio.to_thread(
                    lambda: ray.get(vectorize_papers.remote(analysis_result))
                )
            )

            # Step 5: Build final graph structure
            tasks[systemid]["status"] = "finalizing graph"

            # Create nodes with enhanced data
            nodes = []
            for paper_id, paper_info in graph_result["nodes"].items():
                analysis = analysis_result["papers"].get(paper_id, {})

                # Extract year from publication date
                year = 2023
                if paper_info.get("published_date") and paper_info["published_date"] != "Unknown":
                    try:
                        year = int(paper_info["published_date"][:4])
                    except:
                        year = 2023

                node = {
                    "id": paper_id,
                    "label": paper_info.get("title", "Unknown Title"),
                    "data": {
                        "id": paper_id,
                        "title": paper_info.get("title", "Unknown Title"),
                        "authors": paper_info.get("authors", []),
                        "year": year,
                        "abstract": str(paper_info.get("abstract", "")),
                        "doi": paper_info.get("doi"),
                        "venue": paper_info.get("venue"),
                        "citations": paper_info.get("citation_count", 0),
                        "cluster": analysis.get("domain", "Computer Science"),
                        "confidence": 85.0,
                        "summary": analysis.get("summary", ""),
                        "metrics": analysis.get("metrics", {}),
                        "embedding": vectors["vectors"].get(paper_id, [])
                    }
                }
                nodes.append(node)

            # Create edges from graph result
            edges = []
            for edge in graph_result["edges"]:
                edges.append({
                    "id": f"{edge['type']}-{edge['from']}-{edge['to']}",
                    "source": edge["from"],
                    "target": edge["to"],
                    "type": edge["type"],
                    "data": {
                        "relation": edge["type"],
                        "confidence": 0.8
                    }
                })

            # Final graph structure
            final_graph = {
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    "query": query,
                    "total_papers": len(nodes),
                    "processing_time": time.time() - tasks[systemid]["start_time"],
                    "insights": insights,
                    "seed_papers": seed_papers
                }
            }

            # Mark as complete
            tasks[systemid]["status"] = "done"
            tasks[systemid]["graph"] = final_graph

            tracer.current_span().set_tag("papers_processed", len(nodes))
            tracer.current_span().set_tag("edges_created", len(edges))

    except Exception as e:
        tracer.current_span().set_tag("error", str(e))
        tasks[systemid]["status"] = f"fail: {str(e)}"

@app.get("/get_details", response_model=PaperDetailsResponse)
async def get_details(openalexid: str = Query(...)):
    """
    Fetches OpenAlex paper details for a given paper node.
    """
    with tracer.trace("api.get_details", service="refgraph-api"):
        tracer.current_span().set_tag("openalex_id", openalexid)

        try:
            # Ensure Ray is initialized
            ensure_ray_initialized()

            # Use Ray to search for paper details
            papers = await asyncio.to_thread(
                lambda: ray.get(search_openalex_papers.remote(openalexid))
            )

            if not papers:
                return PaperDetailsResponse(
                    openalex_id=openalexid,
                    title="Paper not found",
                    abstract="Unable to retrieve paper details",
                    year=None,
                    citations=None,
                    doi=None,
                    url=None
                )

            paper = papers[0]

            # Extract year from publication date
            year = None
            if paper.get("published_date") and paper["published_date"] != "Unknown":
                try:
                    year = int(paper["published_date"][:4])
                except:
                    year = None

            return PaperDetailsResponse(
                openalex_id=paper["openalex_id"],
                title=paper["title"],
                abstract=paper["abstract"] or "No abstract available",
                year=year,
                citations=paper["citation_count"],
                doi=paper["doi"],
                url=f"https://openalex.org/{paper['openalex_id']}",
                authors=paper["authors"],
                venue=paper["venue"] if paper["venue"] != "Unknown Venue" else None
            )

        except Exception as e:
            tracer.current_span().set_tag("error", str(e))
            return PaperDetailsResponse(
                openalex_id=openalexid,
                title="Error retrieving paper",
                abstract=f"Error: {str(e)}",
                year=None,
                citations=None,
                doi=None,
                url=None
            )

# ----------- LEGACY ENDPOINTS -----------

@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    """Legacy endpoint - Create a new item and process it with Ray"""
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
    """Legacy endpoint - Placeholder for reading an item"""
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


# Benchmark extraction endpoints
@app.post("/extract-benchmarks/text", response_model=BenchmarkExtractionResponse)
async def extract_benchmarks_from_text(request: BenchmarkExtractionRequest):
    """Extract benchmarks from paper text using LLM."""
    if not request.paper_text:
        raise HTTPException(status_code=400, detail="paper_text is required")

    with tracer.trace("api.extract_benchmarks.text", service="hackmit-backend"):
        tracer.current_span().set_tag("domain_hint", request.domain_hint)
        tracer.current_span().set_tag("text_length", len(request.paper_text))

        try:
            # Create extractor actor
            extractor = EnhancedBenchmarkExtractor.remote(domain=request.domain_hint or "general")

            # Extract benchmarks
            result = await extractor.extract_benchmarks_from_text.remote(
                request.paper_text, request.domain_hint
            )

            # Create response
            response_data = {
                "benchmarks": result.get("benchmarks", []),
                "metric_values": result.get("metric_values", []),
                "extraction_metadata": result.get("extraction_metadata", {}),
                "paper_metadata": {
                    "text_length": len(request.paper_text),
                    "processing_timestamp": time.time()
                }
            }

            if request.extract_full_analysis and result.get("benchmarks"):
                # Create full paper analysis if requested
                paper_id = f"text_analysis_{int(time.time())}"
                extractor_local = await extractor.create_full_paper_analysis.remote(result, paper_id)
                response_data["paper_analysis"] = extractor_local.model_dump() if extractor_local else None

            return BenchmarkExtractionResponse(**response_data)

        except Exception as e:
            tracer.current_span().set_tag("error", str(e))
            raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.post("/extract-benchmarks/pdf")
async def extract_benchmarks_from_pdf(
    file: UploadFile = File(...),
    domain_hint: Optional[str] = None,
    extract_full_analysis: bool = True
):
    """Extract benchmarks from uploaded PDF file."""
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tracer.trace("api.extract_benchmarks.pdf", service="hackmit-backend"):
        tracer.current_span().set_tag("filename", file.filename)
        tracer.current_span().set_tag("domain_hint", domain_hint)

        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                # Create extractor actor
                extractor = EnhancedBenchmarkExtractor.remote(domain=domain_hint or "general")

                # Extract benchmarks from PDF
                result = await extractor.extract_from_pdf.remote(temp_file_path, domain_hint)

                # Create response
                response_data = {
                    "benchmarks": result.get("benchmarks", []),
                    "metric_values": result.get("metric_values", []),
                    "extraction_metadata": result.get("extraction_metadata", {}),
                    "paper_metadata": result.get("paper_metadata", {})
                }

                response_data["paper_metadata"]["original_filename"] = file.filename

                if extract_full_analysis and result.get("benchmarks"):
                    # Create full paper analysis if requested
                    paper_id = f"pdf_{Path(file.filename).stem}"
                    extractor_local = await extractor.create_full_paper_analysis.remote(result, paper_id)
                    response_data["paper_analysis"] = extractor_local.model_dump() if extractor_local else None

                return BenchmarkExtractionResponse(**response_data)

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            tracer.current_span().set_tag("error", str(e))
            raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")


@app.post("/extract-benchmarks/batch")
async def extract_benchmarks_batch(papers_data: List[Dict[str, Any]], domain_hint: Optional[str] = None):
    """Process multiple papers for benchmark extraction in parallel."""
    if not papers_data:
        raise HTTPException(status_code=400, detail="papers_data cannot be empty")

    with tracer.trace("api.extract_benchmarks.batch", service="hackmit-backend"):
        tracer.current_span().set_tag("paper_count", len(papers_data))
        tracer.current_span().set_tag("domain_hint", domain_hint)

        try:
            # Process papers in parallel using Ray
            result = await process_benchmark_extraction_batch.remote(
                papers_data, domain_hint, num_extractors=4
            )

            return result

        except Exception as e:
            tracer.current_span().set_tag("error", str(e))
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@app.get("/sample-papers")
async def list_sample_papers():
    """List available sample papers for testing."""
    sample_dir = Path("sample_papers")
    if not sample_dir.exists():
        return {"papers": [], "message": "Sample papers directory not found"}

    papers = []
    for pdf_file in sample_dir.glob("*.pdf"):
        papers.append({
            "filename": pdf_file.name,
            "path": str(pdf_file),
            "size_bytes": pdf_file.stat().st_size
        })

    return {"papers": papers, "total_count": len(papers)}


@app.post("/extract-benchmarks/sample/{filename}")
async def extract_benchmarks_from_sample(
    filename: str,
    domain_hint: Optional[str] = None,
    extract_full_analysis: bool = True
):
    """Extract benchmarks from a sample paper by filename."""
    sample_path = Path("sample_papers") / filename

    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample paper {filename} not found")

    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tracer.trace("api.extract_benchmarks.sample", service="hackmit-backend"):
        tracer.current_span().set_tag("filename", filename)
        tracer.current_span().set_tag("domain_hint", domain_hint)

        try:
            # Create extractor actor
            extractor = EnhancedBenchmarkExtractor.remote(domain=domain_hint or "general")

            # Extract benchmarks
            result = await extractor.extract_from_pdf.remote(str(sample_path), domain_hint)

            # Create response
            response_data = {
                "benchmarks": result.get("benchmarks", []),
                "metric_values": result.get("metric_values", []),
                "extraction_metadata": result.get("extraction_metadata", {}),
                "paper_metadata": result.get("paper_metadata", {})
            }

            if extract_full_analysis and result.get("benchmarks"):
                # Create full paper analysis if requested
                paper_id = Path(filename).stem
                extractor_local = await extractor.create_full_paper_analysis.remote(result, paper_id)
                response_data["paper_analysis"] = extractor_local.model_dump() if extractor_local else None

            return BenchmarkExtractionResponse(**response_data)

        except Exception as e:
            tracer.current_span().set_tag("error", str(e))
            raise HTTPException(status_code=500, detail=f"Sample extraction failed: {str(e)}")