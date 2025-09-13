"""
Pydantic models for research workflow and paper analysis.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class QueryStatus(str, Enum):
    """Status of a research query."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryType(str, Enum):
    """Type of research query."""
    STATE_OF_ART = "state_of_art"
    COMPARISON = "comparison"
    HISTORICAL = "historical"
    CUSTOM = "custom"


class ResearchQuery(BaseModel):
    """User's research query input."""
    query: str = Field(..., description="Natural language research query")
    query_type: Optional[QueryType] = Field(None, description="Type of query (auto-detected if not provided)")
    max_papers: int = Field(50, ge=1, le=200, description="Maximum number of papers to analyze")
    max_depth: int = Field(2, ge=1, le=4, description="Maximum citation graph depth")
    domains: Optional[List[str]] = Field(None, description="Specific domains to focus on")

    class Config:
        schema_extra = {
            "example": {
                "query": "compare model predictive control to reinforcement learning",
                "query_type": "comparison",
                "max_papers": 30,
                "max_depth": 2,
                "domains": ["control_theory", "machine_learning"]
            }
        }


class SeedPaper(BaseModel):
    """Seed paper for analysis."""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    title: str
    url: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "arxiv_id": "2023.12345",
                "title": "Advanced Model Predictive Control Techniques"
            }
        }


class SeedPaperRequest(BaseModel):
    """Request to add seed papers directly."""
    papers: List[SeedPaper]
    query_description: Optional[str] = Field(None, description="Optional description of what to analyze")


class Paper(BaseModel):
    """Paper metadata and information."""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    venue: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    citation_count: Optional[int] = None


class Benchmark(BaseModel):
    """Standardized benchmark information."""
    name: str
    category: str
    standardized_key: str
    metric_type: str
    description: Optional[str] = None


class MetricValue(BaseModel):
    """A specific metric value from a paper."""
    benchmark: str
    value: str
    unit: Optional[str] = None
    context: Optional[str] = None


class PaperAnalysis(BaseModel):
    """Analysis results for a single paper."""
    paper_id: str
    domain: str
    summary: str
    metrics: Dict[str, Any]
    benchmarks: List[Benchmark]
    metric_values: List[MetricValue] = Field(default_factory=list)
    analysis_timestamp: float


class CitationEdge(BaseModel):
    """Edge in the citation graph."""
    from_paper: str
    to_paper: str
    edge_type: Literal["cites", "references"]


class CitationGraph(BaseModel):
    """Citation graph structure."""
    nodes: Dict[str, Paper]
    edges: List[CitationEdge]
    metadata: Dict[str, Any]


class BenchmarkComparison(BaseModel):
    """Comparison of papers across a shared benchmark."""
    benchmark: str
    category: str
    papers_compared: int
    paper_ids: List[str]
    performance_ranking: List[str]
    metric_type: str


class ResearchInsights(BaseModel):
    """High-level insights from paper analysis."""
    query: str
    summary: str
    key_trends: List[str]
    benchmark_comparisons: List[BenchmarkComparison]
    recommendations: List[str]
    generation_timestamp: float


class QueryResult(BaseModel):
    """Complete results for a research query."""
    query_id: str
    status: QueryStatus
    query: ResearchQuery
    citation_graph: Optional[CitationGraph] = None
    paper_analyses: Optional[Dict[str, PaperAnalysis]] = None
    insights: Optional[ResearchInsights] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processing_time: Optional[float] = None


class QueryStatus(BaseModel):
    """Status information for a running query."""
    query_id: str
    status: QueryStatus
    progress: Dict[str, Any] = Field(default_factory=dict)
    current_stage: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None


class PaperComparison(BaseModel):
    """Direct comparison between specific papers."""
    paper_ids: List[str]
    shared_benchmarks: List[str]
    comparison_matrix: Dict[str, Dict[str, Any]]
    similarity_scores: Dict[str, float]


class ComparisonRequest(BaseModel):
    """Request to compare specific papers."""
    paper_ids: List[str] = Field(..., min_items=2, description="Papers to compare")
    benchmark_filter: Optional[List[str]] = Field(None, description="Only compare on these benchmarks")
    include_similarity: bool = Field(True, description="Include semantic similarity scores")


class HealthStatus(BaseModel):
    """Health status of the system components."""
    status: Literal["healthy", "degraded", "error"]
    components: Dict[str, str]
    timestamp: float
    ray_error: Optional[str] = None
    datadog_error: Optional[str] = None