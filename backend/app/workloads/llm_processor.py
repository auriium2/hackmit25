"""
Ray workloads for LLM-based paper analysis and metric extraction.
"""

import ray
import asyncio
from typing import Dict, List, Any, Optional
import time
import json
import requests
import os
import logging
from dotenv import load_dotenv

# Try to import ddtrace, but make it optional
try:
    from ddtrace import tracer
    DDTRACE_AVAILABLE = True
except ImportError:
    # Create a mock tracer for standalone usage
    class MockTracer:
        def trace(self, *args, **kwargs):
            return self
        def current_span(self):
            return self
        def set_tag(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    tracer = MockTracer()
    DDTRACE_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaudeLLMClient:
    """Client for Claude API to perform metric extraction from research papers."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            logger.warning("Claude API key not found. Falling back to mock implementation.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("Claude LLM client initialized successfully")

    def extract_metrics_from_text(self, paper_text: str, paper_title: str = "") -> Dict[str, Any]:
        """
        Extract key-value pairs of metrics from research paper text using Claude LLM.

        Args:
            paper_text: The body text of the research paper
            paper_title: Optional title for context

        Returns:
            Dictionary of metric_name -> metric_value pairs
        """
        if not self.enabled:
            return self._fallback_metric_extraction(paper_text, paper_title)

        # Truncate text if too long (Claude has token limits)
        max_chars = 15000  # Conservative estimate for token limits
        if len(paper_text) > max_chars:
            paper_text = paper_text[:max_chars] + "...[truncated]"

        prompt = f"""
You are a research paper analysis expert. Extract ALL quantitative metrics and performance measurements from the following research paper text.

Paper Title: {paper_title}

Paper Text:
{paper_text}

Extract metrics as key-value pairs. Include:
- Performance metrics (accuracy, precision, recall, F1-score, BLEU, ROUGE, etc.)
- Computational metrics (training time, inference time, memory usage, FLOPs)
- Dataset statistics (dataset size, number of parameters, epochs)
- Benchmarks scores (on specific datasets like ImageNet, CIFAR, etc.)
- Error rates (RMSE, MAE, cross-entropy loss, perplexity, etc.)
- Model characteristics (model size, depth, width, etc.)

Return ONLY a valid JSON object with metric names as keys and their values as strings. Include the units when available.
Example: {{"accuracy": "94.2%", "training_time": "12 hours", "model_parameters": "175B", "BLEU_score": "42.1"}}

JSON:"""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1000,
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            response.raise_for_status()

            content = response.json()["content"][0]["text"].strip()

            # Try to parse the JSON response
            try:
                metrics = json.loads(content)
                logger.info(f"Successfully extracted {len(metrics)} metrics using Claude LLM")
                return metrics if isinstance(metrics, dict) else {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse Claude response as JSON, attempting to extract metrics from text")
                return self._parse_metrics_from_text(content)

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return self._fallback_metric_extraction(paper_text, paper_title)

    def _parse_metrics_from_text(self, text: str) -> Dict[str, Any]:
        """Attempt to extract metrics from raw text response."""
        metrics = {}
        lines = text.split('\n')

        for line in lines:
            if ':' in line and any(char.isdigit() for char in line):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('"').strip("'")
                    value = parts[1].strip().strip(',').strip('"').strip("'")
                    if key and value:
                        metrics[key] = value

        return metrics

    def _fallback_metric_extraction(self, paper_text: str, paper_title: str) -> Dict[str, Any]:
        """Fallback implementation when Claude API is not available."""
        # Simple regex-based extraction for common patterns
        import re

        metrics = {}
        text = paper_text.lower()

        # Common accuracy patterns
        accuracy_patterns = [
            r'accuracy[:\s]+(\d+\.?\d*%?)',
            r'acc[:\s]+(\d+\.?\d*%?)',
            r'(\d+\.?\d*%?)\s+accuracy'
        ]

        for pattern in accuracy_patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics["accuracy"] = matches[0] + ("%" if "%" not in matches[0] else "")
                break

        # F1 Score patterns
        f1_patterns = [r'f1[:\s-]+(\d+\.?\d*)', r'f1[-_]?score[:\s]+(\d+\.?\d*)']
        for pattern in f1_patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics["f1_score"] = matches[0]
                break

        # BLEU score patterns
        bleu_patterns = [r'bleu[:\s-]+(\d+\.?\d*)', r'bleu[-_]?score[:\s]+(\d+\.?\d*)']
        for pattern in bleu_patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics["bleu_score"] = matches[0]
                break

        # Training time patterns
        time_patterns = [
            r'training.*?(\d+\.?\d*\s*(?:hours?|hrs?|minutes?|mins?|days?))',
            r'(\d+\.?\d*\s*(?:hours?|hrs?))\s+.*?training'
        ]

        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics["training_time"] = matches[0]
                break

        # Model parameters
        param_patterns = [
            r'(\d+\.?\d*[bmk]?)\s*parameters',
            r'model.*?(\d+\.?\d*[bmk]?)\s*(?:parameters|params)'
        ]

        for pattern in param_patterns:
            matches = re.findall(pattern, text)
            if matches:
                metrics["model_parameters"] = matches[0]
                break

        logger.info(f"Fallback extraction found {len(metrics)} metrics")
        return metrics


@ray.remote
class DomainExpertAgent:
    """Ray actor representing a domain expert for analyzing a specific paper."""

    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.extracted_metrics = {}
        self.paper_summaries = {}
        self.llm_client = ClaudeLLMClient()

    def analyze_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a paper and extract relevant metrics and summary.

        Args:
            paper_data: Paper information including text content

        Returns:
            Dictionary containing metrics, summary, and analysis
        """
        paper_id = paper_data.get("openalex_id", "unknown")

        with tracer.trace("llm.analyze_paper", service="hackmit-domain-expert"):
            tracer.current_span().set_tag("paper_id", paper_id)
            tracer.current_span().set_tag("domain", self.domain)

            # Simulate paper analysis (replace with actual LLM calls)
            metrics = self._extract_metrics(paper_data)
            summary = self._generate_summary(paper_data)
            benchmarks = self._identify_benchmarks(paper_data)

            analysis_result = {
                "paper_id": paper_id,
                "domain": self.domain,
                "summary": summary,
                "metrics": metrics,
                "benchmarks": benchmarks,
                "analysis_timestamp": time.time()
            }

            tracer.current_span().set_tag("metrics_found", len(metrics))
            tracer.current_span().set_tag("benchmarks_found", len(benchmarks))

            return analysis_result

    def _extract_metrics(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quantitative metrics from the paper using Claude LLM."""
        paper_id = paper_data.get("openalex_id", "unknown")
        paper_title = paper_data.get("title", "")

        # Get paper text - this could come from abstract, full text, or other fields
        paper_text = ""

        # Try to build text from available paper data
        if "abstract" in paper_data:
            paper_text += f"Abstract: {paper_data['abstract']}\n\n"

        if "full_text" in paper_data:
            paper_text += f"Content: {paper_data['full_text']}\n\n"
        elif "content" in paper_data:
            paper_text += f"Content: {paper_data['content']}\n\n"
        elif "text" in paper_data:
            paper_text += f"Content: {paper_data['text']}\n\n"

        # Include other relevant fields
        if "venue" in paper_data and paper_data["venue"]:
            paper_text += f"Published in: {paper_data['venue']}\n"

        if "year" in paper_data:
            paper_text += f"Year: {paper_data['year']}\n"

        # If we still don't have enough text, create a minimal description
        if len(paper_text.strip()) < 50:
            paper_text = f"Research paper titled: {paper_title}"

        logger.info(f"Extracting metrics from paper {paper_id} ({len(paper_text)} chars)")

        try:
            # Use Claude LLM to extract metrics
            metrics = self.llm_client.extract_metrics_from_text(paper_text, paper_title)

            # Add paper metadata
            if metrics:
                metrics["_paper_id"] = paper_id
                metrics["_extraction_timestamp"] = time.time()

            logger.info(f"Extracted {len(metrics)} metrics from paper {paper_id}")
            return metrics

        except Exception as e:
            logger.error(f"Error extracting metrics from paper {paper_id}: {e}")

            # Return minimal fallback data
            return {
                "_paper_id": paper_id,
                "_extraction_error": str(e),
                "_extraction_timestamp": time.time()
            }

    def _generate_summary(self, paper_data: Dict[str, Any]) -> str:
        """Generate a concise summary of the paper."""
        # Simulate summary generation
        time.sleep(0.4)

        title = paper_data.get("title", "Unknown Title")
        return f"This paper presents research on {title.lower()}. Key contributions include novel methodologies and improved performance metrics."

    def _identify_benchmarks(self, paper_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify and standardize benchmark names used in the paper."""
        # Simulate benchmark identification
        time.sleep(0.2)

        # Mock benchmark identification
        paper_id = paper_data.get("openalex_id", "unknown")
        benchmark_types = ["ImageNet", "CIFAR-10", "MNIST", "COCO", "Penn Treebank"]
        num_benchmarks = (hash(paper_id) % 3) + 1

        benchmarks = []
        for i in range(num_benchmarks):
            benchmark_name = benchmark_types[hash(f"{paper_id}_{i}") % len(benchmark_types)]
            benchmarks.append({
                "name": benchmark_name,
                "category": "computer_vision" if benchmark_name in ["ImageNet", "CIFAR-10", "MNIST", "COCO"] else "nlp",
                "standardized_key": benchmark_name.lower().replace("-", "_"),
                "metric_type": "accuracy"
            })

        return benchmarks


@ray.remote
def process_papers_parallel(papers: List[Dict[str, Any]], num_agents: int = 4) -> Dict[str, Any]:
    """
    Process multiple papers in parallel using domain expert agents.

    Args:
        papers: List of paper data dictionaries
        num_agents: Number of domain expert agents to spawn

    Returns:
        Combined analysis results from all papers
    """
    with tracer.trace("llm.process_papers_parallel", service="hackmit-llm-processor"):
        tracer.current_span().set_tag("paper_count", len(papers))
        tracer.current_span().set_tag("num_agents", num_agents)

        # Create domain expert agents
        agents = [DomainExpertAgent.remote(domain=f"expert_{i}") for i in range(num_agents)]

        # Distribute papers across agents
        analysis_tasks = []
        for i, paper in enumerate(papers):
            agent = agents[i % num_agents]
            task = agent.analyze_paper.remote(paper)
            analysis_tasks.append(task)

        # Wait for all analyses to complete
        analysis_results = ray.get(analysis_tasks)

        # Combine results
        combined_results = {
            "papers": {},
            "global_benchmarks": {},
            "processing_metadata": {
                "total_papers": len(papers),
                "processing_timestamp": time.time(),
                "agents_used": num_agents
            }
        }

        # Process individual paper results
        for result in analysis_results:
            paper_id = result["paper_id"]
            combined_results["papers"][paper_id] = result

            # Update global benchmark registry
            for benchmark in result["benchmarks"]:
                benchmark_key = benchmark["standardized_key"]
                if benchmark_key not in combined_results["global_benchmarks"]:
                    combined_results["global_benchmarks"][benchmark_key] = {
                        "name": benchmark["name"],
                        "category": benchmark["category"],
                        "papers": [],
                        "metric_type": benchmark["metric_type"]
                    }
                combined_results["global_benchmarks"][benchmark_key]["papers"].append(paper_id)

        tracer.current_span().set_tag("unique_benchmarks", len(combined_results["global_benchmarks"]))
        return combined_results


@ray.remote
def generate_research_insights(analysis_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    """
    Generate high-level insights and comparisons based on paper analysis.

    Args:
        analysis_data: Combined analysis results from all papers
        user_query: Original user research query

    Returns:
        Research insights and recommendations
    """
    with tracer.trace("llm.generate_insights", service="hackmit-insight-generator"):
        tracer.current_span().set_tag("user_query", user_query)
        tracer.current_span().set_tag("papers_analyzed", len(analysis_data.get("papers", {})))

        # Simulate insight generation (replace with actual LLM)
        time.sleep(0.6)

        insights = {
            "query": user_query,
            "summary": f"Analysis of {len(analysis_data.get('papers', {}))} papers related to: {user_query}",
            "key_trends": [
                "Increasing focus on transformer architectures",
                "Improvement in benchmark performance over time",
                "Growing dataset sizes and computational requirements"
            ],
            "benchmark_comparison": _compare_benchmarks(analysis_data),
            "recommendations": [
                "Consider recent papers with highest performance on shared benchmarks",
                "Look for papers that introduce novel evaluation metrics",
                "Focus on papers with reproducible results and available code"
            ],
            "generation_timestamp": time.time()
        }

        tracer.current_span().set_tag("trends_identified", len(insights["key_trends"]))
        return insights


def _compare_benchmarks(analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare papers across shared benchmarks."""
    comparisons = []
    global_benchmarks = analysis_data.get("global_benchmarks", {})

    for benchmark_key, benchmark_info in global_benchmarks.items():
        if len(benchmark_info["papers"]) > 1:  # Only compare if multiple papers use this benchmark
            comparison = {
                "benchmark": benchmark_info["name"],
                "category": benchmark_info["category"],
                "papers_compared": len(benchmark_info["papers"]),
                "paper_ids": benchmark_info["papers"],
                # In a real implementation, this would extract and compare actual metric values
                "performance_ranking": benchmark_info["papers"]  # Simplified ranking
            }
            comparisons.append(comparison)

    return comparisons


def extract_metrics(paper_text: str, paper_title: str = "", clean_output: bool = False) -> Dict[str, Any]:
    """
    Direct function to extract metrics from paper text without Ray.
    Can be imported and used anywhere in the application.

    Args:
        paper_text: The body text of the research paper
        paper_title: Optional title for context
        clean_output: If True, removes metadata fields (keys starting with '_')

    Returns:
        Dictionary of metric_name -> metric_value pairs

    Example:
        >>> from backend.app.workloads.llm_processor import extract_metrics
        >>> metrics = extract_metrics("The robot achieves 3.7 m/s speed...", "Robotics Paper")
        >>> print(metrics)  # {'max_speed': '3.7 m/s', ...}
        >>>
        >>> # For JSON-ready output without metadata
        >>> clean_metrics = extract_metrics("...", clean_output=True)
    """
    # Initialize the Claude LLM client
    client = ClaudeLLMClient()

    # Extract metrics
    metrics = client.extract_metrics_from_text(paper_text, paper_title)

    # Add metadata
    metrics["_extraction_method"] = "claude_llm" if client.enabled else "fallback_regex"

    # Return clean output if requested
    if clean_output:
        return {k: v for k, v in metrics.items() if not k.startswith("_")}

    return metrics


@ray.remote
def vectorize_papers(papers_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create vector representations of papers for similarity analysis.

    Args:
        papers_data: Analyzed paper data

    Returns:
        Vector representations and similarity matrix
    """
    with tracer.trace("llm.vectorize_papers", service="hackmit-vectorizer"):
        tracer.current_span().set_tag("paper_count", len(papers_data.get("papers", {})))

        # Simulate vectorization (replace with actual embedding model)
        time.sleep(0.5)

        papers = papers_data.get("papers", {})
        vectors = {}
        similarity_matrix = {}

        for paper_id in papers.keys():
            # Mock vector (in reality, would use sentence transformers or similar)
            vector = [hash(f"{paper_id}_{i}") % 100 / 100.0 for i in range(384)]  # 384-dim vector
            vectors[paper_id] = vector

            # Calculate similarities (simplified cosine similarity simulation)
            similarity_matrix[paper_id] = {}
            for other_paper_id in papers.keys():
                if paper_id != other_paper_id:
                    # Mock similarity score
                    similarity = (hash(f"{paper_id}_{other_paper_id}") % 100) / 100.0
                    similarity_matrix[paper_id][other_paper_id] = similarity

        result = {
            "vectors": vectors,
            "similarity_matrix": similarity_matrix,
            "vectorization_metadata": {
                "model": "mock-sentence-transformer",
                "dimensions": 384,
                "timestamp": time.time()
            }
        }

        tracer.current_span().set_tag("vectors_generated", len(vectors))
        return result


# Main execution - allows running directly with text input
if __name__ == "__main__":
    import sys

    # Check for JSON flag
    output_json = "--json" in sys.argv
    if output_json:
        sys.argv.remove("--json")

    # Check if text was provided
    if len(sys.argv) > 1:
        # Get text from command line argument
        paper_text = " ".join(sys.argv[1:])
        paper_title = "Command Line Input"
    else:
        # Use default example if no text provided
        paper_text = """
        The robot achieves a maximum speed of 3.7 m/s, fully utilizing its hardware capabilities.
        MPC operates at 40 Hz, while WBIC runs at 500 Hz, enabling real-time responsiveness.
        The Mini-Cheetah quadruped robot, equipped with 12 actuated joints, serves as the experimental platform.
        The system is tested across six different gait patterns to demonstrate versatility.
        """
        paper_title = "Example Paper"
        if not output_json:
            print("ℹ️  No text provided. Using example text.")
            print("Usage: python llm_processor.py [--json] 'your paper text here'")
            print()

    # Extract metrics directly
    metrics = extract_metrics(paper_text, paper_title)

    # Remove metadata fields for clean output
    clean_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}

    if output_json:
        # Output as JSON
        print(json.dumps(clean_metrics, indent=2))
    else:
        # Output as formatted text
        print("📄 Processing paper text...")
        print("=" * 60)
        print(f"📊 Extracted {len(clean_metrics)} metrics:")
        print()

        for key, value in clean_metrics.items():
            print(f"  • {key}: {value}")

        print()
        print(f"🔧 Extraction method: {metrics.get('_extraction_method', 'unknown')}")
        print("=" * 60)