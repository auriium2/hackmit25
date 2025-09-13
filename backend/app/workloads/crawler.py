"""
Ray workloads for web crawling and paper discovery using OpenAlex API.
"""

import ray
import asyncio
import httpx
from typing import Dict, List, Any, Set, Optional
from ddtrace import tracer
import time


@ray.remote
class PaperCrawler:
    """Ray actor for crawling citation networks from seed papers."""

    def __init__(self):
        self.visited_papers: Set[str] = set()
        self.paper_cache: Dict[str, Dict[str, Any]] = {}

    def crawl_paper_citations(self, paper_id: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Crawl citations for a given paper up to max_depth.

        Args:
            paper_id: ArXiv ID or DOI of the paper
            max_depth: Maximum depth to crawl citations

        Returns:
            Dictionary containing paper info and citation network
        """
        with tracer.trace("crawler.crawl_citations", service="hackmit-crawler"):
            tracer.current_span().set_tag("paper_id", paper_id)
            tracer.current_span().set_tag("max_depth", max_depth)

            # Simulate paper crawling (replace with actual implementation)
            paper_info = self._fetch_paper_info(paper_id)
            citations = self._fetch_citations(paper_id, max_depth)
            references = self._fetch_references(paper_id, max_depth)

            result = {
                "paper_id": paper_id,
                "paper_info": paper_info,
                "citations": citations,
                "references": references,
                "crawl_timestamp": time.time()
            }

            tracer.current_span().set_tag("citations_found", len(citations))
            tracer.current_span().set_tag("references_found", len(references))

            return result

    def _fetch_paper_info(self, paper_id: str) -> Dict[str, Any]:
        """Fetch basic paper information from OpenAlex."""
        with tracer.trace("crawler.fetch_paper_info", service="hackmit-openalex"):
            try:
                # Determine if paper_id is OpenAlex ID, DOI, or other format
                if paper_id.startswith("https://openalex.org/"):
                    openalex_id = paper_id
                elif paper_id.startswith("10."):
                    openalex_id = f"https://doi.org/{paper_id}"
                else:
                    # Assume it's already an OpenAlex work ID or needs search
                    openalex_id = paper_id if paper_id.startswith("W") else None

                if openalex_id:
                    url = f"https://api.openalex.org/works/{openalex_id}"
                else:
                    # Search by identifier
                    url = f"https://api.openalex.org/works?filter=ids.openalex:{paper_id}"

                with httpx.Client() as client:
                    response = client.get(url, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()

                    if "results" in data and data["results"]:
                        work = data["results"][0]
                    else:
                        work = data

                    return {
                        "openalex_id": work["id"],
                        "title": work.get("title", "Unknown Title"),
                        "authors": [author["author"]["display_name"] for author in work.get("authorships", [])],
                        "abstract": work.get("abstract_inverted_index", {}),
                        "published_date": work.get("publication_date", "Unknown"),
                        "venue": work.get("primary_location", {}).get("source", {}).get("display_name", "Unknown Venue"),
                        "doi": work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
                        "citation_count": work.get("cited_by_count", 0),
                        "concepts": [concept["display_name"] for concept in work.get("concepts", [])[:5]]
                    }

            except Exception as e:
                tracer.current_span().set_tag("error", str(e))
                # Return minimal info on error
                return {
                    "openalex_id": paper_id,
                    "title": f"Error fetching paper {paper_id}",
                    "authors": [],
                    "abstract": "",
                    "published_date": "Unknown",
                    "venue": "Unknown",
                    "doi": None,
                    "citation_count": 0,
                    "concepts": [],
                    "error": str(e)
                }

    def _fetch_citations(self, paper_id: str, max_depth: int) -> List[str]:
        """Fetch papers that cite this paper using OpenAlex."""
        with tracer.trace("crawler.fetch_citations", service="hackmit-openalex"):
            try:
                # Get papers that cite this one
                url = f"https://api.openalex.org/works?filter=cites:{paper_id}&per-page=50"

                with httpx.Client() as client:
                    response = client.get(url, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()

                    citations = []
                    for work in data.get("results", []):
                        citations.append(work["id"])
                        if len(citations) >= max_depth * 10:  # Limit based on depth
                            break

                    return citations

            except Exception as e:
                tracer.current_span().set_tag("error", str(e))
                return []

    def _fetch_references(self, paper_id: str, max_depth: int) -> List[str]:
        """Fetch papers that this paper references using OpenAlex."""
        with tracer.trace("crawler.fetch_references", service="hackmit-openalex"):
            try:
                # Get the paper's references
                url = f"https://api.openalex.org/works/{paper_id}"

                with httpx.Client() as client:
                    response = client.get(url, timeout=10.0)
                    response.raise_for_status()
                    data = response.json()

                    references = []
                    for ref in data.get("referenced_works", []):
                        references.append(ref)
                        if len(references) >= max_depth * 15:  # Limit based on depth
                            break

                    return references

            except Exception as e:
                tracer.current_span().set_tag("error", str(e))
                return []


@ray.remote
def search_openalex_papers(query: str, max_results: int = 10, filter_params: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Search OpenAlex for papers matching the query.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        filter_params: Additional filter parameters for OpenAlex API

    Returns:
        List of paper dictionaries
    """
    with tracer.trace("crawler.openalex_search", service="hackmit-crawler"):
        tracer.current_span().set_tag("query", query)
        tracer.current_span().set_tag("max_results", max_results)

        try:
            # Build OpenAlex search URL
            url = f"https://api.openalex.org/works?search={query}&per-page={min(max_results, 200)}"

            # Add filters if provided
            if filter_params:
                for key, value in filter_params.items():
                    url += f"&filter={key}:{value}"

            with httpx.Client() as client:
                response = client.get(url, timeout=15.0)
                response.raise_for_status()
                data = response.json()

                results = []
                for work in data.get("results", []):
                    # Process abstract from inverted index if available
                    abstract_text = ""
                    if work.get("abstract_inverted_index"):
                        # Reconstruct abstract from inverted index (simplified)
                        abstract_words = [""] * 200  # Assume max 200 words
                        for word, positions in work["abstract_inverted_index"].items():
                            for pos in positions:
                                if pos < len(abstract_words):
                                    abstract_words[pos] = word
                        abstract_text = " ".join([w for w in abstract_words if w]).strip()

                    results.append({
                        "openalex_id": work["id"],
                        "title": work.get("title", "Unknown Title"),
                        "authors": [author["author"]["display_name"] for author in work.get("authorships", [])],
                        "abstract": abstract_text,
                        "published_date": work.get("publication_date", "Unknown"),
                        "venue": work.get("primary_location", {}).get("source", {}).get("display_name", "Unknown Venue"),
                        "doi": work.get("doi", "").replace("https://doi.org/", "") if work.get("doi") else None,
                        "citation_count": work.get("cited_by_count", 0),
                        "concepts": [concept["display_name"] for concept in work.get("concepts", [])[:5]],
                        "open_access": work.get("open_access", {}).get("is_oa", False)
                    })

                tracer.current_span().set_tag("results_found", len(results))
                return results

        except Exception as e:
            tracer.current_span().set_tag("error", str(e))
            return []


@ray.remote
def determine_seed_papers(user_query: str, search_strategy: str = "latest") -> List[str]:
    """
    Front Agent: Determine seed papers based on user query.

    Args:
        user_query: User's research query
        search_strategy: Strategy for finding seeds ("latest", "highly_cited", "historical")

    Returns:
        List of seed paper IDs
    """
    with tracer.trace("crawler.determine_seeds", service="hackmit-front-agent"):
        tracer.current_span().set_tag("user_query", user_query)
        tracer.current_span().set_tag("search_strategy", search_strategy)

        # Simulate LLM processing to determine search terms
        time.sleep(0.3)

        # Mock seed determination logic (replace with actual LLM implementation)
        if "compare" in user_query.lower():
            # Multiple topics comparison - need multiple seeds
            seeds = ["seed_paper_1", "seed_paper_2"]
        elif "history" in user_query.lower():
            # Historical analysis - need older, highly cited papers
            seeds = ["historical_seed_1"]
        else:
            # State of the art - need recent papers
            seeds = ["recent_seed_1"]

        tracer.current_span().set_tag("seeds_determined", len(seeds))
        return seeds


@ray.remote
def build_citation_graph(seed_papers: List[str], max_radius: int = 2) -> Dict[str, Any]:
    """
    Build a citation graph starting from seed papers.

    Args:
        seed_papers: List of seed paper IDs
        max_radius: Maximum distance from seed papers to explore

    Returns:
        Citation graph as adjacency list with paper metadata
    """
    with tracer.trace("crawler.build_graph", service="hackmit-crawler"):
        tracer.current_span().set_tag("seed_count", len(seed_papers))
        tracer.current_span().set_tag("max_radius", max_radius)

        # Create crawler actors
        num_crawlers = min(len(seed_papers), 4)  # Limit concurrent crawlers
        crawlers = [PaperCrawler.remote() for _ in range(num_crawlers)]

        # Distribute seed papers across crawlers
        crawl_tasks = []
        for i, seed in enumerate(seed_papers):
            crawler = crawlers[i % num_crawlers]
            task = crawler.crawl_paper_citations.remote(seed, max_radius)
            crawl_tasks.append(task)

        # Wait for all crawling to complete
        crawl_results = ray.get(crawl_tasks)

        # Build unified graph structure
        graph = {
            "nodes": {},
            "edges": [],
            "metadata": {
                "seed_papers": seed_papers,
                "build_timestamp": time.time(),
                "total_papers": 0
            }
        }

        for result in crawl_results:
            paper_id = result["paper_id"]

            # Add paper as node
            graph["nodes"][paper_id] = result["paper_info"]

            # Add citation edges
            for citation in result["citations"]:
                graph["edges"].append({
                    "from": citation,
                    "to": paper_id,
                    "type": "cites"
                })

            # Add reference edges
            for reference in result["references"]:
                graph["edges"].append({
                    "from": paper_id,
                    "to": reference,
                    "type": "references"
                })

        graph["metadata"]["total_papers"] = len(graph["nodes"])
        tracer.current_span().set_tag("total_papers", graph["metadata"]["total_papers"])

        return graph