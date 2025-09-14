#!/usr/bin/env python3
"""
Test version of front_agent.py that works without Claude API for initial testing.
Uses mock concept extraction to test the OpenAlex functionality.
"""

import sys
import os

# Add the backend directory to Python path so we can import the module
sys.path.insert(0, '/Users/arjuncaputo/hackmit25/backend/app/workloads')

# Now we can import the classes
try:
    from front_agent import OpenAlexClient, Paper, VenueScorer
except ImportError:
    # For when this is imported from the test server
    sys.path.insert(0, '/Users/arjuncaputo/hackmit25')
    from backend.app.workloads.front_agent import OpenAlexClient, Paper, VenueScorer
import requests
import json
import math
import time
import logging
import datetime
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLMQueryProcessor:
    """Mock version that doesn't require Claude API key"""

    def __init__(self, api_key: Optional[str] = None):
        # We don't need the API key for testing
        pass

    def extract_key_concepts(self, query: str) -> List[str]:
        """Extract concepts without using LLM - for testing purposes"""
        logger.info(f"Mock concept extraction for query: {query}")

        # Simple rule-based concept extraction for common queries
        query_lower = query.lower()

        if 'machine learning' in query_lower or 'ml' in query_lower:
            concepts = [
                "machine learning algorithms",
                "deep learning neural networks",
                "supervised learning classification"
            ]
        elif 'transformer' in query_lower or 'attention' in query_lower:
            concepts = [
                "transformer architecture attention mechanism",
                "BERT language model pre-training",
                "GPT generative pre-training"
            ]
        elif 'computer vision' in query_lower or 'image' in query_lower:
            concepts = [
                "computer vision deep learning",
                "convolutional neural networks CNN",
                "image classification recognition"
            ]
        elif 'nlp' in query_lower or 'natural language' in query_lower:
            concepts = [
                "natural language processing NLP",
                "text classification sentiment analysis",
                "language models transformers"
            ]
        elif 'robotics' in query_lower or 'robot' in query_lower:
            concepts = [
                "robotics control systems",
                "robot motion planning",
                "autonomous robots navigation"
            ]
        elif 'reinforcement learning' in query_lower or 'rl' in query_lower:
            concepts = [
                "reinforcement learning algorithms",
                "policy gradient methods",
                "deep reinforcement learning DRL"
            ]
        else:
            # Generic fallback
            words = query_lower.split()
            if len(words) > 2:
                concepts = [
                    " ".join(words[:2]),
                    " ".join(words[1:3]) if len(words) > 2 else query,
                    f"{query} applications"
                ]
            else:
                concepts = [query, f"{query} methods", f"{query} algorithms"]

        logger.info(f"Extracted concepts: {concepts}")
        return concepts[:3]  # Limit to 3 concepts

class TestSeedPaperRetriever:
    """Test version of SeedPaperRetriever using mock LLM"""

    def __init__(self, email: Optional[str] = None):
        self.openalex = OpenAlexClient(email)
        self.llm = MockLLMQueryProcessor()  # Use mock version
        self.venue_scorer = VenueScorer()

    def _reconstruct_abstract(self, inverted_index: Optional[Dict]) -> str:
        if not inverted_index:
            return ""
        positions = [(pos, word) for word, indices in inverted_index.items() for pos in indices]
        positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in positions)

    def _parse_paper(self, raw_paper: Dict, topic_group: str = "") -> Optional[Paper]:
        try:
            # Title
            title = raw_paper.get("title", "").strip()

            # Abstract (reconstructed from inverted index)
            abstract = self._reconstruct_abstract(raw_paper.get("abstract_inverted_index"))

            # Year
            year = raw_paper.get("publication_year", 0)

            # Citation count
            citation_count = raw_paper.get("cited_by_count", 0)

            # OpenAlex ID
            openalex_id = raw_paper.get("id", "").split("/")[-1] if raw_paper.get("id") else ""

            # Venue
            venue = None
            if raw_paper.get("host_venue"):
                venue = raw_paper["host_venue"].get("display_name")

            # Authors
            authors = [
                a["author"]["display_name"]
                for a in raw_paper.get("authorships", [])
                if a.get("author", {}).get("display_name")
            ]

            # DOI and URL
            doi = raw_paper.get("doi")
            url = f"https://doi.org/{doi}" if doi else raw_paper.get("id")

            # Skip low-quality entries
            if not title or citation_count < 5:
                return None

            return Paper(
                title=title,
                abstract=abstract,
                year=year,
                citation_count=citation_count,
                venue=venue,
                authors=authors,
                openalex_id=openalex_id,
                doi=doi,
                url=url,
                topic_group=topic_group
            )

        except Exception as e:
            logger.warning(f"Error parsing paper: {e}")
            return None

    def _calculate_paper_score(self, paper: Paper) -> float:
        current_year = datetime.datetime.now().year

        # Citation score (log scale to prevent dominance by mega-cited papers)
        citation_score = math.log(paper.citation_count + 1)

        # Recency score (sigmoid function favoring recent but not too recent)
        years_ago = current_year - paper.year
        recency_score = 1 / (1 + math.exp(years_ago - 5))  # Peak at ~5 years ago

        # Venue score
        venue_score = self.venue_scorer.score_venue(paper.venue)

        # Weighted combination
        w_citation = 0.5
        w_recency = 0.3
        w_venue = 0.2

        total_score = (w_citation * citation_score +
                      w_recency * recency_score +
                      w_venue * venue_score)

        return total_score

    def _fetch_and_rank_papers(self, concept: str) -> List[Paper]:
        logger.info(f"Fetching papers for concept: {concept}")

        # Fetch papers from OpenAlex
        raw_papers = self.openalex.search_papers(concept, per_page=50)

        # Parse and filter papers
        papers = []
        for raw_paper in raw_papers:
            paper = self._parse_paper(raw_paper, concept)
            if paper:
                paper.score = self._calculate_paper_score(paper)
                papers.append(paper)

        # Sort by score
        papers.sort(key=lambda p: p.score, reverse=True)

        logger.info(f"Found {len(papers)} valid papers for '{concept}'")
        return papers

    def retrieve_seed_papers(self, query: str) -> List[str]:
        logger.info(f"Processing query: {query}")

        # Step 1: Extract key concepts (using mock LLM)
        concepts = self.llm.extract_key_concepts(query)
        logger.info(f"Extracted concepts: {concepts}")

        # Step 2: Fetch and rank papers for each concept (sequentially)
        all_papers: List[Paper] = []
        for concept in concepts:
            try:
                papers = self._fetch_and_rank_papers(concept)
                all_papers.extend(papers)
                logger.info(f"Added {len(papers)} papers from concept '{concept}', total papers: {len(all_papers)}")
            except Exception as e:
                logger.error(f"Error processing concept '{concept}': {e}")

        # Step 3: Select the top 2 scoring papers overall
        if not all_papers:
            logger.warning("No papers found for query.")
            return []

        logger.info(f"Sorting {len(all_papers)} papers by score")
        all_papers.sort(key=lambda p: p.score, reverse=True)
        top_papers = all_papers[:2]

        logger.info(f"Top scoring papers:")
        for i, paper in enumerate(top_papers):
            logger.info(f"  {i+1}. {paper.title}")
            logger.info(f"      Score: {paper.score:.3f}, Citations: {paper.citation_count}, Year: {paper.year}")
            logger.info(f"      Venue: {paper.venue}")
            logger.info(f"      Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            if paper.abstract:
                logger.info(f"      Abstract: {paper.abstract[:150]}...")
            logger.info("")

        dois = []
        for paper in top_papers:
            if paper.doi:
                dois.append(paper.doi)
                logger.info(f"Selected seed: {paper.title} ({paper.doi})")
            else:
                # Fallback to OpenAlex ID if DOI missing
                dois.append(f"openalex:{paper.openalex_id}")
                logger.info(f"Selected seed (no DOI): {paper.title} (openalex:{paper.openalex_id})")

        logger.info(f"Retrieved {len(dois)} seed DOIs")
        return dois

def test_front_agent():
    """Test the front agent functionality"""
    print("🧪 Testing front_agent.py functionality...")
    print("=" * 60)

    # Test queries
    test_queries = [
        "transformer neural networks attention mechanisms",
        "machine learning classification algorithms",
        "computer vision object detection",
        "natural language processing BERT",
        "reinforcement learning robotics"
    ]

    # Initialize retriever
    retriever = TestSeedPaperRetriever(email="test@example.com")

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: Query = '{query}'")
        print("-" * 50)

        try:
            seed_dois = retriever.retrieve_seed_papers(query)

            if seed_dois:
                print(f"✅ SUCCESS: Found {len(seed_dois)} seed papers")
                for j, doi in enumerate(seed_dois, 1):
                    print(f"   {j}. {doi}")
            else:
                print("❌ No papers found")

        except Exception as e:
            print(f"❌ ERROR: {e}")

        if i < len(test_queries):
            print("\nWaiting 2 seconds before next query (rate limiting)...")
            time.sleep(2)

    print("\n" + "=" * 60)
    print("🎉 Front agent testing completed!")

if __name__ == "__main__":
    test_front_agent()