import requests
import json
import math
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    title: str
    abstract: str
    year: int
    citation_count: int
    venue: Optional[str]
    authors: List[str]
    openalex_id: str
    doi: Optional[str]
    url: Optional[str]
    score: float = 0.0
    topic_group: str = ""

@dataclass
class SeedResult:
    topic: str
    title: str
    year: int
    citations: int
    openalex_id: str
    url: Optional[str]
    doi: Optional[str]
    abstract: str

class OpenAlexClient:
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: Optional[str] = None):
        self.session = requests.Session()
        if email:
            self.session.headers.update({"User-Agent": f"SeedPaperRetrieval/1.0 ({email})"})
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second max
    
    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def search_papers(self, query: str, per_page: int = 50, sort: str = "cited_by_count:desc") -> List[Dict]:
        self._rate_limit()
        
        url = f"{self.BASE_URL}/works"
        params = {
            "search": query,
            "per-page": min(per_page, 200),  # OpenAlex max is 200
            "sort": sort,
            "filter": "type:article,publication_year:>2015"  # Focus on recent articles
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.RequestException as e:
            logger.error(f"Error fetching papers for query '{query}': {e}")
            return []

class LLMQueryProcessor:    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
    
    def extract_key_concepts(self, query: str) -> List[str]:
        prompt = f"""
        You are helping extract search terms for academic paper retrieval. Given a research query, 
        extract 2-3 distinct key concept groups that would be good for searching academic databases.
        
        Query: "{query}"
        
        Return ONLY a JSON array of strings, where each string is a search term group.
        Focus on:
        - Core technical concepts
        - Specific methodologies mentioned
        - Application domains
        
        Example output: ["reinforcement learning control robotics", "model predictive control", "robotic control theory"]
        """
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"].strip()
            # Try to parse as JSON
            try:
                concepts = json.loads(content)
                return concepts if isinstance(concepts, list) else [content]
            except json.JSONDecodeError:
                # Fallback: split by lines or commas
                return [line.strip().strip('"') for line in content.split('\n') if line.strip()]
                
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            # Fallback: simple keyword extraction
            return [query.lower()]

class VenueScorer:
    
    # High-impact venues by field (add more as needed)
    TOP_VENUES = {
        "ml": ["nature", "science", "nips", "neurips", "icml", "iclr", "aaai", "ijcai"],
        "robotics": ["icra", "iros", "ijrr", "rss", "science robotics"],
        "control": ["automatica", "ieee transactions on automatic control", "cdc", "acc"],
        "ai": ["artificial intelligence", "jair", "aamas", "aaai", "ijcai"],
        "general": ["nature", "science", "pnas", "cell"]
    }
    
    @classmethod
    def score_venue(cls, venue: Optional[str]) -> float:
        if not venue:
            return 0.0
        
        venue_lower = venue.lower()
        
        # Check against known top venues
        for field_venues in cls.TOP_VENUES.values():
            for top_venue in field_venues:
                if top_venue in venue_lower:
                    return 1.0
        
        # Partial matches for IEEE/ACM/Springer journals
        if any(prefix in venue_lower for prefix in ["ieee", "acm", "springer"]):
            return 0.7
        
        # Conference proceedings
        if any(term in venue_lower for term in ["conference", "proceedings", "workshop"]):
            return 0.5
        
        return 0.3  # Default for other venues

class SeedPaperRetriever:    
    def __init__(self, email: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.openalex = OpenAlexClient(email)
        self.llm = LLMQueryProcessor(openai_api_key)
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

        # Step 1: Extract key concepts
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
            logger.info(f"  {i+1}. {paper.title} (Score: {paper.score:.3f}, Citations: {paper.citation_count}, Year: {paper.year})")

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
