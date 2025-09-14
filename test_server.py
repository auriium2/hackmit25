#!/usr/bin/env python3
"""
Python test server to verify API integration.
Equivalent functionality to the Node.js test-server.js with front_agent integration.
"""

import json
import random
import time
import threading
import sys
import os
import uuid
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# Add the backend directory to Python path so we can import front_agent
sys.path.insert(0, '/Users/arjuncaputo/hackmit25/backend/app/workloads')

# Import both real and test front_agent
try:
    # Load environment variables first
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # Try to import the real front agent first
    from front_agent import SeedPaperRetriever
    REAL_FRONT_AGENT_AVAILABLE = True
    print("🎯 Real front agent with Claude LLM integration enabled")
except ImportError as e:
    print(f"⚠️  Real front agent not available: {e}")
    REAL_FRONT_AGENT_AVAILABLE = False

    # Fallback to test version
    try:
        from test_front_agent import TestSeedPaperRetriever
        FRONT_AGENT_AVAILABLE = True
        print("🔬 Test front agent integration enabled")
    except ImportError as e:
        print(f"⚠️  No front agent available: {e}")
        FRONT_AGENT_AVAILABLE = False

# Track processing jobs
jobs = {}

# Session Management System
class SessionManager:
    """Manages API sessions with queryIDs, user tracking, and session persistence."""

    def __init__(self):
        self.sessions = {}  # queryID -> session_data
        self.user_sessions = {}  # user_id -> [queryIDs]
        self.session_timeout = timedelta(hours=24)  # 24 hour session timeout

    def create_session(self, user_id=None):
        """Create a new session with unique queryID."""
        query_id = str(uuid.uuid4())
        timestamp = datetime.now()

        session_data = {
            'queryID': query_id,
            'user_id': user_id or 'anonymous',
            'created_at': timestamp,
            'last_accessed': timestamp,
            'query_text': None,
            'seed_papers': [],
            'processing_status': 'created',
            'metrics': {},
            'graph_data': None,
            'papers_analyzed': [],
            'benchmarks': [],
            'analysis_results': {},
            'session_active': True
        }

        self.sessions[query_id] = session_data

        # Track user sessions
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(query_id)

        print(f"🔑 SESSION: Created new session {query_id} for user {user_id or 'anonymous'}")
        return query_id

    def get_session(self, query_id):
        """Get session data by queryID."""
        if query_id not in self.sessions:
            return None

        session = self.sessions[query_id]

        # Check if session has expired
        if datetime.now() - session['last_accessed'] > self.session_timeout:
            self.expire_session(query_id)
            return None

        # Update last accessed time
        session['last_accessed'] = datetime.now()
        return session

    def update_session(self, query_id, **kwargs):
        """Update session data."""
        if query_id in self.sessions:
            self.sessions[query_id].update(kwargs)
            self.sessions[query_id]['last_accessed'] = datetime.now()
            return True
        return False

    def expire_session(self, query_id):
        """Expire and clean up a session."""
        if query_id in self.sessions:
            session = self.sessions[query_id]
            user_id = session.get('user_id')

            # Remove from user sessions
            if user_id and user_id in self.user_sessions:
                if query_id in self.user_sessions[user_id]:
                    self.user_sessions[user_id].remove(query_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]

            # Remove session
            del self.sessions[query_id]
            print(f"⏰ SESSION: Expired session {query_id}")
            return True
        return False

    def get_user_sessions(self, user_id):
        """Get all active sessions for a user."""
        if user_id not in self.user_sessions:
            return []

        active_sessions = []
        for query_id in self.user_sessions[user_id][:]:  # Copy to avoid modification during iteration
            session = self.get_session(query_id)
            if session:
                active_sessions.append(session)

        return active_sessions

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now()
        expired = []

        for query_id, session in self.sessions.items():
            if now - session['last_accessed'] > self.session_timeout:
                expired.append(query_id)

        for query_id in expired:
            self.expire_session(query_id)

        if expired:
            print(f"🧹 SESSION: Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def get_session_stats(self):
        """Get session statistics."""
        total_sessions = len(self.sessions)
        active_users = len(self.user_sessions)

        # Session ages
        now = datetime.now()
        session_ages = []
        for session in self.sessions.values():
            age = (now - session['created_at']).total_seconds() / 3600  # hours
            session_ages.append(age)

        stats = {
            'total_active_sessions': total_sessions,
            'unique_users': active_users,
            'average_session_age_hours': sum(session_ages) / len(session_ages) if session_ages else 0,
            'oldest_session_hours': max(session_ages) if session_ages else 0,
            'newest_session_hours': min(session_ages) if session_ages else 0
        }

        return stats

# Initialize session manager
session_manager = SessionManager()

# Start background thread for session cleanup
def session_cleanup_worker():
    """Background worker to clean up expired sessions."""
    while True:
        time.sleep(3600)  # Run every hour
        session_manager.cleanup_expired_sessions()

cleanup_thread = threading.Thread(target=session_cleanup_worker, daemon=True)
cleanup_thread.start()

# Initialize front agent retriever if available (prefer real over test)
if REAL_FRONT_AGENT_AVAILABLE:
    retriever = SeedPaperRetriever(email="hackmit2025@example.com")
    USING_REAL_LLM = True
    print("🚀 Using REAL Claude LLM for concept extraction")
elif FRONT_AGENT_AVAILABLE:
    retriever = TestSeedPaperRetriever(email="hackmit2025@example.com")
    USING_REAL_LLM = False
    print("🔬 Using MOCK LLM for concept extraction")
else:
    retriever = None
    USING_REAL_LLM = False

class CORSHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        """Set CORS headers for all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        if path == '/health':
            self._handle_health()
        elif path == '/check_status':
            self._handle_check_status(query_params)
        elif path == '/pull_final_graph':
            self._handle_pull_final_graph(query_params)
        elif path == '/get_details':
            self._handle_get_details(query_params)
        elif path == '/analyze_paper':
            self._handle_analyze_paper(query_params)
        elif path == '/get_benchmarks':
            self._handle_get_benchmarks(query_params)
        elif path == '/session':
            self._handle_get_session(query_params)
        elif path == '/sessions':
            self._handle_get_user_sessions(query_params)
        elif path == '/session/stats':
            self._handle_session_stats()
        else:
            self._send_404()

    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/query':
            self._handle_query()
        elif path == '/session/create':
            self._handle_create_session()
        elif path == '/session/update':
            self._handle_update_session()
        elif path == '/session/expire':
            self._handle_expire_session()
        else:
            self._send_404()

    def _handle_health(self):
        """Health check endpoint"""
        response_data = {'status': 'healthy'}
        self._send_json_response(200, response_data)

    def _handle_query(self):
        """Handle query submission with session management"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            query = data.get('query', '')
            user_id = data.get('user_id')  # Optional user ID
            query_id = data.get('queryID')  # Optional existing queryID

            # Create new session or use existing one
            if query_id:
                session = session_manager.get_session(query_id)
                if not session:
                    # Session expired or invalid, create new one
                    query_id = session_manager.create_session(user_id)
                    session = session_manager.get_session(query_id)
                else:
                    print(f'🔄 USING EXISTING SESSION: {query_id}')
            else:
                query_id = session_manager.create_session(user_id)
                session = session_manager.get_session(query_id)

            # Legacy systemid for backwards compatibility
            systemid = f'test-{int(time.time() * 1000)}'

            print(f'🎯 LIVE BACKEND: Query processed: {query} | QueryID: {query_id} | SystemID: {systemid} | STATUS: ACTIVE!')

            # Update session with query details
            session_manager.update_session(query_id,
                query_text=query,
                processing_status='processing',
                systemid=systemid  # For backwards compatibility
            )

            # Trigger front_agent to find seed papers
            seed_papers = []
            if (REAL_FRONT_AGENT_AVAILABLE or FRONT_AGENT_AVAILABLE) and retriever:
                try:
                    llm_type = "REAL Claude LLM" if USING_REAL_LLM else "MOCK LLM"
                    print(f'🔬 FRONT AGENT ({llm_type}): Starting seed paper retrieval for: {query}')
                    seed_dois = retriever.retrieve_seed_papers(query)
                    seed_papers = seed_dois
                    print(f'✅ FRONT AGENT ({llm_type}): Found {len(seed_papers)} seed papers')
                    for i, doi in enumerate(seed_papers, 1):
                        print(f'   📄 Seed {i}: {doi}')

                    # Update session with seed papers
                    session_manager.update_session(query_id, seed_papers=seed_papers)

                except Exception as e:
                    print(f'❌ FRONT AGENT ERROR: {e}')
                    seed_papers = []
            else:
                print('⚠️  FRONT AGENT: Not available, using fallback')
                seed_papers = []

            # Start processing simulation in background
            start_processing_simulation(systemid, query, seed_papers, query_id)

            response_data = {
                'queryID': query_id,
                'systemid': systemid,  # For backwards compatibility
                'status': 'ok',
                'seed_papers_count': len(seed_papers)
            }
            self._send_json_response(200, response_data)

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})

    def _handle_check_status(self, query_params):
        """Check processing status"""
        systemid = query_params.get('systemid', [None])[0]

        if not systemid or systemid not in jobs:
            response_data = {
                'systemid': systemid,
                'status': 'fail: job not found'
            }
            self._send_json_response(200, response_data)
            return

        job = jobs[systemid]
        print(f'🔍 Status check for {systemid}: {job["status"]}')

        response_data = {
            'systemid': systemid,
            'status': job['status']
        }
        self._send_json_response(200, response_data)

    def _handle_pull_final_graph(self, query_params):
        """Get final graph data"""
        systemid = query_params.get('systemid', [None])[0]

        # Generate dynamic graph based on the original query
        query = 'research'
        if systemid in jobs:
            query = jobs[systemid]['query']

        graph_data = generate_mock_graph(query, systemid)
        self._send_json_response(200, graph_data)

    def _handle_get_details(self, query_params):
        """Get paper details"""
        paper_id = query_params.get('openalexid', [None])[0]
        print(f'📄 Paper details requested for: {paper_id}')

        # Mock OpenAlex data based on paper ID
        mock_paper_data = {
            'paper1': {
                'openalex_id': 'paper1',
                'title': 'Attention Is All You Need',
                'abstract': 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
                'year': 2017,
                'citations': 70000,
                'doi': '10.5555/3295222.3295349',
                'url': 'https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html',
                'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 'Jakob Uszkoreit', 'Llion Jones', 'Aidan N. Gomez', 'Lukasz Kaiser', 'Illia Polosukhin'],
                'venue': 'Neural Information Processing Systems'
            },
            'paper2': {
                'openalex_id': 'paper2',
                'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
                'abstract': 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.',
                'year': 2018,
                'citations': 45000,
                'doi': '10.18653/v1/N19-1423',
                'url': 'https://aclanthology.org/N19-1423/',
                'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee', 'Kristina Toutanova'],
                'venue': 'NAACL-HLT'
            },
            'paper3': {
                'openalex_id': 'paper3',
                'title': 'Language Models are Few-Shot Learners',
                'abstract': 'Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions.',
                'year': 2020,
                'citations': 25000,
                'doi': '10.5555/3495724.3495883',
                'url': 'https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html',
                'authors': ['Tom B. Brown', 'Benjamin Mann', 'Nick Ryder', 'Melanie Subbiah', 'Jared Kaplan'],
                'venue': 'Neural Information Processing Systems'
            }
        }

        paper_data = mock_paper_data.get(paper_id, {
            'openalex_id': paper_id,
            'title': 'Unknown Paper',
            'abstract': 'Paper details not found in test data.',
            'year': 2024,
            'citations': 0,
            'doi': None,
            'url': None,
            'authors': ['Unknown Author'],
            'venue': 'Unknown Venue'
        })

        self._send_json_response(200, paper_data)

    def _handle_analyze_paper(self, query_params):
        """Analyze paper endpoint"""
        paper_id = query_params.get('paper_id', [None])[0]
        print(f'🚀 BACKEND LIVE: Full paper analysis for {paper_id} - API WORKING!')

        # Different analysis data per paper
        analysis_data = {
            'paper1': {
                'title': 'Attention Is All You Need',
                'abstract': 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
                'contributions': [
                    'Introduced the Transformer architecture',
                    'Eliminated recurrence in favor of attention',
                    'Achieved state-of-the-art translation results',
                    'Enabled parallel training'
                ],
                'methodology': ['Multi-head attention mechanism', 'Positional encoding', 'Layer normalization', 'Residual connections'],
                'impact_score': 9.8,
                'relevance_score': 9.5
            },
            'paper2': {
                'title': 'BERT: Pre-training Deep Bidirectional Representations',
                'abstract': 'We introduce BERT, designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.',
                'contributions': [
                    'Introduced bidirectional pre-training',
                    'Achieved new SOTA on GLUE benchmark',
                    'Demonstrated transfer learning effectiveness',
                    'Influenced numerous follow-up models'
                ],
                'methodology': ['Masked language modeling', 'Next sentence prediction', 'Bidirectional encoding', 'Fine-tuning approach'],
                'impact_score': 9.6,
                'relevance_score': 9.2
            },
            'paper3': {
                'title': 'GPT-3: Language Models are Few-Shot Learners',
                'abstract': 'We train GPT-3, an autoregressive language model with 175 billion parameters, demonstrating strong few-shot learning capabilities.',
                'contributions': [
                    'Demonstrated emergent few-shot learning',
                    'Scaled to 175B parameters',
                    'Showed in-context learning capabilities',
                    'Influenced large language model development'
                ],
                'methodology': ['Autoregressive generation', 'Few-shot prompting', 'Massive scale training', 'In-context learning'],
                'impact_score': 9.4,
                'relevance_score': 8.9
            }
        }

        paper_data = analysis_data.get(paper_id, analysis_data['paper1'])

        mock_analysis = {
            'paper_id': paper_id,
            'openalex_data': {
                'openalex_id': paper_id,
                'title': paper_data['title'],
                'abstract': paper_data['abstract'],
                'year': 2017 if paper_id == 'paper1' else 2018 if paper_id == 'paper2' else 2020,
                'citations': 70000 if paper_id == 'paper1' else 45000 if paper_id == 'paper2' else 25000,
                'doi': f'10.5555/{paper_id}.example',
                'url': f'https://proceedings.neurips.cc/paper/{paper_id}/',
                'authors': ['Vaswani et al.'] if paper_id == 'paper1' else ['Devlin et al.'] if paper_id == 'paper2' else ['Brown et al.'],
                'venue': 'NeurIPS'
            },
            'benchmark_metrics': {
                'accuracy': 92.5 if paper_id == 'paper1' else 88.9 if paper_id == 'paper2' else 85.2,
                'f1_score': 89.3 if paper_id == 'paper1' else 91.1 if paper_id == 'paper2' else 87.4,
                'bleu_score': 34.2 if paper_id == 'paper1' else None if paper_id == 'paper2' else 28.9,
                'perplexity': 1.24 if paper_id == 'paper1' else 1.18 if paper_id == 'paper2' else 1.32,
                'inference_time': 0.045 if paper_id == 'paper1' else 0.052 if paper_id == 'paper2' else 0.089,
                'model_size': 65.0 if paper_id == 'paper1' else 340.0 if paper_id == 'paper2' else 175000.0,
                'dataset': 'WMT 2014 EN-DE' if paper_id == 'paper1' else 'GLUE Benchmark' if paper_id == 'paper2' else 'Common Crawl',
                'benchmark_suite': 'Machine Translation' if paper_id == 'paper1' else 'GLUE' if paper_id == 'paper2' else 'Few-shot Learning',
                'evaluation_date': '2017-06-12' if paper_id == 'paper1' else '2018-10-11' if paper_id == 'paper2' else '2020-05-28'
            },
            'analysis_summary': {
                'key_contributions': paper_data['contributions'],
                'methodology': paper_data['methodology'],
                'strengths': (['Highly parallelizable', 'Better long-range dependencies', 'Strong empirical results', 'Computational efficiency'] if paper_id == 'paper1' else
                           ['Bidirectional context', 'Transfer learning', 'SOTA results', 'Widely applicable'] if paper_id == 'paper2' else
                           ['Few-shot learning', 'Massive scale', 'Emergent abilities', 'General purpose']),
                'limitations': (['Quadratic complexity', 'Requires large data', 'Memory intensive'] if paper_id == 'paper1' else
                             ['Requires fine-tuning', 'Computationally expensive', 'Limited context length'] if paper_id == 'paper2' else
                             ['Enormous compute requirements', 'Hallucination issues', 'Limited reasoning']),
                'impact_score': paper_data['impact_score'],
                'relevance_score': paper_data['relevance_score']
            },
            'processing_metadata': {
                'extraction_confidence': 0.96,
                'last_updated': '2024-01-15T10:30:00Z',
                'data_sources': ['OpenAlex', 'Semantic Scholar', 'ArXiv']
            }
        }

        self._send_json_response(200, mock_analysis)

    def _handle_get_benchmarks(self, query_params):
        """Get benchmark metrics"""
        paper_id = query_params.get('paper_id', [None])[0]
        print(f'⚡ REAL-TIME: Benchmark metrics loaded for {paper_id} - BACKEND RESPONDING!')

        mock_metrics = {
            'accuracy': 92.5 if paper_id == 'paper1' else 88.9 if paper_id == 'paper2' else 85.2,
            'f1_score': 89.3 if paper_id == 'paper1' else 91.1 if paper_id == 'paper2' else 87.4,
            'bleu_score': 34.2 if paper_id == 'paper1' else None if paper_id == 'paper2' else 28.9,
            'rouge_score': 58.7 if paper_id == 'paper1' else 62.3 if paper_id == 'paper2' else 55.1,
            'perplexity': 1.24 if paper_id == 'paper1' else 1.18 if paper_id == 'paper2' else 1.32,
            'inference_time': 0.045 if paper_id == 'paper1' else 0.052 if paper_id == 'paper2' else 0.089,
            'model_size': 65.0 if paper_id == 'paper1' else 340.0 if paper_id == 'paper2' else 175000.0,
            'training_time': 3.2 if paper_id == 'paper1' else 12.5 if paper_id == 'paper2' else 1200.0,
            'dataset': 'WMT 2014 EN-DE' if paper_id == 'paper1' else 'GLUE Benchmark' if paper_id == 'paper2' else 'Common Crawl',
            'benchmark_suite': 'Machine Translation' if paper_id == 'paper1' else 'GLUE' if paper_id == 'paper2' else 'Few-shot Learning',
            'evaluation_date': '2017-06-12' if paper_id == 'paper1' else '2018-10-11' if paper_id == 'paper2' else '2020-05-28'
        }

        self._send_json_response(200, mock_metrics)

    def _send_json_response(self, status_code, data):
        """Send JSON response with CORS headers"""
        self.send_response(status_code)
        self._set_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_404(self):
        """Send 404 response"""
        self.send_response(404)
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(b'Not found')

    # Session Management Endpoints
    def _handle_create_session(self):
        """Create new session endpoint"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            user_id = data.get('user_id')

            query_id = session_manager.create_session(user_id)
            session = session_manager.get_session(query_id)

            response_data = {
                'queryID': query_id,
                'status': 'created',
                'user_id': session['user_id'],
                'created_at': session['created_at'].isoformat()
            }
            self._send_json_response(200, response_data)

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_json_response(500, {'error': str(e)})

    def _handle_update_session(self):
        """Update session data endpoint"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            query_id = data.get('queryID')

            if not query_id:
                self._send_json_response(400, {'error': 'queryID required'})
                return

            # Remove queryID from update data
            update_data = {k: v for k, v in data.items() if k != 'queryID'}

            success = session_manager.update_session(query_id, **update_data)

            if success:
                session = session_manager.get_session(query_id)
                response_data = {
                    'queryID': query_id,
                    'status': 'updated',
                    'session_data': {
                        'processing_status': session.get('processing_status'),
                        'last_accessed': session['last_accessed'].isoformat()
                    }
                }
                self._send_json_response(200, response_data)
            else:
                self._send_json_response(404, {'error': 'Session not found'})

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_json_response(500, {'error': str(e)})

    def _handle_expire_session(self):
        """Expire session endpoint"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            query_id = data.get('queryID')

            if not query_id:
                self._send_json_response(400, {'error': 'queryID required'})
                return

            success = session_manager.expire_session(query_id)

            response_data = {
                'queryID': query_id,
                'status': 'expired' if success else 'not_found'
            }
            self._send_json_response(200, response_data)

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_json_response(500, {'error': str(e)})

    def _handle_get_session(self, query_params):
        """Get session data endpoint"""
        query_id = query_params.get('queryID', [None])[0]

        if not query_id:
            self._send_json_response(400, {'error': 'queryID required'})
            return

        session = session_manager.get_session(query_id)

        if session:
            # Serialize datetime objects
            session_data = session.copy()
            session_data['created_at'] = session_data['created_at'].isoformat()
            session_data['last_accessed'] = session_data['last_accessed'].isoformat()

            response_data = {
                'queryID': query_id,
                'status': 'found',
                'session': session_data
            }
            self._send_json_response(200, response_data)
        else:
            self._send_json_response(404, {'error': 'Session not found or expired'})

    def _handle_get_user_sessions(self, query_params):
        """Get all sessions for a user endpoint"""
        user_id = query_params.get('user_id', [None])[0]

        if not user_id:
            self._send_json_response(400, {'error': 'user_id required'})
            return

        sessions = session_manager.get_user_sessions(user_id)

        # Serialize datetime objects
        serialized_sessions = []
        for session in sessions:
            session_data = session.copy()
            session_data['created_at'] = session_data['created_at'].isoformat()
            session_data['last_accessed'] = session_data['last_accessed'].isoformat()
            serialized_sessions.append(session_data)

        response_data = {
            'user_id': user_id,
            'session_count': len(serialized_sessions),
            'sessions': serialized_sessions
        }
        self._send_json_response(200, response_data)

    def _handle_session_stats(self):
        """Get session statistics endpoint"""
        stats = session_manager.get_session_stats()
        self._send_json_response(200, stats)


def generate_mock_graph(query, systemid):
    """Generate mock graph data based on search query"""
    print(f'📊 Generating dynamic graph for query: {query}')

    def get_topics_for_query(search_query):
        """Get query-specific research topics"""
        lower_query = search_query.lower()

        if 'machine learning' in lower_query or 'ml' in lower_query:
            return [
                {'title': f'Machine Learning Foundations for {search_query}', 'keywords': ['ml', 'supervised-learning', 'algorithms']},
                {'title': f'Deep Learning Approaches in {search_query}', 'keywords': ['deep-learning', 'neural-networks', 'backpropagation']},
                {'title': f'Reinforcement Learning Applications for {search_query}', 'keywords': ['reinforcement-learning', 'policy-gradient', 'reward-function']},
                {'title': f'Unsupervised Learning Methods in {search_query}', 'keywords': ['clustering', 'dimensionality-reduction', 'autoencoder']},
                {'title': f'Meta-Learning and Few-Shot Learning for {search_query}', 'keywords': ['meta-learning', 'few-shot', 'transfer-learning']}
            ]
        elif 'transformer' in lower_query or 'attention' in lower_query:
            return [
                {'title': 'Attention Is All You Need', 'keywords': ['transformer', 'attention', 'self-attention']},
                {'title': 'BERT: Pre-training Deep Bidirectional Transformers', 'keywords': ['bert', 'bidirectional', 'pre-training']},
                {'title': 'GPT: Generative Pre-training of a Language Model', 'keywords': ['gpt', 'generative', 'autoregressive']},
                {'title': 'Vision Transformer: An Image is Worth 16x16 Words', 'keywords': ['vision-transformer', 'image-patches', 'computer-vision']},
                {'title': 'Switch Transformer: Scaling to Trillion Parameter Models', 'keywords': ['switch-transformer', 'mixture-of-experts', 'scaling']}
            ]
        elif 'nlp' in lower_query or 'language' in lower_query:
            return [
                {'title': f'Natural Language Processing for {search_query}', 'keywords': ['nlp', 'tokenization', 'language-model']},
                {'title': f'Sentiment Analysis in {search_query}', 'keywords': ['sentiment-analysis', 'classification', 'opinion-mining']},
                {'title': f'Named Entity Recognition for {search_query}', 'keywords': ['ner', 'entity-extraction', 'sequence-labeling']},
                {'title': f'Question Answering Systems in {search_query}', 'keywords': ['qa', 'reading-comprehension', 'information-retrieval']},
                {'title': f'Machine Translation Approaches for {search_query}', 'keywords': ['machine-translation', 'sequence-to-sequence', 'alignment']}
            ]
        else:
            # Generic topics based on query
            return [
                {'title': f'Novel Approaches to {search_query}', 'keywords': ['methodology', 'innovation', search_query.lower().replace(' ', '-')]},
                {'title': f'Comprehensive Survey of {search_query}', 'keywords': ['survey', 'review', search_query.lower().replace(' ', '-')]},
                {'title': f'Deep Learning Applications in {search_query}', 'keywords': ['deep-learning', 'applications', search_query.lower().replace(' ', '-')]},
                {'title': f'Empirical Studies on {search_query}', 'keywords': ['empirical', 'experimental', search_query.lower().replace(' ', '-')]},
                {'title': f'Future Directions for {search_query} Research', 'keywords': ['future-work', 'research-direction', search_query.lower().replace(' ', '-')]}
            ]

    def get_cluster_from_query(query):
        """Get cluster name from query"""
        lower_query = query.lower()
        if 'nlp' in lower_query or 'language' in lower_query:
            return 'Natural Language Processing'
        if 'vision' in lower_query or 'image' in lower_query:
            return 'Computer Vision'
        if 'ml' in lower_query or 'machine learning' in lower_query:
            return 'Machine Learning'
        if 'ai' in lower_query or 'artificial' in lower_query:
            return 'Artificial Intelligence'
        if 'data' in lower_query:
            return 'Data Science'
        return 'Computer Science'

    topics = get_topics_for_query(query)

    # Generate 8-15 papers for more realistic graph
    num_papers = random.randint(8, 15)
    nodes = []
    edges = []

    # Create main papers from topics
    for i, topic in enumerate(topics):
        year = 2015 + random.randint(0, 9)
        paper_id = f'paper{i + 1}'

        nodes.append({
            'id': paper_id,
            'label': topic['title'],
            'data': {
                'id': paper_id,
                'title': topic['title'],
                'authors': [f'Researcher{i + 1} et al.'],
                'year': year,
                'abstract': f'This paper presents {topic["title"].lower()}, contributing novel insights to the field through innovative methodologies and comprehensive experimental validation.',
                'citations': random.randint(1000, 15000),
                'cluster': get_cluster_from_query(query),
                'confidence': random.randint(80, 100),
                'summary': f'Comprehensive research on {topic["title"].lower()}, providing significant contributions to the {query.lower()} domain.',
                'metrics': {},
                'embedding': []
            }
        })

    # Generate additional papers to reach target number
    for i in range(len(topics), num_papers):
        year = 2010 + random.randint(0, 14)
        paper_id = f'paper{i + 1}'

        nodes.append({
            'id': paper_id,
            'label': f'Advanced {query} Techniques and Applications',
            'data': {
                'id': paper_id,
                'title': f'Advanced {query} Techniques and Applications {i - len(topics) + 1}',
                'authors': [f'Author{i + 1} et al.'],
                'year': year,
                'abstract': f'This work explores advanced techniques in {query.lower()}, presenting novel algorithms and methodologies.',
                'citations': random.randint(200, 8000),
                'cluster': get_cluster_from_query(query),
                'confidence': random.randint(70, 95),
                'summary': f'Research focusing on {query.lower()} with emphasis on practical applications and theoretical foundations.',
                'metrics': {},
                'embedding': []
            }
        })

    # Generate edges (citations) between papers
    for i, node in enumerate(nodes):
        num_connections = random.randint(1, 3)
        potential_targets = [
            other_node for j, other_node in enumerate(nodes)
            if j != i and abs(other_node['data']['year'] - node['data']['year']) <= 4
        ]

        selected_targets = random.sample(potential_targets, min(num_connections, len(potential_targets)))

        for target in selected_targets:
            edge_id = f'citation-{node["id"]}-{target["id"]}'
            reverse_edge_id = f'citation-{target["id"]}-{node["id"]}'

            # Avoid duplicate edges
            if not any(e['id'] == edge_id or e['id'] == reverse_edge_id for e in edges):
                edges.append({
                    'id': edge_id,
                    'source': node['id'],
                    'target': target['id'],
                    'type': 'citation',
                    'data': {
                        'relation': 'cites',
                        'confidence': 0.7 + random.random() * 0.3
                    }
                })

    return {
        'systemid': systemid,
        'graph': {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'query': query,
                'total_papers': len(nodes),
                'processing_time': 15.0 + random.random() * 10,
                'insights': {
                    'dominant_cluster': get_cluster_from_query(query),
                    'average_year': round(sum(n['data']['year'] for n in nodes) / len(nodes)),
                    'total_citations': sum(n['data']['citations'] for n in nodes)
                },
                'seed_papers': [n['id'] for n in nodes[:min(3, len(nodes))]]
            }
        }
    }


def start_processing_simulation(systemid, query, seed_papers=None, query_id=None):
    """Simulate realistic processing with status updates and session tracking"""
    statuses = [
        {'status': 'finding seed papers', 'duration': 3.0},
        {'status': 'building citation graph', 'duration': 4.0},
        {'status': 'analyzing papers with AI agents', 'duration': 5.0},
        {'status': 'generating insights', 'duration': 3.0},
        {'status': 'finalizing graph', 'duration': 2.0},
        {'status': 'done', 'duration': 0}
    ]

    # Initialize job (legacy compatibility)
    jobs[systemid] = {
        'query': query,
        'status': 'started',
        'start_time': time.time(),
        'seed_papers': seed_papers or [],
        'seed_count': len(seed_papers) if seed_papers else 0,
        'query_id': query_id  # Link to session
    }

    print(f'🚀 Starting processing for: {systemid} (Session: {query_id})')

    def progress_to_next_status(step=0):
        """Progress through status updates"""
        if step >= len(statuses):
            return

        current_status = statuses[step]
        status_text = current_status['status']

        # Update legacy job tracking
        if systemid in jobs:
            jobs[systemid]['status'] = status_text
            print(f'⏳ Status update: {systemid} → {status_text}')

        # Update session if available
        if query_id:
            session_manager.update_session(query_id, processing_status=status_text)

        if step < len(statuses) - 1:
            # Schedule next status update
            timer = threading.Timer(current_status['duration'], progress_to_next_status, args=[step + 1])
            timer.start()
        else:
            print(f'✅ Processing complete for: {systemid} (Session: {query_id})')
            # Mark session as completed
            if query_id:
                session_manager.update_session(query_id, processing_status='completed')

    # Start the progression after 1 second
    timer = threading.Timer(1.0, progress_to_next_status, args=[0])
    timer.start()


# Add session management endpoints to CORSHandler class
def add_session_endpoints_to_handler():
    """Add session management endpoint methods to CORSHandler class"""

    def _handle_create_session(self):
        """Create new session endpoint"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            user_id = data.get('user_id')

            query_id = session_manager.create_session(user_id)
            session = session_manager.get_session(query_id)

            response_data = {
                'queryID': query_id,
                'status': 'created',
                'user_id': session['user_id'],
                'created_at': session['created_at'].isoformat()
            }
            self._send_json_response(200, response_data)

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_json_response(500, {'error': str(e)})

    def _handle_update_session(self):
        """Create new session endpoint"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            user_id = data.get('user_id')

            query_id = session_manager.create_session(user_id)
            session = session_manager.get_session(query_id)

            response_data = {
                'queryID': query_id,
                'status': 'created',
                'user_id': session['user_id'],
                'created_at': session['created_at'].isoformat()
            }
            self._send_json_response(200, response_data)

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_json_response(500, {'error': str(e)})

    def _handle_update_session(self):
        """Update session data endpoint"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            query_id = data.get('queryID')

            if not query_id:
                self._send_json_response(400, {'error': 'queryID required'})
                return

            # Remove queryID from update data
            update_data = {k: v for k, v in data.items() if k != 'queryID'}

            success = session_manager.update_session(query_id, **update_data)

            if success:
                session = session_manager.get_session(query_id)
                response_data = {
                    'queryID': query_id,
                    'status': 'updated',
                    'session_data': {
                        'processing_status': session.get('processing_status'),
                        'last_accessed': session['last_accessed'].isoformat()
                    }
                }
                self._send_json_response(200, response_data)
            else:
                self._send_json_response(404, {'error': 'Session not found'})

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_json_response(500, {'error': str(e)})

    def _handle_expire_session(self):
        """Expire session endpoint"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            query_id = data.get('queryID')

            if not query_id:
                self._send_json_response(400, {'error': 'queryID required'})
                return

            success = session_manager.expire_session(query_id)

            response_data = {
                'queryID': query_id,
                'status': 'expired' if success else 'not_found'
            }
            self._send_json_response(200, response_data)

        except json.JSONDecodeError:
            self._send_json_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_json_response(500, {'error': str(e)})

    def _handle_get_session(self, query_params):
        """Get session data endpoint"""
        query_id = query_params.get('queryID', [None])[0]

        if not query_id:
            self._send_json_response(400, {'error': 'queryID required'})
            return

        session = session_manager.get_session(query_id)

        if session:
            # Serialize datetime objects
            session_data = session.copy()
            session_data['created_at'] = session_data['created_at'].isoformat()
            session_data['last_accessed'] = session_data['last_accessed'].isoformat()

            response_data = {
                'queryID': query_id,
                'status': 'found',
                'session': session_data
            }
            self._send_json_response(200, response_data)
        else:
            self._send_json_response(404, {'error': 'Session not found or expired'})

    def _handle_get_user_sessions(self, query_params):
        """Get all sessions for a user endpoint"""
        user_id = query_params.get('user_id', [None])[0]

        if not user_id:
            self._send_json_response(400, {'error': 'user_id required'})
            return

        sessions = session_manager.get_user_sessions(user_id)

        # Serialize datetime objects
        serialized_sessions = []
        for session in sessions:
            session_data = session.copy()
            session_data['created_at'] = session_data['created_at'].isoformat()
            session_data['last_accessed'] = session_data['last_accessed'].isoformat()
            serialized_sessions.append(session_data)

        response_data = {
            'user_id': user_id,
            'session_count': len(serialized_sessions),
            'sessions': serialized_sessions
        }
        self._send_json_response(200, response_data)

    def _handle_session_stats(self):
        """Get session statistics endpoint"""
        stats = session_manager.get_session_stats()
        self._send_json_response(200, stats)


def run_server():
    """Run the test server"""
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CORSHandler)

    print('🧪 Python Test server running on http://localhost:8000')
    print('📡 Frontend should now show "Live Data" mode!')
    print('⏱️  Processing simulation: ~17 seconds total')

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n🛑 Server stopped')
        httpd.server_close()


if __name__ == '__main__':
    run_server()