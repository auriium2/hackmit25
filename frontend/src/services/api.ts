// API Service for RefGraph Backend
const API_BASE_URL = 'http://localhost:8000';

export interface QueryRequest {
  query: string;
}

export interface QueryResponse {
  systemid: string;
  status: string;
}

export interface StatusResponse {
  systemid: string;
  status: string;
}

export interface GraphNode {
  id: string;
  label: string;
  data: {
    id: string;
    title: string;
    authors: string[];
    year: number;
    abstract: string;
    doi?: string;
    venue?: string;
    citations: number;
    cluster: string;
    confidence: number;
    summary: string;
    metrics: Record<string, any>;
    embedding: number[];
  };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string;
  data: {
    relation: string;
    confidence: number;
  };
}

export interface GraphResponse {
  systemid: string;
  graph: {
    nodes: GraphNode[];
    edges: GraphEdge[];
    metadata: {
      query: string;
      total_papers: number;
      processing_time: number;
      insights: any;
      seed_papers: any[];
    };
  };
}

export interface PaperDetailsResponse {
  openalex_id: string;
  title: string;
  abstract?: string;
  year?: number;
  citations?: number;
  doi?: string;
  url?: string;
  authors?: string[];
  venue?: string;
}

export interface BenchmarkMetrics {
  accuracy?: number;
  f1_score?: number;
  bleu_score?: number;
  rouge_score?: number;
  perplexity?: number;
  inference_time?: number;
  model_size?: number;
  training_time?: number;
  dataset?: string;
  benchmark_suite?: string;
  evaluation_date?: string;
}

export interface PaperAnalysisResponse {
  paper_id: string;
  openalex_data: PaperDetailsResponse;
  benchmark_metrics: BenchmarkMetrics;
  analysis_summary: {
    key_contributions: string[];
    methodology: string[];
    strengths: string[];
    limitations: string[];
    impact_score: number;
    relevance_score: number;
  };
  processing_metadata: {
    extraction_confidence: number;
    last_updated: string;
    data_sources: string[];
  };
}

class APIService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 10000): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  }

  async checkHealth(): Promise<boolean> {
    try {
      const response = await this.fetchWithTimeout(`${this.baseUrl}/health`, {}, 5000);
      return response.ok;
    } catch (error) {
      console.warn('Backend health check failed:', error);
      return false;
    }
  }

  async submitQuery(query: string): Promise<QueryResponse> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/query`, {
      method: 'POST',
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      throw new Error(`Failed to submit query: ${response.statusText}`);
    }

    return response.json();
  }

  async checkStatus(systemId: string): Promise<StatusResponse> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/check_status?systemid=${systemId}`);

    if (!response.ok) {
      throw new Error(`Failed to check status: ${response.statusText}`);
    }

    return response.json();
  }

  async getGraph(systemId: string): Promise<GraphResponse> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/pull_final_graph?systemid=${systemId}`);

    if (!response.ok) {
      throw new Error(`Failed to get graph: ${response.statusText}`);
    }

    return response.json();
  }

  async getPaperDetails(openalexId: string): Promise<PaperDetailsResponse> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/get_details?openalexid=${openalexId}`);

    if (!response.ok) {
      throw new Error(`Failed to get paper details: ${response.statusText}`);
    }

    return response.json();
  }

  async getPaperAnalysis(paperId: string): Promise<PaperAnalysisResponse> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/analyze_paper?paper_id=${paperId}`);

    if (!response.ok) {
      throw new Error(`Failed to get paper analysis: ${response.statusText}`);
    }

    return response.json();
  }

  async getBenchmarkMetrics(paperId: string): Promise<BenchmarkMetrics> {
    const response = await this.fetchWithTimeout(`${this.baseUrl}/get_benchmarks?paper_id=${paperId}`);

    if (!response.ok) {
      throw new Error(`Failed to get benchmark metrics: ${response.statusText}`);
    }

    return response.json();
  }

  // Polling helper for status updates
  async pollForGraph(systemId: string, onStatusUpdate?: (status: string) => void): Promise<GraphResponse> {
    const pollInterval = 500; // 0.5 seconds (twice per second)
    const maxAttempts = 600; // 5 minutes max (600 * 0.5s = 300s)
    let attempts = 0;

    return new Promise(async (resolve, reject) => {
      const poll = async () => {
        try {
          attempts++;

          if (attempts > maxAttempts) {
            reject(new Error('Polling timeout: Processing took too long'));
            return;
          }

          const statusResponse = await this.checkStatus(systemId);
          onStatusUpdate?.(statusResponse.status);

          if (statusResponse.status === 'done') {
            const graphResponse = await this.getGraph(systemId);
            resolve(graphResponse);
          } else if (statusResponse.status.startsWith('fail')) {
            reject(new Error(`Processing failed: ${statusResponse.status}`));
          } else {
            // Continue polling
            setTimeout(poll, pollInterval);
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }
}

// Create singleton instance
export const apiService = new APIService();

export default APIService;