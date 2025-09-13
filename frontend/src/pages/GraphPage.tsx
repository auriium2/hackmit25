import React, { useState, useCallback, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import GraphVisualization from "../components/Graph/GraphVisualization";
import FilterPanel from "../components/Filters/FilterPanel";
import PaperDetails from "../components/PaperDetails/PaperDetails";
import PaperComparison from "../components/PaperComparison/PaperComparison";
import { 
  Paper, 
  GraphNode, 
  GraphEdge, 
  FilterState, 
  ComparisonState,
  GraphState 
} from "../types";

// Mock data for development
const mockPapers: Paper[] = [
  {
    id: "1",
    title: "Attention Is All You Need",
    authors: ["Vaswani et al."],
    year: 2017,
    abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
    venue: "NIPS",
    arxivId: "1706.03762",
    citations: ["2", "3"],
    beatsRelations: [],
    cluster: "NLP",
    confidence: 95,
    summary: "Introduced the Transformer architecture, revolutionizing NLP.",
    metrics: { "BLEU": 28.4, "Perplexity": 4.2 }
  },
  {
    id: "2", 
    title: "BERT: Pre-training of Deep Bidirectional Transformers",
    authors: ["Devlin et al."],
    year: 2018,
    abstract: "We introduce a new language representation model called BERT...",
    venue: "NAACL",
    citations: ["3"],
    beatsRelations: [{ targetPaperId: "1", metric: "GLUE Score", confidence: 85, description: "Outperforms on most GLUE tasks" }],
    cluster: "NLP",
    confidence: 92,
    summary: "Bidirectional training of transformers for language understanding.",
  },
  {
    id: "3",
    title: "GPT-3: Language Models are Few-Shot Learners",
    authors: ["Brown et al."],
    year: 2020,
    abstract: "Recent work has demonstrated substantial gains on many NLP tasks...",
    venue: "NeurIPS",
    citations: [],
    beatsRelations: [
      { targetPaperId: "1", metric: "Few-shot Performance", confidence: 88, description: "Superior few-shot capabilities" },
      { targetPaperId: "2", metric: "Zero-shot Performance", confidence: 79, description: "Better zero-shot performance on many tasks" }
    ],
    cluster: "NLP",
    confidence: 89,
    summary: "Large-scale language model demonstrating emergent few-shot learning abilities.",
  }
];

const GraphPage: React.FC = () => {
  const navigate = useNavigate();
  
  const [filters, setFilters] = useState<FilterState>({
    yearRange: [2015, 2024],
    minConfidence: 0,
    clusters: [],
    searchQuery: "",
    showCitations: true,
    showBeatsRelations: true,
  });

  const [graphState, setGraphState] = useState<GraphState>({
    nodes: [],
    edges: [],
    selectedNode: null,
    hoveredNode: null,
  });

  const [comparison, setComparison] = useState<ComparisonState>({
    paper1: null,
    paper2: null,
    isVisible: false,
  });

  // Filter papers based on current filters
  const filteredPapers = useMemo(() => {
    return mockPapers.filter(paper => {
      // Year filter
      if (paper.year < filters.yearRange[0] || paper.year > filters.yearRange[1]) {
        return false;
      }
      
      // Confidence filter
      if (paper.confidence < filters.minConfidence) {
        return false;
      }
      
      // Cluster filter
      if (filters.clusters.length > 0 && !filters.clusters.includes(paper.cluster || "")) {
        return false;
      }
      
      // Search query filter
      if (filters.searchQuery && !paper.title.toLowerCase().includes(filters.searchQuery.toLowerCase()) &&
          !paper.authors.some(author => author.toLowerCase().includes(filters.searchQuery.toLowerCase()))) {
        return false;
      }
      
      return true;
    });
  }, [filters]);

  // Generate graph data from filtered papers
  const { nodes, edges } = useMemo(() => {
    const nodes: GraphNode[] = filteredPapers.map(paper => ({
      id: paper.id,
      label: paper.title.substring(0, 50) + "...",
      data: paper,
    }));

    const edges: GraphEdge[] = [];
    
    filteredPapers.forEach(paper => {
      // Citation edges
      if (filters.showCitations) {
        paper.citations.forEach(citationId => {
          if (filteredPapers.find(p => p.id === citationId)) {
            edges.push({
              id: `citation-${paper.id}-${citationId}`,
              source: citationId,
              target: paper.id,
              type: "citation",
              data: {},
            });
          }
        });
      }
      
      // Beats relation edges
      if (filters.showBeatsRelations) {
        paper.beatsRelations.forEach(relation => {
          if (filteredPapers.find(p => p.id === relation.targetPaperId)) {
            edges.push({
              id: `beats-${paper.id}-${relation.targetPaperId}`,
              source: paper.id,
              target: relation.targetPaperId,
              type: "beats",
              data: {
                confidence: relation.confidence,
                metric: relation.metric,
                description: relation.description,
              },
            });
          }
        });
      }
    });

    return { nodes, edges };
  }, [filteredPapers, filters.showCitations, filters.showBeatsRelations]);

  // Update graph state when nodes/edges change
  React.useEffect(() => {
    setGraphState(prev => ({ ...prev, nodes, edges }));
  }, [nodes, edges]);

  const handleNodeSelect = useCallback((nodeId: string) => {
    setGraphState(prev => ({ ...prev, selectedNode: nodeId }));
  }, []);

  const handleNodeHover = useCallback((nodeId: string | null) => {
    setGraphState(prev => ({ ...prev, hoveredNode: nodeId }));
  }, []);

  const selectedPaper = graphState.selectedNode 
    ? mockPapers.find(p => p.id === graphState.selectedNode) || null
    : null;

  const handleCompare = (paper: Paper) => {
    if (!comparison.paper1) {
      setComparison(prev => ({ ...prev, paper1: paper }));
    } else if (!comparison.paper2) {
      setComparison(prev => ({ ...prev, paper2: paper, isVisible: true }));
    } else {
      setComparison({ paper1: paper, paper2: null, isVisible: false });
    }
  };

  const availableClusters = Array.from(new Set(mockPapers.map(p => p.cluster).filter(Boolean))) as string[];

  return (
    <div style={{ 
      display: "flex", 
      height: "100vh", 
      fontFamily: "Arial, sans-serif",
      overflow: "hidden"
    }}>
      {/* Left sidebar - Filters */}
      <div style={{ 
        width: "300px", 
        minWidth: "300px",
        maxWidth: "300px",
        borderRight: "1px solid #ddd", 
        overflow: "auto",
        flexShrink: 0
      }}>
        <FilterPanel
          filters={filters}
          onFiltersChange={setFilters}
          availableClusters={availableClusters}
        />
      </div>

      {/* Main content - Graph */}
      <div style={{ 
        flex: 1,
        minWidth: 0,
        display: "flex", 
        flexDirection: "column",
        overflow: "hidden"
      }}>
        <div style={{ 
          padding: "20px", 
          borderBottom: "1px solid #ddd",
          flexShrink: 0,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}>
          <div>
            <h1 style={{ margin: 0, fontSize: "24px", color: "#333" }}>RefGraph</h1>
            <p style={{ margin: "5px 0 0 0", color: "#666" }}>
              Research Paper Visualization Tool - Showing {filteredPapers.length} papers
            </p>
          </div>
          <button
            onClick={() => navigate("/")}
            style={{
              padding: "10px 20px",
              backgroundColor: "#f5f5f5",
              border: "1px solid #ddd",
              borderRadius: "6px",
              cursor: "pointer",
              fontSize: "14px",
              color: "#333",
            }}
          >
            ← Back to Home
          </button>
        </div>
        
        <div style={{ 
          flex: 1, 
          padding: "20px",
          overflow: "hidden",
          minHeight: 0
        }}>
          <GraphVisualization
            nodes={nodes}
            edges={edges}
            onNodeSelect={handleNodeSelect}
            onNodeHover={handleNodeHover}
          />
        </div>
      </div>

      {/* Right sidebar - Paper details */}
      <div style={{ 
        width: "400px", 
        minWidth: "400px",
        maxWidth: "400px",
        borderLeft: "1px solid #ddd", 
        overflow: "auto",
        flexShrink: 0
      }}>
        <div style={{ padding: "20px" }}>
          <PaperDetails paper={selectedPaper} onCompare={handleCompare} />
          
          {comparison.paper1 && (
            <div style={{ marginTop: "20px", padding: "15px", backgroundColor: "#e3f2fd", borderRadius: "8px" }}>
              <h4 style={{ margin: "0 0 10px 0" }}>Comparison Queue:</h4>
              <div style={{ fontSize: "14px" }}>
                <strong>Paper 1:</strong> {comparison.paper1.title.substring(0, 40)}...
              </div>
              {comparison.paper2 && (
                <div style={{ fontSize: "14px", marginTop: "5px" }}>
                  <strong>Paper 2:</strong> {comparison.paper2.title.substring(0, 40)}...
                </div>
              )}
              <button
                onClick={() => setComparison({ paper1: null, paper2: null, isVisible: false })}
                style={{
                  marginTop: "10px",
                  padding: "5px 10px",
                  fontSize: "12px",
                  backgroundColor: "#f44336",
                  color: "white",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                }}
              >
                Clear
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Comparison Modal */}
      <PaperComparison
        paper1={comparison.paper1}
        paper2={comparison.paper2}
        isVisible={comparison.isVisible}
        onClose={() => setComparison(prev => ({ ...prev, isVisible: false }))}
      />
    </div>
  );
};

export default GraphPage;
