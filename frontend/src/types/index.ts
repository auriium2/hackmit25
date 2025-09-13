export interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number;
  abstract: string;
  arxivId?: string;
  doi?: string;
  venue?: string;
  citations: string[];
  beatsRelations: BeatsRelation[];
  embedding?: number[];
  cluster?: string;
  confidence: number;
  summary?: string;
  metrics?: Record<string, number>;
}

export interface BeatsRelation {
  targetPaperId: string;
  metric: string;
  confidence: number;
  description: string;
}

export interface GraphNode {
  id: string;
  label: string;
  data: Paper;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: "citation" | "beats";
  data: {
    confidence?: number;
    metric?: string;
    description?: string;
  };
}

export interface FilterState {
  yearRange: [number, number];
  minConfidence: number;
  clusters: string[];
  searchQuery: string;
  showCitations: boolean;
  showBeatsRelations: boolean;
}

export interface ComparisonState {
  paper1: Paper | null;
  paper2: Paper | null;
  isVisible: boolean;
}

export interface GraphState {
  nodes: GraphNode[];
  edges: GraphEdge[];
  selectedNode: string | null;
  hoveredNode: string | null;
}
