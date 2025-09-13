import { Paper, GraphNode, GraphEdge } from "../types";

export const filterPapersByQuery = (papers: Paper[], query: string): Paper[] => {
  if (!query.trim()) return papers;
  
  const lowerQuery = query.toLowerCase();
  return papers.filter(paper => 
    paper.title.toLowerCase().includes(lowerQuery) ||
    paper.authors.some(author => author.toLowerCase().includes(lowerQuery)) ||
    paper.abstract.toLowerCase().includes(lowerQuery) ||
    (paper.venue && paper.venue.toLowerCase().includes(lowerQuery))
  );
};

export const generateGraphElements = (
  papers: Paper[],
  showCitations: boolean,
  showBeatsRelations: boolean
): { nodes: GraphNode[]; edges: GraphEdge[] } => {
  const nodes: GraphNode[] = papers.map(paper => ({
    id: paper.id,
    label: paper.title.length > 50 ? paper.title.substring(0, 50) + "..." : paper.title,
    data: paper,
  }));

  const edges: GraphEdge[] = [];
  const paperIds = new Set(papers.map(p => p.id));

  papers.forEach(paper => {
    // Citation edges
    if (showCitations) {
      paper.citations.forEach(citationId => {
        if (paperIds.has(citationId)) {
          edges.push({
            id: `citation-${citationId}-${paper.id}`,
            source: citationId,
            target: paper.id,
            type: "citation",
            data: {},
          });
        }
      });
    }

    // Beats relation edges
    if (showBeatsRelations) {
      paper.beatsRelations.forEach(relation => {
        if (paperIds.has(relation.targetPaperId)) {
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
};

export const calculateSimilarity = (paper1: Paper, paper2: Paper): number => {
  if (!paper1.embedding || !paper2.embedding) return 0;
  
  // Cosine similarity
  let dotProduct = 0;
  let norm1 = 0;
  let norm2 = 0;
  
  for (let i = 0; i < Math.min(paper1.embedding.length, paper2.embedding.length); i++) {
    dotProduct += paper1.embedding[i] * paper2.embedding[i];
    norm1 += paper1.embedding[i] * paper1.embedding[i];
    norm2 += paper2.embedding[i] * paper2.embedding[i];
  }
  
  return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
};

export const getClusterColors = (): Record<string, string> => ({
  "NLP": "#4A90E2",
  "Computer Vision": "#7CB342", 
  "Machine Learning": "#FB8C00",
  "Robotics": "#8E24AA",
  "Systems": "#F4511E",
  "Theory": "#00ACC1",
  "Security": "#E53935",
  "HCI": "#FFB300",
  "Default": "#666666",
});

export const formatAuthors = (authors: string[], maxLength: number = 50): string => {
  const joined = authors.join(", ");
  return joined.length > maxLength ? joined.substring(0, maxLength) + "..." : joined;
};

export const debounce = <T extends (...args: any[]) => void>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};
