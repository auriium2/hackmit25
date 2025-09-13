import React, { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import { GraphNode, GraphEdge } from "../../types";

interface GraphVisualizationProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeSelect: (nodeId: string) => void;
  onNodeHover: (nodeId: string | null) => void;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  nodes,
  edges,
  onNodeSelect,
  onNodeHover,
}) => {
  const cyRef = useRef<HTMLDivElement>(null);
  const cyInstance = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!cyRef.current) return;

    cyInstance.current = cytoscape({
      container: cyRef.current,
      elements: [
        ...nodes.map(node => ({
          data: {
            id: node.id,
            label: node.label,
            title: node.data.title,
          },
        })),
        ...edges.map(edge => ({
          data: {
            id: edge.id,
            source: edge.source,
            target: edge.target,
            type: edge.type,
          },
        })),
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#4A90E2",
            label: "data(title)",
            "text-valign": "center",
            "text-halign": "center",
            width: "60px",
            height: "60px",
            "font-size": "12px",
            "text-wrap": "wrap",
            "text-max-width": "100px",
          },
        },
        {
          selector: "edge",
          style: {
            "line-color": "#666",
            "target-arrow-color": "#666",
            "target-arrow-shape": "triangle",
            width: 2,
          },
        },
      ],
      layout: {
        name: "grid",
        rows: 2,
      },
    });

    cyInstance.current.on("tap", "node", (evt) => {
      const nodeId = evt.target.id();
      onNodeSelect(nodeId);
    });

    cyInstance.current.on("mouseover", "node", (evt) => {
      const nodeId = evt.target.id();
      onNodeHover(nodeId);
    });

    cyInstance.current.on("mouseout", "node", () => {
      onNodeHover(null);
    });

    return () => {
      if (cyInstance.current) {
        cyInstance.current.destroy();
      }
    };
  }, [nodes, edges, onNodeSelect, onNodeHover]);

  return (
    <div
      ref={cyRef}
      style={{
        width: "100%",
        height: "600px",
        border: "1px solid #ccc",
        borderRadius: "8px",
      }}
    />
  );
};

export default GraphVisualization;
