import React from "react";
import { Paper } from "../../types";

interface PaperDetailsProps {
  paper: Paper | null;
  onCompare: (paper: Paper) => void;
}

const PaperDetails: React.FC<PaperDetailsProps> = ({ paper, onCompare }) => {
  if (!paper) {
    return (
      <div style={{ padding: "20px", backgroundColor: "#f5f5f5", borderRadius: "8px" }}>
        <p style={{ fontStyle: "italic", color: "#666" }}>
          Select a paper node to view details
        </p>
      </div>
    );
  }

  return (
    <div style={{ padding: "20px", backgroundColor: "#f5f5f5", borderRadius: "8px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <h3 style={{ margin: "0 0 15px 0", flex: 1 }}>{paper.title}</h3>
        <button
          onClick={() => onCompare(paper)}
          style={{
            padding: "8px 16px",
            backgroundColor: "#4A90E2",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            marginLeft: "10px",
          }}
        >
          Compare
        </button>
      </div>
      
      <div style={{ marginBottom: "15px" }}>
        <strong>Authors:</strong> {paper.authors.join(", ")}
      </div>
      
      <div style={{ marginBottom: "15px" }}>
        <strong>Year:</strong> {paper.year} | 
        <strong> Confidence:</strong> {paper.confidence}% |
        <strong> Cluster:</strong> {paper.cluster || "N/A"}
      </div>

      {paper.venue && (
        <div style={{ marginBottom: "15px" }}>
          <strong>Venue:</strong> {paper.venue}
        </div>
      )}

      {paper.arxivId && (
        <div style={{ marginBottom: "15px" }}>
          <strong>arXiv ID:</strong> {paper.arxivId}
        </div>
      )}

      <div style={{ marginBottom: "15px" }}>
        <strong>Abstract:</strong>
        <p style={{ fontSize: "14px", lineHeight: "1.5", marginTop: "5px" }}>
          {paper.abstract}
        </p>
      </div>

      {paper.summary && (
        <div style={{ marginBottom: "15px" }}>
          <strong>AI Summary:</strong>
          <p style={{ fontSize: "14px", lineHeight: "1.5", marginTop: "5px", fontStyle: "italic" }}>
            {paper.summary}
          </p>
        </div>
      )}

      <div style={{ display: "flex", gap: "30px", marginBottom: "15px" }}>
        <div>
          <strong>Citations:</strong> {paper.citations.length}
        </div>
        <div>
          <strong>"Beats" Relations:</strong> {paper.beatsRelations.length}
        </div>
      </div>

      {paper.beatsRelations.length > 0 && (
        <div style={{ marginBottom: "15px" }}>
          <strong>Papers this outperforms:</strong>
          <ul style={{ marginTop: "5px", fontSize: "14px" }}>
            {paper.beatsRelations.map((relation, index) => (
              <li key={index} style={{ marginBottom: "5px" }}>
                <em>{relation.metric}</em> (confidence: {relation.confidence}%)
                {relation.description && <span> - {relation.description}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {paper.metrics && Object.keys(paper.metrics).length > 0 && (
        <div>
          <strong>Metrics:</strong>
          <div style={{ marginTop: "5px" }}>
            {Object.entries(paper.metrics).map(([metric, value]) => (
              <div key={metric} style={{ fontSize: "14px", marginBottom: "3px" }}>
                <strong>{metric}:</strong> {value}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PaperDetails;
