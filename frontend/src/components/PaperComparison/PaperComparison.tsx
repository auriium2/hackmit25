import React from "react";
import { Paper } from "../../types";

interface PaperComparisonProps {
  paper1: Paper | null;
  paper2: Paper | null;
  isVisible: boolean;
  onClose: () => void;
}

const PaperComparison: React.FC<PaperComparisonProps> = ({
  paper1,
  paper2,
  isVisible,
  onClose,
}) => {
  if (!isVisible) return null;

  const renderPaper = (paper: Paper | null, title: string) => (
    <div style={{ flex: 1, padding: "20px", backgroundColor: "#f9f9f9", margin: "10px" }}>
      <h3>{title}</h3>
      {paper ? (
        <>
          <h4>{paper.title}</h4>
          <p><strong>Authors:</strong> {paper.authors.join(", ")}</p>
          <p><strong>Year:</strong> {paper.year}</p>
          <p><strong>Venue:</strong> {paper.venue || "N/A"}</p>
          <p><strong>Confidence:</strong> {paper.confidence}%</p>
          <p><strong>Cluster:</strong> {paper.cluster || "N/A"}</p>
          <div style={{ marginTop: "15px" }}>
            <strong>Abstract:</strong>
            <p style={{ fontSize: "14px", lineHeight: "1.5", marginTop: "5px" }}>
              {paper.abstract}
            </p>
          </div>
          {paper.summary && (
            <div style={{ marginTop: "15px" }}>
              <strong>Summary:</strong>
              <p style={{ fontSize: "14px", lineHeight: "1.5", marginTop: "5px" }}>
                {paper.summary}
              </p>
            </div>
          )}
          <div style={{ marginTop: "15px" }}>
            <strong>Citations:</strong> {paper.citations.length}
          </div>
          <div style={{ marginTop: "10px" }}>
            <strong>"Beats" Relations:</strong> {paper.beatsRelations.length}
          </div>
          {paper.metrics && Object.keys(paper.metrics).length > 0 && (
            <div style={{ marginTop: "15px" }}>
              <strong>Metrics:</strong>
              <ul style={{ marginTop: "5px" }}>
                {Object.entries(paper.metrics).map(([metric, value]) => (
                  <li key={metric}>{metric}: {value}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      ) : (
        <p style={{ fontStyle: "italic", color: "#666" }}>
          No paper selected
        </p>
      )}
    </div>
  );

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
    >
      <div
        style={{
          backgroundColor: "white",
          borderRadius: "8px",
          maxWidth: "90%",
          maxHeight: "90%",
          overflow: "auto",
          position: "relative",
        }}
      >
        <button
          onClick={onClose}
          style={{
            position: "absolute",
            top: "10px",
            right: "15px",
            background: "none",
            border: "none",
            fontSize: "20px",
            cursor: "pointer",
            zIndex: 1001,
          }}
        >
          ×
        </button>
        <div style={{ display: "flex", minHeight: "500px" }}>
          {renderPaper(paper1, "Paper 1")}
          {renderPaper(paper2, "Paper 2")}
        </div>
        {paper1 && paper2 && (
          <div style={{ padding: "20px", borderTop: "1px solid #ddd" }}>
            <h3>Comparison Insights</h3>
            <div style={{ display: "flex", gap: "30px", marginTop: "15px" }}>
              <div>
                <strong>Year Difference:</strong> {Math.abs(paper1.year - paper2.year)} years
              </div>
              <div>
                <strong>Citation Difference:</strong> {Math.abs(paper1.citations.length - paper2.citations.length)}
              </div>
              <div>
                <strong>Confidence Difference:</strong> {Math.abs(paper1.confidence - paper2.confidence)}%
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PaperComparison;
