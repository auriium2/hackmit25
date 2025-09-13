import React from "react";
import { useNavigate } from "react-router-dom";

const Homepage: React.FC = () => {
  const navigate = useNavigate();

  const handleExploreClick = () => {
    navigate("/graph");
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "Arial, sans-serif",
      padding: "20px",
    }}>
      <div style={{
        maxWidth: "800px",
        textAlign: "center",
        color: "white",
        padding: "60px 40px",
        borderRadius: "20px",
        backgroundColor: "rgba(255, 255, 255, 0.1)",
        backdropFilter: "blur(10px)",
        border: "1px solid rgba(255, 255, 255, 0.2)",
        boxShadow: "0 20px 40px rgba(0, 0, 0, 0.1)",
      }}>
        {/* Logo/Brand */}
        <div style={{
          fontSize: "4rem",
          fontWeight: "800",
          marginBottom: "20px",
          background: "linear-gradient(45deg, #fff, #e1e8ff)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text",
          letterSpacing: "-2px",
        }}>
          RefGraph
        </div>

        {/* Tagline */}
        <div style={{
          fontSize: "1.5rem",
          fontWeight: "300",
          marginBottom: "30px",
          opacity: 0.9,
          letterSpacing: "1px",
        }}>
          Understand research in context
        </div>

        {/* One-sentence explanation */}
        <div style={{
          fontSize: "1.1rem",
          lineHeight: "1.6",
          marginBottom: "50px",
          opacity: 0.85,
          maxWidth: "600px",
          margin: "0 auto 50px auto",
        }}>
          Visualize how research papers connect through citations and performance comparisons, 
          instantly see which papers have been surpassed, and understand the current state of any field.
        </div>

        {/* Features */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "30px",
          marginBottom: "50px",
          opacity: 0.9,
        }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: "2rem", marginBottom: "10px" }}>🕸️</div>
            <div style={{ fontSize: "1rem", fontWeight: "600" }}>Interactive Network</div>
            <div style={{ fontSize: "0.9rem", opacity: 0.8, marginTop: "5px" }}>
              Explore citation relationships
            </div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: "2rem", marginBottom: "10px" }}>⚡</div>
            <div style={{ fontSize: "1rem", fontWeight: "600" }}>Performance Tracking</div>
            <div style={{ fontSize: "0.9rem", opacity: 0.8, marginTop: "5px" }}>
              See which papers beat others
            </div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: "2rem", marginBottom: "10px" }}>🎯</div>
            <div style={{ fontSize: "1rem", fontWeight: "600" }}>Smart Filtering</div>
            <div style={{ fontSize: "0.9rem", opacity: 0.8, marginTop: "5px" }}>
              Filter by year, topic, confidence
            </div>
          </div>
        </div>

        {/* CTA Button */}
        <button
          onClick={handleExploreClick}
          style={{
            fontSize: "1.2rem",
            fontWeight: "600",
            padding: "18px 40px",
            backgroundColor: "#4A90E2",
            color: "white",
            border: "none",
            borderRadius: "50px",
            cursor: "pointer",
            transition: "all 0.3s ease",
            boxShadow: "0 10px 30px rgba(74, 144, 226, 0.3)",
            transform: "translateY(0)",
            letterSpacing: "0.5px",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-3px)";
            e.currentTarget.style.boxShadow = "0 15px 40px rgba(74, 144, 226, 0.4)";
            e.currentTarget.style.backgroundColor = "#357ABD";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(0)";
            e.currentTarget.style.boxShadow = "0 10px 30px rgba(74, 144, 226, 0.3)";
            e.currentTarget.style.backgroundColor = "#4A90E2";
          }}
        >
          Explore Papers →
        </button>

        {/* Subtitle under button */}
        <div style={{
          fontSize: "0.9rem",
          opacity: 0.7,
          marginTop: "20px",
        }}>
          Start with sample research papers or upload your own data
        </div>
      </div>

      {/* Background decorations */}
      <div style={{
        position: "absolute",
        top: "10%",
        left: "10%",
        width: "100px",
        height: "100px",
        borderRadius: "50%",
        backgroundColor: "rgba(255, 255, 255, 0.1)",
        animation: "float 6s ease-in-out infinite",
      }} />
      <div style={{
        position: "absolute",
        bottom: "15%",
        right: "15%",
        width: "150px",
        height: "150px",
        borderRadius: "50%",
        backgroundColor: "rgba(255, 255, 255, 0.05)",
        animation: "float 8s ease-in-out infinite reverse",
      }} />
      
      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
      `}</style>
    </div>
  );
};

export default Homepage;
