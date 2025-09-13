import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const Homepage: React.FC = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [selectedOption, setSelectedOption] = useState<string | null>(null);

  const handleExplore = () => {
    console.log("Query:", query, "Option:", selectedOption);
    navigate("/graph", { state: { query, option: selectedOption } });
  };

  const handleOptionClick = (option: string) => {
    setSelectedOption(option);
    switch (option) {
      case "Paper Search":
        setQuery("Find papers on transformer architectures");
        break;
      case "Literature Review":
        setQuery("Survey recent advances in computer vision");
        break;
      case "Domain Deep-Dive":
        setQuery("Explore all work related to language models");
        break;
    }
  };

  const isReadyToExplore = query.trim().length > 0;

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
      padding: "20px",
    }}>
      <div style={{
        maxWidth: "900px",
        width: "100%",
        textAlign: "center" as const,
        color: "white",
        padding: "60px 40px",
        borderRadius: "24px",
        backgroundColor: "rgba(255, 255, 255, 0.1)",
        backdropFilter: "blur(20px)",
        border: "1px solid rgba(255, 255, 255, 0.2)",
        boxShadow: "0 20px 60px rgba(0, 0, 0, 0.1)",
      }}>
        <div style={{
          fontSize: "4.5rem",
          fontWeight: "800",
          marginBottom: "20px",
          background: "linear-gradient(45deg, #fff, #e1e8ff)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          letterSpacing: "-3px",
        }}>
          RefGraph
        </div>

        <div style={{
          fontSize: "1.6rem",
          fontWeight: "300",
          marginBottom: "15px",
          opacity: 0.95,
          letterSpacing: "1px",
        }}>
          Understand research in context
        </div>

        <div style={{
          fontSize: "1.1rem",
          lineHeight: "1.6",
          marginBottom: "50px",
          opacity: 0.85,
          maxWidth: "700px",
          margin: "0 auto 50px auto",
        }}>
          Visualize how research papers connect, instantly see which papers have been surpassed, 
          and understand the current state of any field.
        </div>

        <div style={{ marginBottom: "30px", position: "relative" }}>
          <div style={{ position: "relative", maxWidth: "700px", margin: "0 auto" }}>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask me to explore any research area..."
              rows={2}
              style={{
                width: "100%",
                padding: "20px 60px 20px 20px",
                fontSize: "16px",
                border: "1px solid rgba(255, 255, 255, 0.3)",
                borderRadius: "30px",
                backgroundColor: "rgba(255, 255, 255, 0.9)",
                color: "#333",
                outline: "none",
                resize: "none",
                fontFamily: "inherit",
                boxShadow: "0 10px 30px rgba(0, 0, 0, 0.1)",
              }}
            />
            <button
              onClick={handleExplore}
              disabled={!isReadyToExplore}
              style={{
                position: "absolute",
                right: "8px",
                top: "50%",
                transform: "translateY(-50%)",
                width: "44px",
                height: "44px",
                borderRadius: "50%",
                border: "none",
                backgroundColor: isReadyToExplore ? "#4A90E2" : "#ccc",
                color: "white",
                cursor: isReadyToExplore ? "pointer" : "not-allowed",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "18px",
              }}
            >
              →
            </button>
          </div>
        </div>

        <div style={{
          display: "flex",
          gap: "15px",
          justifyContent: "center",
          flexWrap: "wrap" as const,
          marginBottom: "50px",
        }}>
          {["Paper Search", "Literature Review", "Domain Deep-Dive"].map((option) => (
            <button
              key={option}
              onClick={() => handleOptionClick(option)}
              style={{
                padding: "12px 24px",
                borderRadius: "25px",
                border: selectedOption === option 
                  ? "2px solid rgba(255, 255, 255, 0.8)" 
                  : "1px solid rgba(255, 255, 255, 0.3)",
                backgroundColor: selectedOption === option 
                  ? "rgba(255, 255, 255, 0.2)" 
                  : "rgba(255, 255, 255, 0.1)",
                color: "white",
                cursor: "pointer",
                fontSize: "14px",
                fontWeight: "500",
                transition: "all 0.3s ease",
              }}
            >
              {option}
            </button>
          ))}
        </div>

        <div style={{ fontSize: "0.9rem", opacity: 0.7 }}>
          Powered by AI agents and real-time paper analysis
        </div>
      </div>
    </div>
  );
};

export default Homepage;