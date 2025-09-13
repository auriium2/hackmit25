import React from "react";
import { FilterState } from "../../types";

interface FilterPanelProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
  availableClusters: string[];
}

const FilterPanel: React.FC<FilterPanelProps> = ({
  filters,
  onFiltersChange,
  availableClusters,
}) => {
  const handleYearChange = (type: "min" | "max", value: number) => {
    const newRange: [number, number] = [...filters.yearRange];
    if (type === "min") newRange[0] = value;
    else newRange[1] = value;
    onFiltersChange({ ...filters, yearRange: newRange });
  };

  const handleClusterToggle = (cluster: string) => {
    const newClusters = filters.clusters.includes(cluster)
      ? filters.clusters.filter(c => c !== cluster)
      : [...filters.clusters, cluster];
    onFiltersChange({ ...filters, clusters: newClusters });
  };

  return (
    <div style={{ padding: "20px", backgroundColor: "#f5f5f5", borderRadius: "8px" }}>
      <h3>Filters</h3>
      
      <div style={{ marginBottom: "20px" }}>
        <label>Search:</label>
        <input
          type="text"
          value={filters.searchQuery}
          onChange={(e) => onFiltersChange({ ...filters, searchQuery: e.target.value })}
          placeholder="Search papers..."
          style={{ width: "100%", padding: "8px", marginTop: "5px" }}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Year Range:</label>
        <div style={{ display: "flex", gap: "10px", marginTop: "5px" }}>
          <input
            type="number"
            value={filters.yearRange[0]}
            onChange={(e) => handleYearChange("min", parseInt(e.target.value))}
            placeholder="Min year"
            style={{ flex: 1, padding: "5px" }}
          />
          <input
            type="number"
            value={filters.yearRange[1]}
            onChange={(e) => handleYearChange("max", parseInt(e.target.value))}
            placeholder="Max year"
            style={{ flex: 1, padding: "5px" }}
          />
        </div>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Minimum Confidence: {filters.minConfidence}%</label>
        <input
          type="range"
          min="0"
          max="100"
          value={filters.minConfidence}
          onChange={(e) => onFiltersChange({ ...filters, minConfidence: parseInt(e.target.value) })}
          style={{ width: "100%", marginTop: "5px" }}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <label>Clusters:</label>
        <div style={{ marginTop: "5px" }}>
          {availableClusters.map(cluster => (
            <div key={cluster} style={{ margin: "5px 0" }}>
              <label>
                <input
                  type="checkbox"
                  checked={filters.clusters.includes(cluster)}
                  onChange={() => handleClusterToggle(cluster)}
                  style={{ marginRight: "8px" }}
                />
                {cluster}
              </label>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <div style={{ margin: "10px 0" }}>
          <label>
            <input
              type="checkbox"
              checked={filters.showCitations}
              onChange={(e) => onFiltersChange({ ...filters, showCitations: e.target.checked })}
              style={{ marginRight: "8px" }}
            />
            Show Citations
          </label>
        </div>
        <div style={{ margin: "10px 0" }}>
          <label>
            <input
              type="checkbox"
              checked={filters.showBeatsRelations}
              onChange={(e) => onFiltersChange({ ...filters, showBeatsRelations: e.target.checked })}
              style={{ marginRight: "8px" }}
            />
            Show "Beats" Relations
          </label>
        </div>
      </div>
    </div>
  );
};

export default FilterPanel;
