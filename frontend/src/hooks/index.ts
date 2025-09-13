import { useState, useEffect, useMemo } from "react";
import { Paper, FilterState } from "../types";
import { filterPapersByQuery, debounce } from "../utils";

export const useFilteredPapers = (papers: Paper[], filters: FilterState) => {
  return useMemo(() => {
    let filtered = papers;

    // Year range filter
    filtered = filtered.filter(paper => 
      paper.year >= filters.yearRange[0] && paper.year <= filters.yearRange[1]
    );

    // Confidence filter
    filtered = filtered.filter(paper => paper.confidence >= filters.minConfidence);

    // Cluster filter
    if (filters.clusters.length > 0) {
      filtered = filtered.filter(paper => 
        paper.cluster && filters.clusters.includes(paper.cluster)
      );
    }

    // Search query filter
    filtered = filterPapersByQuery(filtered, filters.searchQuery);

    return filtered;
  }, [papers, filters]);
};

export const useDebounce = <T>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

export const useLocalStorage = <T>(key: string, initialValue: T) => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  };

  return [storedValue, setValue] as const;
};

export const useGraphData = (papers: Paper[]) => {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  const selectedPaper = useMemo(() => {
    return selectedNodeId ? papers.find(p => p.id === selectedNodeId) || null : null;
  }, [selectedNodeId, papers]);

  const hoveredPaper = useMemo(() => {
    return hoveredNodeId ? papers.find(p => p.id === hoveredNodeId) || null : null;
  }, [hoveredNodeId, papers]);

  return {
    selectedNodeId,
    setSelectedNodeId,
    hoveredNodeId, 
    setHoveredNodeId,
    selectedPaper,
    hoveredPaper,
  };
};
