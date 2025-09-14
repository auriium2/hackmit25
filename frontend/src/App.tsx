import { useState } from 'react';
import { LandingPage } from './components/LandingPage';
import { LoadingPage } from './components/LoadingPage';
import { GraphView } from './components/GraphView';
import { apiService, GraphResponse } from './services/api';

type AppState = 'landing' | 'loading' | 'graph';

export default function App() {
  const [currentState, setCurrentState] = useState<AppState>('landing');
  const [searchQuery, setSearchQuery] = useState('');
  const [systemId, setSystemId] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<GraphResponse | null>(null);

  const handleSearch = async (query: string, type: 'search' | 'review' | 'dive') => {
    if (!query.trim()) return; // Don't proceed if no query

    const trimmedQuery = query.trim();
    setSearchQuery(trimmedQuery);
    setCurrentState('loading');

    try {
      // Check if backend is available and submit query
      const isHealthy = await apiService.checkHealth();

      if (isHealthy) {
        const response = await apiService.submitQuery(trimmedQuery);
        if (response.status === 'ok') {
          setSystemId(response.systemid);
          console.log('Query submitted successfully:', response.systemid);
        } else {
          console.warn('Query submission failed, using mock mode');
          setSystemId(null);
        }
      } else {
        console.warn('Backend not available, using mock mode');
        setSystemId(null);
      }
    } catch (error) {
      console.error('Failed to submit query:', error);
      setSystemId(null); // Fallback to mock mode
    }
  };

  const handleLoadingComplete = (responseGraphData?: GraphResponse) => {
    if (responseGraphData) {
      setGraphData(responseGraphData);
      console.log('Graph data received:', responseGraphData);
    }
    setCurrentState('graph');
  };

  const handleBackToLanding = () => {
    setCurrentState('landing');
    setSearchQuery('');
    setSystemId(null);
    setGraphData(null);
  };

  switch (currentState) {
    case 'landing':
      return <LandingPage onSearch={handleSearch} />;
    case 'loading':
      return (
        <LoadingPage
          onComplete={handleLoadingComplete}
          searchQuery={searchQuery}
          systemId={systemId}
        />
      );
    case 'graph':
      return (
        <GraphView
          searchQuery={searchQuery}
          onBack={handleBackToLanding}
          graphData={graphData}
        />
      );
    default:
      return <LandingPage onSearch={handleSearch} />;
  }
}