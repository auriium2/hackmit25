import { useState } from 'react';
import { LandingPage } from './components/LandingPage';
import { LoadingPage } from './components/LoadingPage';
import { GraphView } from './components/GraphView';

type AppState = 'landing' | 'loading' | 'graph';

export default function App() {
  const [currentState, setCurrentState] = useState<AppState>('landing');
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = (query: string, type: 'search' | 'review' | 'dive') => {
    if (!query.trim()) return; // Don't proceed if no query
    setSearchQuery(query.trim());
    setCurrentState('loading');
  };

  const handleLoadingComplete = () => {
    setCurrentState('graph');
  };

  const handleBackToLanding = () => {
    setCurrentState('landing');
    setSearchQuery('');
  };

  switch (currentState) {
    case 'landing':
      return <LandingPage onSearch={handleSearch} />;
    case 'loading':
      return <LoadingPage onComplete={handleLoadingComplete} />;
    case 'graph':
      return <GraphView searchQuery={searchQuery} onBack={handleBackToLanding} />;
    default:
      return <LandingPage onSearch={handleSearch} />;
  }
}