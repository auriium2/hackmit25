import { motion } from 'motion/react';
import { useState } from 'react';
import { Calendar, TrendingUp, Target, Search, ChevronDown, X, Eye, EyeOff, Layers } from 'lucide-react';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { Badge } from './ui/badge';

export interface FilterOptions {
  dateRange: [number, number];
  trustScore: number; // Changed to single value
  relationToSeed: 'all' | 'direct' | 'indirect' | 'highly-related';
  benchmark: 'accuracy' | 'citations' | 'relevance' | 'recency';
  sortBy: 'time' | 'trust' | 'relevance' | 'citations' | 'force-directed';
  searchIn: string[];
  visibleLayers: string[]; // Added layers
}

interface FilterPanelProps {
  filters: FilterOptions;
  onFiltersChange: (filters: FilterOptions) => void;
  onClose: () => void;
}

export function FilterPanel({ filters, onFiltersChange, onClose }: FilterPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const benchmarkOptions = [
    { value: 'accuracy', label: 'Benchmark Accuracy', icon: Target },
    { value: 'citations', label: 'Citation Count', icon: TrendingUp },
    { value: 'relevance', label: 'Relevance Score', icon: Search },
    { value: 'recency', label: 'Publication Date', icon: Calendar },
  ];

  const relationOptions = [
    { value: 'all', label: 'All Papers' },
    { value: 'direct', label: 'Direct Citations' },
    { value: 'indirect', label: 'Indirect Relations' },
    { value: 'highly-related', label: 'Highly Related' },
  ];

  const searchInOptions = [
    'titles', 'abstracts', 'authors', 'keywords', 'citations', 'methodology'
  ];

  const layers = [
    { id: 'citations', label: 'Citations', color: '#3b82f6' },
    { id: 'collaborations', label: 'Collaborations', color: '#10b981' },
    { id: 'methodology', label: 'Methodology', color: '#8b5cf6' },
    { id: 'temporal', label: 'Temporal', color: '#f59e0b' },
  ];

  const updateFilter = (key: keyof FilterOptions, value: any) => {
    onFiltersChange({ ...filters, [key]: value });
  };

  const toggleSearchIn = (option: string) => {
    const current = filters.searchIn;
    const updated = current.includes(option)
      ? current.filter(item => item !== option)
      : [...current, option];
    updateFilter('searchIn', updated);
  };

  const toggleLayer = (layerId: string) => {
    const current = filters.visibleLayers;
    const updated = current.includes(layerId)
      ? current.filter(id => id !== layerId)
      : [...current, layerId];
    updateFilter('visibleLayers', updated);
  };

  const clearFilters = () => {
    onFiltersChange({
      dateRange: [2000, 2025],
      trustScore: 0, // Default to 0% (show all papers)
      relationToSeed: 'all',
      benchmark: 'accuracy',
      sortBy: 'force-directed',
      searchIn: [],
      visibleLayers: ['citations', 'collaborations', 'methodology', 'temporal'],
    });
  };

  const hasActiveFilters = 
    filters.dateRange[0] !== 2000 || 
    filters.dateRange[1] !== 2025 ||
    filters.trustScore !== 0 ||
    filters.relationToSeed !== 'all' ||
    filters.searchIn.length > 0;

  return (
    <motion.div
      className="fixed top-0 right-0 h-full w-80 bg-black/95 backdrop-blur-lg border-l border-cyan-500/20 shadow-2xl z-50"
      initial={{ x: 320 }}
      animate={{ x: 0 }}
      exit={{ x: 320 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
    >
      <div className="p-6 h-full overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-cyan-100 text-lg font-medium">Research Filters</h3>
          <div className="flex items-center space-x-2">
            {hasActiveFilters && (
              <Button
                onClick={clearFilters}
                variant="ghost"
                size="sm"
                className="text-cyan-100/70 hover:text-cyan-100"
              >
                Clear
              </Button>
            )}
            <Button
              onClick={onClose}
              variant="ghost"
              size="sm"
              className="text-white/70 hover:text-white"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="space-y-6">
          {/* Quick Search within results */}
          <div>
            <label className="text-cyan-100/80 text-sm mb-2 block">
              Search within results
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-cyan-100/50" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter current results..."
                className="w-full pl-10 pr-4 py-2 bg-black/30 border border-cyan-500/20 rounded-lg text-cyan-100 placeholder-cyan-100/50 focus:outline-none focus:border-cyan-400"
              />
            </div>
          </div>

          {/* Search In Options */}
          <div>
            <label className="text-cyan-100/80 text-sm mb-3 block">
              Search in fields
            </label>
            <div className="flex flex-wrap gap-2">
              {searchInOptions.map((option) => (
                <Badge
                  key={option}
                  variant={filters.searchIn.includes(option) ? "default" : "outline"}
                  className={`cursor-pointer transition-all duration-200 ${
                    filters.searchIn.includes(option)
                      ? 'bg-cyan-500 text-white border-cyan-500'
                      : 'text-cyan-100/70 border-cyan-500/30 hover:border-cyan-400 hover:text-cyan-400'
                  }`}
                  onClick={() => toggleSearchIn(option)}
                >
                  {option}
                </Badge>
              ))}
            </div>
          </div>

          {/* Publication Date with Transparency Gradient Based on Selection */}
          <div>
            <label className="text-cyan-100/80 text-sm mb-3 block">
              Publication Date: {filters.dateRange[0]} - {filters.dateRange[1]}
            </label>
            <div className="relative">
              {/* Dynamic transparency gradient based on current selection */}
              <div 
                className="absolute inset-0 h-3 rounded-lg pointer-events-none"
                style={{
                  background: `linear-gradient(to right, 
                    rgba(255, 255, 255, 0.2), 
                    rgba(255, 255, 255, ${0.2 + ((filters.dateRange[0] - 2000) / 25) * 0.8}), 
                    rgba(255, 255, 255, ${0.2 + ((filters.dateRange[1] - 2000) / 25) * 0.8}), 
                    rgba(255, 255, 255, 1))`,
                  top: '50%',
                  transform: 'translateY(-50%)',
                }}
              />
              <Slider
                value={filters.dateRange}
                onValueChange={(value) => updateFilter('dateRange', value)}
                min={2000}
                max={2025}
                step={1}
                className="w-full relative z-10"
              />
            </div>
          </div>

          {/* Performance Score - Single slider with white to green gradient */}
          <div>
            <label className="text-cyan-100/80 text-sm mb-3 block">
              Performance Score: {filters.trustScore}%
            </label>
            <div className="relative">
              <Slider
                value={[filters.trustScore]}
                onValueChange={(value) => updateFilter('trustScore', value[0])}
                min={0}
                max={100}
                step={5}
                className="w-full [&_[data-orientation=horizontal]]:bg-gradient-to-r [&_[data-orientation=horizontal]]:from-white [&_[data-orientation=horizontal]]:to-green-500 [&_[role=slider]]:bg-white [&_[role=slider]]:border-white"
              />
            </div>
          </div>

          {/* Relation to Seed */}
          <div>
            <label className="text-cyan-100/80 text-sm mb-3 block">
              Relation to Seed Paper
            </label>
            <Select 
              value={filters.relationToSeed} 
              onValueChange={(value) => updateFilter('relationToSeed', value)}
            >
              <SelectTrigger className="w-full bg-black/30 border-cyan-500/20 text-cyan-100">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {relationOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Benchmark Selection */}
          <div>
            <label className="text-cyan-100/80 text-sm mb-3 block">
              Primary Benchmark
            </label>
            <div className="space-y-2">
              {benchmarkOptions.map((option) => {
                const Icon = option.icon;
                return (
                  <motion.div
                    key={option.value}
                    className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                      filters.benchmark === option.value
                        ? 'bg-cyan-500/20 border-cyan-400 text-cyan-400'
                        : 'bg-black/20 border-cyan-500/20 text-cyan-100/70 hover:border-cyan-400/50 hover:text-cyan-100'
                    }`}
                    onClick={() => updateFilter('benchmark', option.value)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="flex items-center space-x-3">
                      <Icon className="w-4 h-4" />
                      <span className="text-sm">{option.label}</span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Layers */}
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <Layers className="w-4 h-4 text-cyan-100/60" />
              <span className="text-cyan-100/80 text-sm">Layers</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {layers.map((layer) => (
                <button
                  key={layer.id}
                  onClick={() => toggleLayer(layer.id)}
                  className={`flex items-center justify-between p-2 rounded-lg transition-all duration-200 ${
                    filters.visibleLayers.includes(layer.id)
                      ? 'bg-cyan-500/20 border border-cyan-400/50'
                      : 'bg-black/30 border border-cyan-500/20'
                  }`}
                >
                  <div className="flex items-center space-x-2">
                    <div 
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: layer.color }}
                    />
                    <span className="text-cyan-100/80 text-xs">{layer.label}</span>
                  </div>
                  {filters.visibleLayers.includes(layer.id) ? (
                    <Eye className="w-3 h-3 text-cyan-100/60" />
                  ) : (
                    <EyeOff className="w-3 h-3 text-cyan-100/40" />
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Needle in a Haystack */}
          <div className="p-4 bg-cyan-500/10 border border-cyan-400/30 rounded-lg">
            <div className="flex items-center space-x-2 mb-2">
              <Target className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-medium">Needle in Haystack</span>
            </div>
            <p className="text-cyan-100/70 text-xs">
              Find specific information within the research corpus. Papers are ranked by precision of matching your query.
            </p>
          </div>

          {/* Active Filters Summary */}
          {hasActiveFilters && (
            <div className="mt-6 pt-4 border-t border-white/10">
              <h4 className="text-white/80 text-sm mb-2">Active Filters</h4>
              <div className="text-xs text-white/60 space-y-1">
                {filters.relationToSeed !== 'all' && (
                  <div>• Relation: {relationOptions.find(o => o.value === filters.relationToSeed)?.label}</div>
                )}
                {(filters.dateRange[0] !== 2000 || filters.dateRange[1] !== 2025) && (
                  <div>• Date: {filters.dateRange[0]} - {filters.dateRange[1]}</div>
                )}
                {filters.trustScore !== 0 && (
                  <div>• Performance: {filters.trustScore}%</div>
                )}
                {filters.searchIn.length > 0 && (
                  <div>• Fields: {filters.searchIn.join(', ')}</div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}