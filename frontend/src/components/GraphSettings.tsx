import { motion, AnimatePresence } from 'motion/react';
import { useState } from 'react';
import { Settings, ZoomIn, ZoomOut, RotateCcw, Eye, EyeOff, Layers, X, Clock } from 'lucide-react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';

interface GraphSettingsProps {
  zoom: number;
  onZoomChange: (zoom: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
  selectedNodes: string[];
  onClearSelection: () => void;
  visibleLayers: string[];
  onToggleLayer: (layer: string) => void;
  timeRange: [number, number];
  onTimeRangeChange: (range: [number, number]) => void;
}

const layers = [
  { id: 'citations', label: 'Citations', color: '#3b82f6', bgColor: 'bg-blue-500' },
  { id: 'collaborations', label: 'Collaborations', color: '#10b981', bgColor: 'bg-emerald-500' },
  { id: 'methodology', label: 'Methodology', color: '#8b5cf6', bgColor: 'bg-purple-500' },
  { id: 'temporal', label: 'Temporal', color: '#f59e0b', bgColor: 'bg-amber-500' },
];

export function GraphSettings({
  zoom,
  onZoomChange,
  onZoomIn,
  onZoomOut,
  onReset,
  selectedNodes,
  onClearSelection,
  visibleLayers,
  onToggleLayer,
  timeRange,
  onTimeRangeChange,
}: GraphSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div className="absolute bottom-6 left-6 z-50">
      {/* Settings Toggle Button */}
      <motion.button
        onClick={() => setIsOpen(!isOpen)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className="relative overflow-hidden rounded-full p-2 shadow-2xl"
        style={{
          background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.9))',
          backdropFilter: 'blur(20px)',
        }}
        whileHover={{ 
          scale: 1.1,
          boxShadow: '0 20px 40px -10px rgba(59, 130, 246, 0.4)',
        }}
        whileTap={{ scale: 0.95 }}
        animate={{ 
          rotate: isOpen ? 180 : 0,
        }}
        transition={{ 
          duration: 0.5,
          ease: [0.25, 0.46, 0.45, 0.94],
          type: "spring",
          stiffness: 300,
          damping: 30
        }}
      >
        {/* Animated border */}
        <motion.div
          className="absolute inset-0 rounded-full"
          style={{
            background: 'linear-gradient(45deg, rgba(59, 130, 246, 0.8), rgba(139, 92, 246, 0.8), rgba(6, 182, 212, 0.8))',
            backgroundSize: '300% 300%',
          }}
          animate={{
            backgroundPosition: isHovered || isOpen ? ['0% 50%', '100% 50%', '0% 50%'] : ['0% 50%'],
            opacity: isOpen ? 0.6 : isHovered ? 0.4 : 0.2,
          }}
          transition={{
            backgroundPosition: {
              duration: 3,
              repeat: Infinity,
              ease: "linear",
            },
            opacity: { duration: 0.3 },
          }}
        />
        
        {/* Inner content */}
        <div className="relative z-10">
          <motion.div
            animate={{
              color: isOpen ? '#60a5fa' : isHovered ? '#93c5fd' : '#cbd5e1',
            }}
            transition={{ duration: 0.3 }}
          >
            <Settings className="w-4 h-4" />
          </motion.div>
        </div>
      </motion.button>

      {/* Expanded Settings Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute bottom-12 left-0 overflow-hidden rounded-xl shadow-2xl min-w-64"
            style={{
              background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.95))',
              backdropFilter: 'blur(24px)',
              border: '1px solid rgba(71, 85, 105, 0.3)',
            }}
            initial={{ 
              opacity: 0, 
              y: 30, 
              scale: 0.85,
              filter: 'blur(10px)'
            }}
            animate={{ 
              opacity: 1, 
              y: 0, 
              scale: 1,
              filter: 'blur(0px)'
            }}
            exit={{ 
              opacity: 0, 
              y: 20, 
              scale: 0.9,
              filter: 'blur(8px)'
            }}
            transition={{ 
              duration: 0.4, 
              ease: [0.25, 0.46, 0.45, 0.94],
              type: "spring",
              stiffness: 400,
              damping: 25
            }}
          >
            {/* Subtle animated background */}
            <motion.div
              className="absolute inset-0 opacity-30"
              style={{
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.05), rgba(139, 92, 246, 0.05))',
                backgroundSize: '400% 400%',
              }}
              animate={{
                backgroundPosition: ['0% 0%', '100% 100%', '0% 0%'],
              }}
              transition={{
                duration: 8,
                repeat: Infinity,
                ease: "linear",
              }}
            />

            <div className="relative z-10 p-3">
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <motion.h3 
                  className="text-slate-200 font-medium text-sm"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1, duration: 0.3 }}
                >
                  Controls
                </motion.h3>
                <motion.button
                  onClick={() => setIsOpen(false)}
                  className="text-slate-400 hover:text-slate-200 p-1 rounded-md hover:bg-slate-800/50 transition-all duration-200"
                  whileHover={{ scale: 1.1, rotate: 90 }}
                  whileTap={{ scale: 0.9 }}
                  initial={{ opacity: 0, rotate: -90 }}
                  animate={{ opacity: 1, rotate: 0 }}
                  transition={{ delay: 0.2, duration: 0.3 }}
                >
                  <X className="w-3 h-3" />
                </motion.button>
              </div>

              <div className="space-y-3">
                {/* Zoom Controls */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1, duration: 0.3 }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-slate-300 text-xs font-medium">Zoom</span>
                    <motion.span 
                      className="text-blue-400 text-xs font-mono bg-slate-800/50 px-1.5 py-0.5 rounded"
                      key={zoom}
                      initial={{ scale: 1.2 }}
                      animate={{ scale: 1 }}
                      transition={{ duration: 0.2 }}
                    >
                      {Math.round(zoom * 100)}%
                    </motion.span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <Button
                        onClick={onZoomOut}
                        size="sm"
                        variant="ghost"
                        className="text-slate-400 hover:text-slate-200 hover:bg-slate-800/70 p-1 h-6 w-6 rounded transition-all duration-200"
                      >
                        <ZoomOut className="w-3 h-3" />
                      </Button>
                    </motion.div>
                    <div className="flex-1">
                      <Slider
                        value={[zoom]}
                        onValueChange={(value) => onZoomChange(value[0])}
                        min={0.1}
                        max={5}
                        step={0.1}
                        className="w-full"
                      />
                    </div>
                    <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                      <Button
                        onClick={onZoomIn}
                        size="sm"
                        variant="ghost"
                        className="text-slate-400 hover:text-slate-200 hover:bg-slate-800/70 p-1 h-6 w-6 rounded transition-all duration-200"
                      >
                        <ZoomIn className="w-3 h-3" />
                      </Button>
                    </motion.div>
                  </div>
                </motion.div>

                {/* Time Range Controls */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.15, duration: 0.3 }}
                >
                  <div className="flex items-center space-x-2 mb-2">
                    <Clock className="w-3 h-3 text-slate-400" />
                    <span className="text-slate-300 text-xs font-medium">Timeline</span>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-slate-400">
                      <span>{timeRange[0]}</span>
                      <span>{timeRange[1]}</span>
                    </div>
                    <Slider
                      value={timeRange}
                      onValueChange={(value) => onTimeRangeChange([value[0], value[1]])}
                      min={2000}
                      max={2025}
                      step={1}
                      className="w-full"
                    />
                  </div>
                </motion.div>

                {/* Reset Button */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.17, duration: 0.3 }}
                >
                  <motion.div whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                    <Button
                      onClick={onReset}
                      size="sm"
                      variant="ghost"
                      className="w-full text-slate-400 hover:text-slate-200 hover:bg-slate-800/70 justify-start py-1 h-7 rounded transition-all duration-200 text-xs"
                    >
                      <RotateCcw className="w-3 h-3 mr-2" />
                      Reset
                    </Button>
                  </motion.div>
                </motion.div>

                {/* Layer Visibility */}
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2, duration: 0.3 }}
                >
                  <div className="flex items-center space-x-2 mb-2">
                    <Layers className="w-3 h-3 text-slate-400" />
                    <span className="text-slate-300 text-xs font-medium">Layers</span>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5">
                    {layers.map((layer, index) => (
                      <motion.button
                        key={layer.id}
                        onClick={() => onToggleLayer(layer.id)}
                        className={`flex items-center justify-between p-2 rounded-lg transition-all duration-300 ${
                          visibleLayers.includes(layer.id)
                            ? 'bg-slate-800/70 border border-slate-600/50'
                            : 'bg-slate-800/30 border border-slate-700/30'
                        }`}
                        whileHover={{ 
                          scale: 1.02,
                          backgroundColor: visibleLayers.includes(layer.id) 
                            ? 'rgba(30, 41, 59, 0.8)' 
                            : 'rgba(30, 41, 59, 0.5)'
                        }}
                        whileTap={{ scale: 0.98 }}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.25 + index * 0.05, duration: 0.3 }}
                      >
                        <div className="flex items-center space-x-1.5">
                          <motion.div 
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: layer.color }}
                            animate={{
                              boxShadow: visibleLayers.includes(layer.id) 
                                ? `0 0 6px ${layer.color}40` 
                                : '0 0 0px transparent',
                            }}
                            transition={{ duration: 0.3 }}
                          />
                          <span className="text-slate-300 text-xs font-medium">{layer.label}</span>
                        </div>
                        <motion.div
                          animate={{
                            color: visibleLayers.includes(layer.id) ? '#94a3b8' : '#64748b',
                            scale: visibleLayers.includes(layer.id) ? 1.1 : 1,
                          }}
                          transition={{ duration: 0.2 }}
                        >
                          {visibleLayers.includes(layer.id) ? (
                            <Eye className="w-2.5 h-2.5" />
                          ) : (
                            <EyeOff className="w-2.5 h-2.5" />
                          )}
                        </motion.div>
                      </motion.button>
                    ))}
                  </div>
                </motion.div>

                {/* Selection Info */}
                <AnimatePresence>
                  {selectedNodes.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0, y: -10 }}
                      animate={{ opacity: 1, height: 'auto', y: 0 }}
                      exit={{ opacity: 0, height: 0, y: -10 }}
                      transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
                      className="pt-4 border-t border-slate-700/50"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-slate-300 text-xs font-medium">
                          {selectedNodes.length} selected
                        </span>
                        <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                          <Button
                            onClick={onClearSelection}
                            size="sm"
                            variant="ghost"
                            className="text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 text-xs px-2 py-0.5 h-5 rounded transition-all duration-200"
                          >
                            Clear
                          </Button>
                        </motion.div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Quick Tips */}
                <motion.div 
                  className="pt-2 border-t border-slate-700/50 text-xs text-slate-500 space-y-0.5"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3, duration: 0.3 }}
                >
                  <div className="flex items-center space-x-1.5">
                    <div className="w-0.5 h-0.5 bg-slate-500 rounded-full"></div>
                    <span>Scroll to zoom</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <div className="w-0.5 h-0.5 bg-slate-500 rounded-full"></div>
                    <span>Drag to pan</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <div className="w-0.5 h-0.5 bg-slate-500 rounded-full"></div>
                    <span>Ctrl+click to multi-select</span>
                  </div>
                </motion.div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}