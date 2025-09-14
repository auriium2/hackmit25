import { motion } from 'motion/react';
import { Plus, Minus, RotateCcw } from 'lucide-react';

interface GraphControlsProps {
  zoom: number;
  onZoomChange: (zoom: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onReset: () => void;
}

export function GraphControls({
  zoom,
  onZoomChange,
  onZoomIn,
  onZoomOut,
  onReset,
}: GraphControlsProps) {
  return (
    <>
      {/* Intuitive Zoom Controls - Always Visible in Graph Corner */}
      <div className="absolute top-4 right-4 z-40">
        <div className="flex flex-col bg-slate-800/90 backdrop-blur-lg border border-white/10 rounded-lg shadow-lg overflow-hidden">
          {/* Zoom In Button */}
          <motion.button
            onClick={onZoomIn}
            className="w-12 h-12 flex items-center justify-center text-white/70 hover:text-white hover:bg-slate-700/60 transition-all duration-300 border-b border-white/10"
            whileHover={{ 
              scale: 1.05,
              backgroundColor: "rgba(51, 65, 85, 0.7)",
              boxShadow: "0 8px 25px -8px rgba(59, 130, 246, 0.3)"
            }}
            whileTap={{ 
              scale: 0.95,
              transition: { duration: 0.1, ease: "easeInOut" }
            }}
            transition={{ 
              duration: 0.2, 
              ease: [0.25, 0.46, 0.45, 0.94] 
            }}
          >
            <Plus className="w-5 h-5" />
          </motion.button>
          
          {/* Zoom Out Button */}
          <motion.button
            onClick={onZoomOut}
            className="w-12 h-12 flex items-center justify-center text-white/70 hover:text-white hover:bg-slate-700/60 transition-all duration-300"
            whileHover={{ 
              scale: 1.05,
              backgroundColor: "rgba(51, 65, 85, 0.7)",
              boxShadow: "0 8px 25px -8px rgba(59, 130, 246, 0.3)"
            }}
            whileTap={{ 
              scale: 0.95,
              transition: { duration: 0.1, ease: "easeInOut" }
            }}
            transition={{ 
              duration: 0.2, 
              ease: [0.25, 0.46, 0.45, 0.94] 
            }}
          >
            <Minus className="w-5 h-5" />
          </motion.button>
        </div>
      </div>

      {/* Reset Button - Bottom Right Corner */}
      <div className="absolute bottom-4 right-4 z-40">
        <motion.button
          onClick={onReset}
          className="w-12 h-12 bg-slate-800/90 backdrop-blur-lg border border-white/10 rounded-full flex items-center justify-center text-white/70 hover:text-white hover:bg-slate-700/90 transition-all duration-200 shadow-lg"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          title="Reset View"
        >
          <RotateCcw className="w-5 h-5" />
        </motion.button>
      </div>
    </>
  );
}