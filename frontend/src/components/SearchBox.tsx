import { motion } from 'motion/react';
import { Search } from 'lucide-react';
import { useState, useRef } from 'react';

interface SearchBoxProps {
  onSearch: (query: string) => void;
  onChange?: (query: string) => void;
  placeholder?: string;
}

export function SearchBox({ onSearch, onChange, placeholder = "What do you want to research today?" }: SearchBoxProps) {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      setMousePosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      });
    }
  };

  return (
    <motion.div
      ref={containerRef}
      className="relative w-full max-w-2xl"
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <motion.div
        className="relative"
        animate={{ 
          y: isFocused ? -12 : isHovered ? -6 : 0, 
          scale: isFocused ? 1.05 : isHovered ? 1.02 : 1 
        }}
        transition={{ 
          duration: 0.4, 
          ease: [0.25, 0.46, 0.45, 0.94],
          type: "spring",
          stiffness: 300,
          damping: 30
        }}
      >
        {/* Base gradient - always visible and animated */}
        <motion.div
          className="absolute inset-0 rounded-2xl blur-sm"
          style={{
            background: 'linear-gradient(45deg, rgba(6, 182, 212, 0.4), rgba(20, 184, 166, 0.4), rgba(59, 130, 246, 0.4), rgba(6, 182, 212, 0.4))',
            backgroundSize: '400% 400%',
          }}
          animate={{
            backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
            opacity: 0.6,
          }}
          transition={{
            backgroundPosition: {
              duration: 3,
              repeat: Infinity,
              ease: "linear",
            },
          }}
        />

        {/* Enhanced cursor-following cyan glow - always partially visible */}
        <motion.div
          className="absolute inset-0 rounded-2xl pointer-events-none"
          style={{
            background: `radial-gradient(400px circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(6, 182, 212, 0.6), rgba(20, 184, 166, 0.3) 30%, transparent 60%)`,
          }}
          animate={{
            opacity: isHovered || isFocused ? 1 : 0.4,
          }}
          transition={{ duration: 0.3 }}
        />

        {/* Intensified hover gradient */}
        <motion.div
          className="absolute inset-0 rounded-2xl blur-sm"
          style={{
            background: 'linear-gradient(45deg, rgba(6, 182, 212, 0.8), rgba(20, 184, 166, 0.8), rgba(59, 130, 246, 0.8), rgba(6, 182, 212, 0.8))',
            backgroundSize: '400% 400%',
          }}
          animate={{
            backgroundPosition: (isHovered || isFocused) ? ['0% 50%', '100% 50%', '0% 50%'] : ['0% 50%'],
            opacity: isFocused ? 0.8 : isHovered ? 0.6 : 0,
            scale: isFocused ? 1.02 : isHovered ? 1.01 : 1,
          }}
          transition={{
            backgroundPosition: {
              duration: 2,
              repeat: Infinity,
              ease: "linear",
            },
            opacity: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
            scale: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
          }}
        />
        
        <form onSubmit={handleSubmit} className="relative">
          <motion.div 
            className="relative backdrop-blur-xl rounded-2xl border overflow-hidden"
            style={{
              background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.95), rgba(15, 23, 42, 0.9))',
            }}
            animate={{
              borderColor: isFocused 
                ? 'rgba(6, 182, 212, 0.6)' 
                : isHovered 
                ? 'rgba(6, 182, 212, 0.3)' 
                : 'rgba(71, 85, 105, 0.3)',
              boxShadow: isFocused 
                ? '0 25px 50px -12px rgba(6, 182, 212, 0.25), 0 0 0 1px rgba(6, 182, 212, 0.1)' 
                : isHovered
                ? '0 20px 35px -8px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(6, 182, 212, 0.05)'
                : '0 10px 25px -5px rgba(0, 0, 0, 0.2)',
            }}
            transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
          >
            {/* Subtle shimmer overlay */}
            <motion.div
              className="absolute inset-0 opacity-0"
              style={{
                background: `linear-gradient(115deg, transparent 40%, rgba(6, 182, 212, 0.1) 50%, transparent 60%)`,
                backgroundSize: '200% 100%',
              }}
              animate={{
                backgroundPosition: isHovered ? ['0% 0%', '100% 0%'] : ['0% 0%'],
                opacity: isHovered ? [0, 1, 0] : [0],
              }}
              transition={{
                backgroundPosition: { duration: 1.5, ease: "easeInOut" },
                opacity: { duration: 1.5, ease: "easeInOut" },
              }}
            />

            <div className="flex items-center px-6 py-4 relative z-10">
              <motion.div
                animate={{
                  scale: isFocused ? 1.15 : isHovered ? 1.05 : 1,
                  color: isFocused ? '#06b6d4' : isHovered ? '#22d3ee' : '#94a3b8',
                }}
                transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
              >
                <Search className="w-5 h-5 mr-4" />
              </motion.div>
              <input
                type="text"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  onChange?.(e.target.value);
                }}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                placeholder={placeholder}
                className="flex-1 bg-transparent outline-none placeholder-slate-400 text-cyan-100 transition-all duration-300"
                style={{ fontSize: '16px' }}
              />
              <motion.button
                type="submit"
                className="ml-4 px-6 py-2 rounded-xl overflow-hidden relative"
                style={{
                  background: 'linear-gradient(135deg, #06b6d4, #14b8a6)',
                }}
                animate={{
                  opacity: query.trim() ? 1 : 0,
                  scale: query.trim() ? 1 : 0.8,
                  pointerEvents: query.trim() ? 'auto' : 'none',
                }}
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: '0 15px 35px -5px rgba(6, 182, 212, 0.5)',
                }}
                whileTap={{ scale: 0.95 }}
                transition={{ 
                  duration: 0.3,
                  ease: [0.25, 0.46, 0.45, 0.94],
                  type: "spring",
                  stiffness: 400,
                  damping: 25
                }}
              >
                <motion.div
                  className="absolute inset-0"
                  style={{
                    background: 'linear-gradient(135deg, #22d3ee, #06b6d4, #14b8a6)',
                    backgroundSize: '300% 300%',
                  }}
                  animate={{
                    backgroundPosition: query.trim() ? ['0% 50%', '100% 50%', '0% 50%'] : ['0% 50%'],
                  }}
                  transition={{
                    duration: 2.5,
                    repeat: Infinity,
                    ease: "linear",
                  }}
                />
                <span className="relative z-10 text-white font-medium">Search</span>
              </motion.button>
            </div>
          </motion.div>
        </form>
      </motion.div>
    </motion.div>
  );
}