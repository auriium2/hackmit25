import { motion } from 'motion/react';
import { SearchBox } from './SearchBox';
import { ActionButton } from './ActionButton';
import { FileText, BookOpen, Zap } from 'lucide-react';
import { useState } from 'react';

interface LandingPageProps {
  onSearch: (query: string, type: 'search' | 'review' | 'dive') => void;
}

export function LandingPage({ onSearch }: LandingPageProps) {
  const [searchQuery, setSearchQuery] = useState('');
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black flex flex-col items-center justify-center p-8 relative overflow-hidden">
      {/* Welcome message */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="text-cyan-100/90 mb-2 z-10 relative"
      >
        Welcome, User
      </motion.div>

      {/* RefGraph logo/title */}
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
        className="mb-12 z-10 relative"
      >
        <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-cyan-400 via-blue-400 to-teal-400 bg-clip-text text-transparent text-center">
          RefGraph
        </h1>
        <p className="text-cyan-100/70 text-center">
          <a 
            href="https://arxiv.org/abs/1706.03762" 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-[rgba(167,173,173,1)] hover:text-cyan-300 no-underline decoration-cyan-400/50 hover:decoration-cyan-300 underline-offset-2 transition-colors"
          >
AI-Powered Research, Reimagined
          </a>
        </p>
      </motion.div>

      {/* Search box */}
      <div className="mb-16 w-full max-w-2xl z-10 relative">
        <SearchBox 
          onSearch={(query) => {
            setSearchQuery(query);
            onSearch(query, 'search'); // Trigger search immediately
          }} 
          onChange={(query) => setSearchQuery(query)} // Update searchQuery as user types
        />
      </div>

      {/* Action buttons */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full z-10 relative">
        <ActionButton
          icon={FileText}
          title="Paper Search"
          description="Find and analyze specific research papers"
          onClick={() => onSearch(searchQuery, 'search')}
          disabled={!searchQuery.trim()}
          delay={0.4}
        />
        <ActionButton
          icon={BookOpen}
          title="Literature Review"
          description="Generate comprehensive literature reviews"
          onClick={() => onSearch(searchQuery, 'review')}
          disabled={!searchQuery.trim()}
          delay={0.5}
        />
        <ActionButton
          icon={Zap}
          title="Domain Deep-dive"
          description="Explore research domains in depth"
          onClick={() => onSearch(searchQuery, 'dive')}
          disabled={!searchQuery.trim()}
          delay={0.6}
        />
      </div>

      {/* Enhanced animated background elements - Always active */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {/* Main flowing gradient overlay - Always animating */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.12) 30%, rgba(0, 0, 0, 0.8) 70%)',
          }}
          animate={{
            background: [
              'radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.12) 30%, rgba(0, 0, 0, 0.8) 70%)',
              'radial-gradient(circle at 30% 70%, rgba(20, 184, 166, 0.18), rgba(6, 182, 212, 0.15) 40%, rgba(0, 0, 0, 0.8) 70%)',
              'radial-gradient(circle at 70% 30%, rgba(59, 130, 246, 0.15), rgba(20, 184, 166, 0.12) 35%, rgba(0, 0, 0, 0.8) 70%)',
              'radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.12) 30%, rgba(0, 0, 0, 0.8) 70%)',
            ],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Secondary gradient layer - More subtle */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'conic-gradient(from 0deg at 50% 50%, rgba(6, 182, 212, 0.08) 0deg, rgba(59, 130, 246, 0.06) 120deg, rgba(20, 184, 166, 0.08) 240deg, rgba(6, 182, 212, 0.08) 360deg)',
          }}
          animate={{
            rotate: [0, 360],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear",
          }}
        />

        {/* Floating tech orbs - Always moving */}
        {[...Array(8)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full blur-xl"
            style={{
              width: Math.random() * 300 + 150,
              height: Math.random() * 300 + 150,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: i % 4 === 0 
                ? 'radial-gradient(circle, rgba(6, 182, 212, 0.25), transparent 65%)'
                : i % 4 === 1
                ? 'radial-gradient(circle, rgba(20, 184, 166, 0.22), transparent 65%)'
                : i % 4 === 2
                ? 'radial-gradient(circle, rgba(59, 130, 246, 0.20), transparent 65%)'
                : 'radial-gradient(circle, rgba(34, 211, 238, 0.18), transparent 65%)',
            }}
            animate={{
              x: [0, Math.random() * 400 - 200, 0],
              y: [0, Math.random() * 400 - 200, 0],
              scale: [1, 1.3, 1],
              opacity: [0.4, 0.8, 0.4],
            }}
            transition={{
              duration: Math.random() * 20 + 15,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "easeInOut",
              delay: Math.random() * 5,
            }}
          />
        ))}

        {/* Flowing data streams - Horizontal */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.08) 30%, rgba(20, 184, 166, 0.06) 50%, rgba(6, 182, 212, 0.08) 70%, transparent)',
            backgroundSize: '300px 100%',
          }}
          animate={{
            backgroundPosition: ['0% 0%', '100% 0%'],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "linear",
          }}
        />

        {/* Flowing data streams - Vertical */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'linear-gradient(0deg, transparent, rgba(34, 211, 238, 0.05) 30%, rgba(59, 130, 246, 0.04) 50%, rgba(34, 211, 238, 0.05) 70%, transparent)',
            backgroundSize: '100% 400px',
          }}
          animate={{
            backgroundPosition: ['0% 0%', '0% 100%'],
          }}
          transition={{
            duration: 18,
            repeat: Infinity,
            ease: "linear",
          }}
        />

        {/* Tech grid overlay - Subtle and animated */}
        <motion.div
          className="absolute inset-0 opacity-30"
          style={{
            backgroundImage: `
              linear-gradient(rgba(6, 182, 212, 0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(6, 182, 212, 0.1) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px',
          }}
          animate={{
            opacity: [0.1, 0.3, 0.1],
            backgroundPosition: ['0px 0px', '25px 25px', '0px 0px'],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Pulsing accent points */}
        {[...Array(12)].map((_, i) => (
          <motion.div
            key={`accent-${i}`}
            className="absolute w-1 h-1 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: i % 3 === 0 ? '#06b6d4' : i % 3 === 1 ? '#14b8a6' : '#3b82f6',
            }}
            animate={{
              scale: [0, 1, 0],
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: Math.random() * 3 + 2,
              repeat: Infinity,
              delay: Math.random() * 5,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>
    </div>
  );
}