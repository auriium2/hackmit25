import { motion } from 'motion/react';
import { CheckCircle, Circle, Database, FileText, Brain } from 'lucide-react';
import { useState, useEffect } from 'react';

interface LoadingPageProps {
  onComplete: () => void;
}

export function LoadingPage({ onComplete }: LoadingPageProps) {
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState(0);
  const [pagesIndexed, setPagesIndexed] = useState(0);
  
  const tasks = [
    { icon: Database, label: "Connecting to research databases", completed: false },
    { icon: FileText, label: "Indexing research papers", completed: false },
    { icon: Brain, label: "Analyzing citations and references", completed: false },
    { icon: CheckCircle, label: "Building knowledge graph", completed: false },
  ];

  const [taskStates, setTaskStates] = useState(tasks);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 1000);
          return 100;
        }
        return prev + Math.random() * 3 + 1;
      });

      setPagesIndexed(prev => prev + Math.floor(Math.random() * 50) + 10);

      // Update task completion
      const newProgress = Math.min(progress + Math.random() * 3 + 1, 100);
      const newCurrentTask = Math.floor((newProgress / 100) * tasks.length);
      
      if (newCurrentTask !== currentTask) {
        setCurrentTask(newCurrentTask);
        setTaskStates(prev => prev.map((task, index) => ({
          ...task,
          completed: index < newCurrentTask
        })));
      }
    }, 100);

    return () => clearInterval(interval);
  }, [progress, currentTask, onComplete]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black flex flex-col items-center justify-center p-8 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {/* Main flowing gradient overlay */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.12) 30%, rgba(0, 0, 0, 0.8) 70%)',
          }}
          animate={{
            background: [
              'radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.12) 30%, rgba(0, 0, 0, 0.8) 70%)',
              'radial-gradient(circle at 70% 30%, rgba(20, 184, 166, 0.18), rgba(6, 182, 212, 0.15) 40%, rgba(0, 0, 0, 0.8) 70%)',
              'radial-gradient(circle at 30% 70%, rgba(59, 130, 246, 0.15), rgba(20, 184, 166, 0.12) 35%, rgba(0, 0, 0, 0.8) 70%)',
              'radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.15), rgba(59, 130, 246, 0.12) 30%, rgba(0, 0, 0, 0.8) 70%)',
            ],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Floating tech orbs */}
        {[...Array(4)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full blur-xl"
            style={{
              width: Math.random() * 200 + 100,
              height: Math.random() * 200 + 100,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: i % 2 === 0 
                ? 'radial-gradient(circle, rgba(6, 182, 212, 0.2), transparent 65%)'
                : 'radial-gradient(circle, rgba(20, 184, 166, 0.18), transparent 65%)',
            }}
            animate={{
              x: [0, Math.random() * 200 - 100, 0],
              y: [0, Math.random() * 200 - 100, 0],
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.6, 0.3],
            }}
            transition={{
              duration: Math.random() * 15 + 10,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "easeInOut",
              delay: Math.random() * 5,
            }}
          />
        ))}

        {/* Data stream lines */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.05) 50%, transparent)',
            backgroundSize: '200px 100%',
          }}
          animate={{
            backgroundPosition: ['0% 0%', '100% 0%'],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      </div>

      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="max-w-md w-full relative z-10"
      >
        {/* Loading title */}
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-cyan-100 mb-2">Processing Research Data</h2>
          <p className="text-cyan-100/70">Building your knowledge graph</p>
        </div>

        {/* Progress bar */}
        <div className="mb-8">
          <div className="flex justify-between text-sm text-cyan-100/80 mb-2">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-black/40 rounded-full h-3 overflow-hidden border border-cyan-500/20">
            <motion.div
              className="h-full bg-gradient-to-r from-cyan-500 to-teal-400 rounded-full relative overflow-hidden"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.3 }}
            >
              {/* Animated shimmer on progress bar */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                style={{ backgroundSize: '200% 100%' }}
                animate={{ backgroundPosition: ['0% 0%', '100% 0%'] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
              />
            </motion.div>
          </div>
        </div>

        {/* Pages indexed counter */}
        <motion.div
          className="text-center mb-8 p-4 bg-black/30 rounded-lg backdrop-blur-sm border border-cyan-500/20 relative overflow-hidden"
          animate={{ scale: [1, 1.02, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          {/* Background glow effect */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-teal-500/10"
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <div className="text-2xl font-bold text-cyan-100 relative z-10">
            {pagesIndexed.toLocaleString()}
          </div>
          <div className="text-cyan-100/70 text-sm relative z-10">papers indexed</div>
        </motion.div>

        {/* Task list */}
        <div className="space-y-3">
          {taskStates.map((task, index) => {
            const Icon = task.icon;
            const isActive = index === currentTask;
            const isCompleted = task.completed;
            
            return (
              <motion.div
                key={index}
                className={`flex items-center space-x-3 p-3 rounded-lg transition-all duration-300 backdrop-blur-sm border relative overflow-hidden ${
                  isActive 
                    ? 'bg-cyan-500/20 border-cyan-500/40' 
                    : 'bg-black/20 border-cyan-500/10'
                }`}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                {/* Active task glow */}
                {isActive && (
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-teal-500/20"
                    animate={{ opacity: [0.3, 0.7, 0.3] }}
                    transition={{ duration: 2, repeat: Infinity }}
                  />
                )}
                
                <div className="relative z-10">
                  {isCompleted ? (
                    <CheckCircle className="w-5 h-5 text-teal-400" />
                  ) : isActive ? (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    >
                      <Circle className="w-5 h-5 text-cyan-400" />
                    </motion.div>
                  ) : (
                    <Circle className="w-5 h-5 text-cyan-100/30" />
                  )}
                </div>
                
                <Icon className={`w-4 h-4 relative z-10 ${
                  isCompleted ? 'text-teal-400' : 
                  isActive ? 'text-cyan-400' : 'text-cyan-100/50'
                }`} />
                
                <span className={`text-sm relative z-10 ${
                  isCompleted ? 'text-teal-400' : 
                  isActive ? 'text-cyan-100' : 'text-cyan-100/50'
                }`}>
                  {task.label}
                </span>

                {isActive && (
                  <motion.div
                    className="ml-auto flex space-x-1 relative z-10"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    {[0, 1, 2].map((dot) => (
                      <motion.div
                        key={dot}
                        className="w-1 h-1 bg-cyan-400 rounded-full"
                        animate={{ scale: [0, 1, 0] }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          delay: dot * 0.2,
                        }}
                      />
                    ))}
                  </motion.div>
                )}
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </div>
  );
}