import { motion } from 'motion/react';
import { LucideIcon } from 'lucide-react';
import { useState } from 'react';

interface ActionButtonProps {
  icon: LucideIcon;
  title: string;
  description: string;
  onClick: () => void;
  delay?: number;
  disabled?: boolean;
}

export function ActionButton({ icon: Icon, title, description, onClick, delay = 0, disabled = false }: ActionButtonProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      initial={{ y: 50, opacity: 0, filter: 'blur(10px)' }}
      animate={{ y: 0, opacity: 1, filter: 'blur(0px)' }}
      transition={{ 
        duration: 0.8, 
        delay, 
        ease: [0.25, 0.46, 0.45, 0.94],
        type: "spring",
        stiffness: 200,
        damping: 20
      }}
      whileHover={{ 
        y: -12, 
        scale: 1.04,
        transition: {
          duration: 0.4,
          ease: [0.25, 0.46, 0.45, 0.94],
          type: "spring",
          stiffness: 300,
          damping: 25
        }
      }}
      whileTap={!disabled ? { 
        scale: 0.96,
        transition: { duration: 0.1 }
      } : {}}
      className={`relative ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}
      onClick={() => !disabled && onClick()}
      onMouseEnter={() => !disabled && setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Advanced animated glow background - always visible */}
      <motion.div
        className="absolute inset-0 rounded-2xl blur-xl"
        style={{
          background: 'linear-gradient(45deg, rgba(6, 182, 212, 0.4), rgba(20, 184, 166, 0.4), rgba(59, 130, 246, 0.4), rgba(6, 182, 212, 0.4))',
          backgroundSize: '400% 400%',
        }}
        animate={{
          backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
          opacity: disabled ? 0.1 : isHovered ? 0.8 : 0.4,
          scale: isHovered ? 1.1 : 1,
        }}
        transition={{
          backgroundPosition: {
            duration: 4,
            repeat: Infinity,
            ease: "linear",
          },
          opacity: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
          scale: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] },
        }}
      />

      {/* Cursor-following highlight */}
      <motion.div
        className="absolute inset-0 rounded-2xl opacity-0"
        style={{
          background: 'radial-gradient(circle at center, rgba(6, 182, 212, 0.3) 0%, transparent 50%)',
        }}
        animate={{
          opacity: isHovered ? 0.4 : 0,
        }}
        transition={{ duration: 0.3 }}
      />

      <motion.div 
        className="relative backdrop-blur-xl rounded-2xl p-6 overflow-hidden group border"
        style={{
          background: 'linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(15, 23, 42, 0.7))',
        }}
        animate={{
          backgroundColor: isHovered 
            ? 'rgba(15, 23, 42, 0.9)' 
            : 'rgba(0, 0, 0, 0.8)',
          borderColor: isHovered 
            ? 'rgba(6, 182, 212, 0.5)' 
            : 'rgba(71, 85, 105, 0.3)',
          boxShadow: isHovered 
            ? '0 25px 50px -12px rgba(6, 182, 212, 0.25), 0 0 0 1px rgba(6, 182, 212, 0.1)' 
            : '0 10px 25px -5px rgba(0, 0, 0, 0.3)',
        }}
        transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
      >
        {/* Enhanced shimmer effect */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'linear-gradient(135deg, transparent 30%, rgba(6, 182, 212, 0.15) 50%, transparent 70%)',
            backgroundSize: '300% 300%',
          }}
          animate={{
            backgroundPosition: isHovered ? ['0% 0%', '100% 100%'] : ['0% 0%'],
            opacity: isHovered ? [0, 1, 0] : [0],
          }}
          transition={{
            backgroundPosition: { duration: 2, ease: "easeInOut" },
            opacity: { duration: 2, ease: "easeInOut" },
          }}
        />

        <div className="flex items-center space-x-4 relative z-10">
          <motion.div 
            className="p-3 rounded-xl overflow-hidden relative"
            style={{
              background: 'linear-gradient(135deg, #06b6d4, #14b8a6)',
            }}
            animate={{
              scale: isHovered ? 1.15 : 1,
              boxShadow: isHovered 
                ? '0 15px 35px -5px rgba(6, 182, 212, 0.5), 0 0 20px rgba(6, 182, 212, 0.3)' 
                : '0 5px 15px -3px rgba(6, 182, 212, 0.2)',
            }}
            transition={{ 
              duration: 0.3, 
              ease: [0.25, 0.46, 0.45, 0.94],
              type: "spring",
              stiffness: 300,
              damping: 25
            }}
          >
            {/* Enhanced icon background animation */}
            <motion.div
              className="absolute inset-0"
              style={{
                background: 'linear-gradient(135deg, #22d3ee, #06b6d4, #14b8a6, #22d3ee)',
                backgroundSize: '400% 400%',
              }}
              animate={{
                backgroundPosition: isHovered ? ['0% 50%', '100% 50%', '0% 50%'] : ['0% 50%'],
              }}
              transition={{
                duration: 2.5,
                repeat: Infinity,
                ease: "linear",
              }}
            />
            <motion.div
              animate={{
                rotate: isHovered ? [0, 5, -5, 0] : [0],
              }}
              transition={{
                duration: 0.6,
                ease: "easeInOut",
              }}
            >
              <Icon className="w-6 h-6 text-white relative z-10" />
            </motion.div>
          </motion.div>
          
          <div>
            <motion.h3 
              className="text-cyan-100 mb-1"
              animate={{ 
                color: isHovered ? '#22d3ee' : '#e0f7fa',
              }}
              transition={{ duration: 0.3 }}
            >
              {title}
            </motion.h3>
            <motion.p 
              className="text-cyan-100/70 text-sm"
              animate={{ 
                color: isHovered ? 'rgba(224, 247, 250, 0.9)' : 'rgba(224, 247, 250, 0.7)',
              }}
              transition={{ duration: 0.3 }}
            >
              {description}
            </motion.p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}