import { motion, AnimatePresence } from 'motion/react';
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { ArrowLeft, Download, Filter, ExternalLink } from 'lucide-react';
import { FilterPanel, FilterOptions } from './FilterPanel';
import { GraphControls } from './GraphControls';
import { D3ManyBodyForce } from './D3ForceSimulation';

interface Paper {
  id: string;
  title: string;
  authors: string[];
  year: number;
  citations: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  connections: string[];
  accuracy?: number;
  relevanceToSeed?: number;
  trustScore?: number;
  benchmark?: string;
  methodology?: string[];
  url?: string;
  summary?: string; // Added summary field
  finalX?: number;
  finalY?: number;
}

interface GraphViewProps {
  searchQuery: string;
  onBack: () => void;
}

export function GraphView({ searchQuery, onBack }: GraphViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [hoveredPaper, setHoveredPaper] = useState<Paper | null>(null);
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [zoom, setZoom] = useState(0.6);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [visibleLayers, setVisibleLayers] = useState(['citations', 'collaborations', 'methodology', 'temporal']);
  const animationRef = useRef<number>();
  const [isSimulationRunning, setIsSimulationRunning] = useState(true);
  const [simulationComplete, setSimulationComplete] = useState(false);
  const simulationRef = useRef<{ 
    alpha: number; 
    alphaDecay: number; 
    velocityDecay: number;
    centerX: number;
    centerY: number;
  }>({
    alpha: 1.0,
    alphaDecay: 0.008,
    velocityDecay: 0.25,
    centerX: 600,
    centerY: 450
  });
  
  const d3ForceRef = useRef<D3ManyBodyForce>(new D3ManyBodyForce());

  const [filters, setFilters] = useState<FilterOptions>({
    dateRange: [2000, 2025],
    trustScore: 0, // Single value defaulting to 0% (show all papers)
    relationToSeed: 'all',
    benchmark: 'accuracy',
    sortBy: 'force-directed',  
    searchIn: [],
    visibleLayers: ['citations', 'collaborations', 'methodology', 'temporal'],
  });

  // Enhanced paper data with better titles and summaries
  const [papers] = useState<Paper[]>(() => {
    const researchTopics = [
      {
        title: "Transformer Architectures for Language Understanding",
        summary: "Introduces the transformer architecture using attention mechanisms for neural machine translation, eliminating the need for recurrent networks while achieving superior performance.",
        keywords: ["transformer", "attention", "neural-machine-translation"]
      },
      {
        title: "Bidirectional Encoder Representations for Language Models", 
        summary: "Presents BERT, a bidirectional transformer model that pre-trains deep representations by jointly conditioning on both left and right context, achieving new state-of-the-art results.",
        keywords: ["bidirectional", "pre-training", "language-model"]
      },
      {
        title: "Large-Scale Language Models as Few-Shot Learners",
        summary: "Demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches.",
        keywords: ["few-shot", "large-language-model", "scaling"]
      },
      {
        title: "Deep Residual Networks for Image Classification",
        summary: "Introduces residual learning framework to ease training of networks that are substantially deeper, using skip connections to enable training of very deep neural networks.",
        keywords: ["resnet", "skip-connections", "deep-learning"]
      },
      {
        title: "Convolutional Neural Networks for Visual Recognition",
        summary: "Demonstrates that convolutional neural networks can achieve breakthrough performance on ImageNet classification, significantly outperforming traditional computer vision approaches.",
        keywords: ["cnn", "computer-vision", "image-classification"]
      },
    ];

    const additionalTopics = [
      "Multi-Head Attention Mechanisms", "Self-Supervised Learning Approaches", "Graph Neural Network Architectures",
      "Reinforcement Learning with Policy Gradients", "Generative Adversarial Network Training", "Variational Autoencoder Methods",
      "Meta-Learning for Few-Shot Classification", "Neural Architecture Search Techniques", "Contrastive Learning Frameworks",
      "Knowledge Distillation Methods", "Federated Learning Systems", "Differential Privacy in ML",
      "Adversarial Robustness Training", "Multi-Modal Representation Learning", "Causal Inference in AI",
      "Explainable AI Techniques", "Transfer Learning Strategies", "Active Learning Methods",
      "Curriculum Learning Approaches", "Memory-Augmented Networks", "Attention-Based Models",
      "Sequence-to-Sequence Learning", "Semi-Supervised Learning", "Domain Adaptation Methods",
      "Neural Machine Translation", "Computer Vision Transformers", "Speech Recognition Systems",
      "Recommendation System Architectures", "Time Series Forecasting Models", "Natural Language Generation"
    ];

    const mockPapers: Paper[] = [];

    // Add main papers
    researchTopics.forEach((topic, i) => {
      const year = 2012 + i * 2;
      const angle = (i / researchTopics.length) * 2 * Math.PI;
      const radius = 400 + Math.random() * 400;
      const centerX = 600;
      const centerY = 450;
      
      mockPapers.push({
        id: (i + 1).toString(),
        title: topic.title,
        authors: [`Author${i + 1} et al.`],
        year,
        citations: Math.floor(20000 + Math.random() * 40000),
        x: centerX + Math.cos(angle) * radius + (Math.random() - 0.5) * 200,
        y: centerY + Math.sin(angle) * radius + (Math.random() - 0.5) * 200,
        vx: 0,
        vy: 0,
        connections: [],
        accuracy: 50 + Math.random() * 50, // Ensures good distribution
        relevanceToSeed: 0.7 + Math.random() * 0.3,
        trustScore: Math.floor(70 + Math.random() * 30),
        benchmark: ['Accuracy', 'F1-Score', 'BLEU', 'ROUGE'][Math.floor(Math.random() * 4)],
        methodology: topic.keywords,
        url: `https://arxiv.org/abs/${1706 + i}.${String(Math.floor(Math.random() * 100000)).padStart(5, '0')}`,
        summary: topic.summary,
      });
    });

    // Add additional papers
    additionalTopics.forEach((topic, i) => {
      const year = 2010 + Math.random() * 15;
      const performanceRand = Math.random();
      let accuracy;
      if (performanceRand < 0.25) {
        accuracy = 50 + Math.random() * 20; // 25% low performance (white nodes)
      } else if (performanceRand < 0.65) {
        accuracy = 70 + Math.random() * 15; // 40% medium performance  
      } else {
        accuracy = 85 + Math.random() * 15; // 35% high performance (green nodes)
      }

      const angle = ((i + 5) / (additionalTopics.length + 5)) * 2 * Math.PI;
      const radius = 300 + Math.random() * 600;
      const centerX = 600;
      const centerY = 450;
      
      mockPapers.push({
        id: (i + 6).toString(),
        title: topic,
        authors: [`Researcher${i + 1} et al.`],
        year: Math.floor(year),
        citations: Math.floor(Math.random() * 35000 + 500),
        x: centerX + Math.cos(angle) * radius + (Math.random() - 0.5) * 400,
        y: centerY + Math.sin(angle) * radius + (Math.random() - 0.5) * 400,
        vx: 0,
        vy: 0,
        connections: [],
        accuracy,
        relevanceToSeed: Math.random(),
        trustScore: Math.floor(60 + Math.random() * 40),
        benchmark: ['Accuracy', 'F1-Score', 'BLEU', 'ROUGE', 'Precision', 'Recall'][Math.floor(Math.random() * 6)],
        methodology: ['deep-learning', 'neural-networks', 'machine-learning'],
        url: `https://example.com/paper/${i + 6}`,
        summary: `This research explores ${topic.toLowerCase()}, contributing novel insights to the field through innovative methodologies and comprehensive experimental validation.`,
      });
    });

    // Create connections based on similar topics/years
    mockPapers.forEach(paper => {
      const potentialConnections = mockPapers.filter(otherPaper => 
        otherPaper.id !== paper.id && 
        Math.abs(otherPaper.year - paper.year) <= 3
      );
      
      const numConnections = Math.floor(Math.random() * 3) + 1;
      const selectedConnections = potentialConnections
        .sort(() => Math.random() - 0.5)
        .slice(0, numConnections)
        .map(p => p.id);
      
      paper.connections = selectedConnections;
    });

    return mockPapers;
  });

  const getNodeRadius = () => 25; // Uniform size

  const getNodeOpacity = (paper: Paper) => {
    const currentYear = 2025;
    const age = currentYear - paper.year;
    const maxAge = 20;
    const minOpacity = 0.15;
    const maxOpacity = 1.0;
    
    const ageOpacity = Math.max(minOpacity, maxOpacity - (age / maxAge) * (maxOpacity - minOpacity));
    return ageOpacity;
  };

  const getNodeColor = (paper: Paper) => {
    const performance = paper.accuracy || 70;
    const normalizedPerf = Math.max(0, Math.min(1, (performance - 50) / 50));
    
    const r = Math.round(255 - (255 - 34) * normalizedPerf);
    const g = Math.round(255 - (255 - 197) * normalizedPerf);
    const b = Math.round(255 - (255 - 94) * normalizedPerf);
    
    return { r, g, b };
  };

  const calculateCitationStrength = useCallback((paper1: Paper, paper2: Paper) => {
    const avgCitations = (paper1.citations + paper2.citations) / 2;
    const maxCitations = 60000;
    return Math.min(1, avgCitations / maxCitations);
  }, []);

  const filteredPapers = useMemo(() => {
    return papers.filter(paper => {
      if (paper.year < filters.dateRange[0] || paper.year > filters.dateRange[1]) return false;
      if ((paper.trustScore || 50) < filters.trustScore) return false; // Show papers with performance >= threshold
      if (filters.relationToSeed !== 'all') {
        const relevance = paper.relevanceToSeed || 0.5;
        if (filters.relationToSeed === 'direct' && relevance < 0.8) return false;
        if (filters.relationToSeed === 'indirect' && (relevance < 0.4 || relevance >= 0.8)) return false;
        if (filters.relationToSeed === 'highly-related' && relevance < 0.9) return false;
      }
      return true;
    });
  }, [papers, filters]);

  const getFilteredConnections = useCallback((paper: Paper) => {
    return paper.connections.filter(connId => 
      filteredPapers.some(p => p.id === connId)
    );
  }, [filteredPapers]);

  const applyForces = useCallback(() => {
    const simulation = simulationRef.current;
    if (simulation.alpha < 0.005) {
      setIsSimulationRunning(false);
      setSimulationComplete(true);
      setPapers(prevPapers => prevPapers.map(paper => ({
        ...paper,
        finalX: paper.x,
        finalY: paper.y
      })));
      return;
    }

    const centerForce = 0.0005;
    const repulsionForce = 2500;
    const attractionMultiplier = 0.15;
    const minDistance = 180;

    filteredPapers.forEach(paper => {
      let fx = 0;
      let fy = 0;

      fx += (simulation.centerX - paper.x) * centerForce;
      fy += (simulation.centerY - paper.y) * centerForce;

      filteredPapers.forEach(other => {
        if (paper.id === other.id) return;
        
        const dx = paper.x - other.x;
        const dy = paper.y - other.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < minDistance && distance > 0) {
          const repulsion = repulsionForce / (distance * distance);
          fx += (dx / distance) * repulsion;
          fy += (dy / distance) * repulsion;
        }
      });

      const connections = getFilteredConnections(paper);
      connections.forEach(connectionId => {
        const connectedPaper = filteredPapers.find(p => p.id === connectionId);
        if (connectedPaper) {
          const citationStrength = calculateCitationStrength(paper, connectedPaper);
          const dx = connectedPaper.x - paper.x;
          const dy = connectedPaper.y - paper.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance > 0) {
            const targetDistance = 150 + (1 - citationStrength) * 250;
            const attraction = (distance - targetDistance) * attractionMultiplier * citationStrength * 0.01;
            fx += (dx / distance) * attraction;
            fy += (dy / distance) * attraction;
          }
        }
      });

      paper.vx = (paper.vx + fx) * simulation.velocityDecay;
      paper.vy = (paper.vy + fy) * simulation.velocityDecay;
      
      paper.x += paper.vx;
      paper.y += paper.vy;

      const margin = 150;
      const canvasWidth = 1800;
      const canvasHeight = 1400;
      
      if (paper.x < margin) {
        paper.x = margin;
        paper.vx *= -0.2;
      } else if (paper.x > canvasWidth - margin) {
        paper.x = canvasWidth - margin;
        paper.vx *= -0.2;
      }
      
      if (paper.y < margin) {
        paper.y = margin;
        paper.vy *= -0.2;
      } else if (paper.y > canvasHeight - margin) {
        paper.y = canvasHeight - margin;
        paper.vy *= -0.2;
      }
    });

    simulation.alpha *= (1 - simulation.alphaDecay);
  }, [filteredPapers, getFilteredConnections, calculateCitationStrength]);

  const drawGraph = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.scale(zoom, zoom);
    ctx.translate(-canvas.width / 2 + pan.x, -canvas.height / 2 + pan.y);

    const papersToRender = simulationComplete 
      ? filteredPapers.map(paper => ({ ...paper, x: paper.finalX || paper.x, y: paper.finalY || paper.y }))
      : filteredPapers;

    // Draw connections
    if (filters.visibleLayers.includes('citations')) {
      papersToRender.forEach(paper => {
        const connections = getFilteredConnections(paper);
        connections.forEach(connectionId => {
          const connectedPaper = papersToRender.find(p => p.id === connectionId);
          if (connectedPaper) {
            const citationStrength = calculateCitationStrength(paper, connectedPaper);
            const isHighlighted = selectedPaper?.id === paper.id || selectedPaper?.id === connectedPaper.id ||
                                selectedNodes.includes(paper.id) || selectedNodes.includes(connectedPaper.id);
            
            const thickness = isHighlighted ? 3 + citationStrength * 4 : 1 + citationStrength * 3;
            const alpha = isHighlighted ? 0.9 : 0.4 + citationStrength * 0.4;
            
            if (isHighlighted) {
              const gradient = ctx.createLinearGradient(paper.x, paper.y, connectedPaper.x, connectedPaper.y);
              gradient.addColorStop(0, `rgba(6, 182, 212, ${alpha})`);
              gradient.addColorStop(0.5, `rgba(20, 184, 166, ${alpha})`);
              gradient.addColorStop(1, `rgba(6, 182, 212, ${alpha})`);
              ctx.strokeStyle = gradient;
            } else {
              ctx.strokeStyle = `rgba(6, 182, 212, ${alpha * 0.7})`;
            }
            
            ctx.lineWidth = thickness;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(paper.x, paper.y);
            ctx.lineTo(connectedPaper.x, connectedPaper.y);
            ctx.stroke();
          }
        });
      });
    }

    // Draw nodes
    papersToRender.forEach(paper => {
      const radius = getNodeRadius();
      const opacity = getNodeOpacity(paper);
      const color = getNodeColor(paper);
      const isHovered = hoveredPaper?.id === paper.id;
      const isSelected = selectedPaper?.id === paper.id || selectedNodes.includes(paper.id);

      if (isHovered || isSelected) {
        const glowRadius = radius * (isSelected ? 2.5 : 2);
        const gradient = ctx.createRadialGradient(paper.x, paper.y, 0, paper.x, paper.y, glowRadius);
        
        if (isSelected) {
          gradient.addColorStop(0, `rgba(6, 182, 212, ${opacity * 0.6})`);
          gradient.addColorStop(0.3, `rgba(6, 182, 212, ${opacity * 0.3})`);
          gradient.addColorStop(1, 'rgba(6, 182, 212, 0)');
        } else {
          gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity * 0.4})`);
          gradient.addColorStop(0.4, `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity * 0.2})`);
          gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
        }
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(paper.x, paper.y, glowRadius, 0, 2 * Math.PI);
        ctx.fill();
      }

      // Main node circle
      ctx.beginPath();
      ctx.arc(paper.x, paper.y, radius, 0, 2 * Math.PI);
      
      if (isSelected) {
        ctx.fillStyle = `rgba(6, 182, 212, ${opacity})`;
        ctx.shadowColor = 'rgba(6, 182, 212, 0.5)';
        ctx.shadowBlur = 10;
      } else {
        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      }
      
      ctx.fill();

      // Node border
      if (isHovered) {
        ctx.strokeStyle = `rgba(34, 211, 238, ${Math.min(1, opacity + 0.3)})`;
        ctx.lineWidth = 2;
      } else if (isSelected) {
        ctx.strokeStyle = `rgba(6, 182, 212, ${Math.min(1, opacity + 0.3)})`;
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${Math.min(1, opacity + 0.2)})`;
        ctx.lineWidth = 1;
      }
      ctx.stroke();

      // ALWAYS show text labels - don't hide on zoom out
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
      ctx.shadowBlur = 4;
      ctx.fillStyle = isSelected ? '#22d3ee' : '#e0f7fa';
      ctx.font = isSelected ? 'bold 12px system-ui' : '11px system-ui';
      ctx.textAlign = 'center';
      
      // Show keywords + author name
      const keywords = paper.methodology ? paper.methodology.slice(0, 2).join(', ') : 'ML, AI';
      const authorName = paper.authors[0].split(' ')[0];
      const label = `${keywords} - ${authorName}`;
      ctx.fillText(label, paper.x, paper.y - radius - 15);
      
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    });

    ctx.restore();
  }, [filteredPapers, hoveredPaper, selectedPaper, selectedNodes, zoom, pan, visibleLayers, getFilteredConnections, calculateCitationStrength, simulationComplete]);

  useEffect(() => {
    const animate = () => {
      if (isSimulationRunning && !simulationComplete) {
        applyForces();
      }
      drawGraph();
      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [applyForces, drawGraph, isSimulationRunning, simulationComplete]);

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [dragThreshold] = useState(5);

  // FIXED: More accurate screen to world coordinate transformation
  const screenToWorld = useCallback((screenX: number, screenY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    
    const rect = canvas.getBoundingClientRect();
    const canvasX = screenX - rect.left;
    const canvasY = screenY - rect.top;
    
    // Account for canvas scaling and transforms
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Apply inverse transformations to get world coordinates
    const worldX = ((canvasX - centerX) / zoom) - pan.x + centerX;
    const worldY = ((canvasY - centerY) / zoom) - pan.y + centerY;
    
    return { x: worldX, y: worldY };
  }, [zoom, pan]);

  const findPaperAt = useCallback((worldX: number, worldY: number) => {
    const papersToCheck = simulationComplete 
      ? filteredPapers.map(paper => ({ ...paper, x: paper.finalX || paper.x, y: paper.finalY || paper.y }))
      : filteredPapers;

    return papersToCheck.find(paper => {
      const dx = worldX - paper.x;
      const dy = worldY - paper.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      return distance <= getNodeRadius() + 10; // Increased tolerance for better clicking
    });
  }, [filteredPapers, simulationComplete]);

  const handleCanvasMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    setDragStart({ x: event.clientX, y: event.clientY });
    setIsDragging(false);
    canvas.style.cursor = 'grabbing';
  };

  const handleCanvasMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    if (event.buttons === 1 && !isDragging) {
      const deltaX = Math.abs(event.clientX - dragStart.x);
      const deltaY = Math.abs(event.clientY - dragStart.y);
      
      if (deltaX > dragThreshold || deltaY > dragThreshold) {
        setIsDragging(true);
      }
    }

    if (isDragging && event.buttons === 1) {
      const deltaX = event.clientX - dragStart.x;
      const deltaY = event.clientY - dragStart.y;
      
      setPan(prev => ({
        x: prev.x + deltaX / zoom,
        y: prev.y + deltaY / zoom,
      }));
      
      setDragStart({ x: event.clientX, y: event.clientY });
      return;
    }

    if (!isDragging) {
      const worldPos = screenToWorld(event.clientX, event.clientY);
      const hoveredPaper = findPaperAt(worldPos.x, worldPos.y);
      
      setHoveredPaper(hoveredPaper || null);
      canvas.style.cursor = hoveredPaper ? 'pointer' : 'grab';
    }
  };

  const handleCanvasMouseUp = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!isDragging) {
      const worldPos = screenToWorld(event.clientX, event.clientY);
      const clickedPaper = findPaperAt(worldPos.x, worldPos.y);

      if (clickedPaper) {
        setSelectedPaper(clickedPaper);
        
        if (event.ctrlKey || event.metaKey) {
          setSelectedNodes(prev => 
            prev.includes(clickedPaper.id) 
              ? prev.filter(id => id !== clickedPaper.id)
              : [...prev, clickedPaper.id]
          );
        } else {
          setSelectedNodes([clickedPaper.id]);
        }
      } else {
        if (!event.ctrlKey && !event.metaKey) {
          setSelectedNodes([]);
          setSelectedPaper(null);
        }
      }
    }

    setIsDragging(false);
    canvas.style.cursor = 'grab';
  };

  const handleCanvasMouseLeave = () => {
    setIsDragging(false);
    setHoveredPaper(null);
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.style.cursor = 'default';
    }
  };

  const handleWheel = (event: React.WheelEvent) => {
    event.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    // More sensitive touchpad zoom - less movement needed
    const zoomSpeed = 0.15; // Increased from 0.05 for better sensitivity
    const delta = -event.deltaY * zoomSpeed * 0.01;
    
    // Use exponential zoom for very natural feel
    const zoomMultiplier = Math.exp(delta);
    const newZoom = Math.max(0.2, Math.min(8, zoom * zoomMultiplier));
    
    // Zoom towards mouse cursor for intuitive behavior
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    const mouseOffsetX = mouseX - centerX;
    const mouseOffsetY = mouseY - centerY;
    
    const scaleFactor = newZoom / zoom;
    const newPanX = pan.x - mouseOffsetX * (scaleFactor - 1) / newZoom;
    const newPanY = pan.y - mouseOffsetY * (scaleFactor - 1) / newZoom;

    setZoom(newZoom);
    setPan({ x: newPanX, y: newPanY });
  };

  const resetView = () => {
    setZoom(0.6);
    setPan({ x: 0, y: 0 });
    setSelectedNodes([]);
    setSelectedPaper(null);
    if (simulationComplete) {
      simulationRef.current.alpha = 0.3;
      setIsSimulationRunning(true);
      setSimulationComplete(false);
    }
  };

  const openPaperUrl = (url: string) => {
    if (url) {
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = 1000;
      canvas.height = 700;
      
      simulationRef.current.centerX = 900;
      simulationRef.current.centerY = 700;
    }
  }, []);

  useEffect(() => {
    if (simulationComplete) {
      simulationRef.current.alpha = 0.5;
      setIsSimulationRunning(true);
      setSimulationComplete(false);
    }
  }, [filters]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black relative overflow-hidden">
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
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Data flow lines */}
        <motion.div
          className="absolute inset-0"
          style={{
            background: 'linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.05) 50%, transparent)',
            backgroundSize: '300px 100%',
          }}
          animate={{
            backgroundPosition: ['0% 0%', '100% 0%'],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-cyan-500/20 backdrop-blur-sm bg-black/20 relative z-10">
        <motion.button
          onClick={onBack}
          className="flex items-center space-x-2 text-cyan-100 hover:text-cyan-400 transition-colors"
          whileHover={{ x: -5 }}
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back to Search</span>
        </motion.button>

        <div className="flex items-center space-x-4">
          <span className="text-cyan-100/70">Research Graph for: </span>
          <span className="text-cyan-100 font-medium">"{searchQuery}"</span>
          <span className="text-cyan-400 text-sm">({filteredPapers.length} papers)</span>
        </div>

        <div className="flex items-center space-x-2">
          <motion.button 
            onClick={() => setIsFilterOpen(!isFilterOpen)}
            className={`p-2 transition-colors rounded-lg ${isFilterOpen ? 'text-cyan-400 bg-cyan-500/20' : 'text-cyan-100 hover:text-cyan-400 hover:bg-cyan-500/10'}`}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            <Filter className="w-5 h-5" />
          </motion.button>
          <button className="p-2 text-cyan-100 hover:text-cyan-400 transition-colors rounded-lg hover:bg-cyan-500/10">
            <Download className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="flex relative">
        {/* Graph Canvas */}
        <div className={`flex-1 p-6 transition-all duration-300 ${isFilterOpen ? 'mr-80' : ''}`}>
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="bg-black/30 rounded-xl border border-cyan-500/20 overflow-hidden relative backdrop-blur-sm"
          >
            <canvas
              ref={canvasRef}
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onMouseLeave={handleCanvasMouseLeave}
              onWheel={handleWheel}
              className="w-full h-full cursor-grab select-none"
              style={{ touchAction: 'none' }}
            />

            {/* In-graph Controls positioned relative to canvas */}
            <GraphControls
              zoom={zoom}
              onZoomChange={setZoom}
              onZoomIn={() => setZoom(prev => Math.min(8, prev * 1.4))} // More significant zoom steps
              onZoomOut={() => setZoom(prev => Math.max(0.2, prev * 0.71))} // Inverse of 1.4
              onReset={resetView}
            />
          </motion.div>

          {/* Updated Legend */}
          <div className="mt-4 flex items-center justify-center space-x-8 text-sm text-white/70">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-green-500"></div>
              <span>High Performance</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-white"></div>
              <span>Low Performance</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 rounded-full bg-white opacity-30"></div>
              <span>Older Papers</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="text-white/60 text-xs">📏</div>
              <span>Distance = Relatedness</span>
            </div>
          </div>
        </div>

        {/* Enhanced Sidebar with Summary */}
        {!isFilterOpen && (
          <motion.div
            className="w-80 bg-slate-800/30 border-l border-white/10 p-6 overflow-y-auto max-h-[calc(100vh-120px)]"
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            {selectedPaper ? (
              <div className="space-y-6">
                <h3 className="text-white mb-4">Paper Analysis</h3>
                
                {/* Paper Title and Link */}
                <div>
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="text-blue-400 flex-1 leading-tight">{selectedPaper.title}</h4>
                    {selectedPaper.url && (
                      <button
                        onClick={() => openPaperUrl(selectedPaper.url!)}
                        className="ml-2 p-1 text-white/60 hover:text-blue-400 transition-colors"
                        title="Open paper"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                  <p className="text-white/70 text-sm">
                    {selectedPaper.authors.join(', ')} ({selectedPaper.year})
                  </p>
                </div>

                {/* Summary Section */}
                {selectedPaper.summary && (
                  <div>
                    <h5 className="text-white/90 text-sm mb-2">Summary</h5>
                    <p className="text-white/70 text-sm leading-relaxed bg-white/5 p-3 rounded-lg">
                      {selectedPaper.summary}
                    </p>
                  </div>
                )}

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-white/5 rounded-lg">
                    <div className="text-white font-medium">{Math.floor(selectedPaper.citations).toLocaleString()}</div>
                    <div className="text-white/70 text-xs">Citations</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg">
                    <div className="text-green-400 font-medium">{Math.floor(selectedPaper.trustScore || 0)}%</div>
                    <div className="text-white/70 text-xs">Performance Score</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg">
                    <div className="text-blue-400 font-medium">{selectedPaper.accuracy?.toFixed(1)}%</div>
                    <div className="text-white/70 text-xs">Accuracy</div>
                  </div>
                  <div className="p-3 bg-white/5 rounded-lg">
                    <div className="text-purple-400 font-medium">{2025 - selectedPaper.year}y</div>
                    <div className="text-white/70 text-xs">Age</div>
                  </div>
                </div>

                {/* Connected Papers */}
                <div>
                  <h5 className="text-white/90 text-sm mb-2">Connected Papers</h5>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {getFilteredConnections(selectedPaper).map(connId => {
                      const connectedPaper = filteredPapers.find(p => p.id === connId);
                      if (!connectedPaper) return null;
                      
                      return (
                        <div 
                          key={connId} 
                          className="p-2 bg-white/5 rounded text-xs cursor-pointer hover:bg-white/10 transition-colors"
                          onClick={() => setSelectedPaper(connectedPaper)}
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="text-white/80">{connectedPaper.title.substring(0, 40)}...</div>
                              <div className="text-white/50 flex justify-between mt-1">
                                <span>{connectedPaper.year}</span>
                                <span>{Math.floor(connectedPaper.citations).toLocaleString()} cites</span>
                              </div>
                            </div>
                            {connectedPaper.url && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openPaperUrl(connectedPaper.url!);
                                }}
                                className="ml-2 p-1 text-white/40 hover:text-blue-400 transition-colors"
                              >
                                <ExternalLink className="w-3 h-3" />
                              </button>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div>
                <h3 className="text-white mb-4">Citation Network</h3>
                <div className="space-y-4 text-sm text-white/70">
                  <p>
                    This network visualizes papers with <strong className="text-white/90">uniform node sizes</strong>. 
                    Position and connections are determined by citation relationships.
                  </p>
                  <div className="space-y-2">
                    <div>• <strong>Node transparency:</strong> Paper age (older = more transparent)</div>
                    <div>• <strong>Node color:</strong> Performance score (white → green)</div>
                    <div>• <strong>Distance:</strong> Citation relatedness</div>
                    <div>• <strong>Connection thickness:</strong> Citation volume</div>
                  </div>
                  <div className="pt-4 border-t border-white/10">
                    <div className="text-white/90 mb-2">Navigation</div>
                    <div className="space-y-1 text-xs">
                      <div>• Scroll to zoom in/out</div>
                      <div>• Drag to pan around</div>
                      <div>• Click papers to select</div>
                      <div>• Use filters to refine results</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Filter Panel */}
        <AnimatePresence>
          {isFilterOpen && (
            <FilterPanel
              filters={filters}
              onFiltersChange={setFilters}
              onClose={() => setIsFilterOpen(false)}
            />
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}