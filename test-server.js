// Simple test server to verify API integration
const http = require('http');
const url = require('url');

// Track processing jobs
const jobs = new Map();

const server = http.createServer((req, res) => {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  const parsedUrl = url.parse(req.url, true);

  if (parsedUrl.pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'healthy' }));
  }
  else if (parsedUrl.pathname === '/query' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      const data = JSON.parse(body);
      const systemid = 'test-' + Date.now();
      console.log('🎯 LIVE BACKEND: Query processed:', data.query, '| System ID:', systemid, '| STATUS: ACTIVE!');

      // Start processing simulation
      startProcessingSimulation(systemid, data.query);

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        systemid: systemid,
        status: 'ok'
      }));
    });
  }
  else if (parsedUrl.pathname === '/check_status') {
    const systemid = parsedUrl.query.systemid;
    const job = jobs.get(systemid);

    if (!job) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        systemid: systemid,
        status: 'fail: job not found'
      }));
      return;
    }

    console.log('🔍 Status check for', systemid + ':', job.status);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      systemid: systemid,
      status: job.status
    }));
  }
  else if (parsedUrl.pathname === '/pull_final_graph') {
    const systemid = parsedUrl.query.systemid;
    const job = jobs.get(systemid);

    // Generate dynamic graph based on the original query
    const query = job ? job.query : 'research';
    const graphData = generateMockGraph(query, systemid);

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(graphData));
  }
  else if (parsedUrl.pathname === '/get_details') {
    const paperId = parsedUrl.query.openalexid;
    console.log('📄 Paper details requested for:', paperId);

    // Mock OpenAlex data based on paper ID
    const mockPaperData = {
      'paper1': {
        openalex_id: 'paper1',
        title: 'Attention Is All You Need',
        abstract: 'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
        year: 2017,
        citations: 70000,
        doi: '10.5555/3295222.3295349',
        url: 'https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html',
        authors: ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 'Jakob Uszkoreit', 'Llion Jones', 'Aidan N. Gomez', 'Lukasz Kaiser', 'Illia Polosukhin'],
        venue: 'Neural Information Processing Systems'
      },
      'paper2': {
        openalex_id: 'paper2',
        title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
        abstract: 'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.',
        year: 2018,
        citations: 45000,
        doi: '10.18653/v1/N19-1423',
        url: 'https://aclanthology.org/N19-1423/',
        authors: ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee', 'Kristina Toutanova'],
        venue: 'NAACL-HLT'
      },
      'paper3': {
        openalex_id: 'paper3',
        title: 'Language Models are Few-Shot Learners',
        abstract: 'Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions.',
        year: 2020,
        citations: 25000,
        doi: '10.5555/3495724.3495883',
        url: 'https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html',
        authors: ['Tom B. Brown', 'Benjamin Mann', 'Nick Ryder', 'Melanie Subbiah', 'Jared Kaplan'],
        venue: 'Neural Information Processing Systems'
      }
    };

    const paperData = mockPaperData[paperId] || {
      openalex_id: paperId,
      title: 'Unknown Paper',
      abstract: 'Paper details not found in test data.',
      year: 2024,
      citations: 0,
      doi: null,
      url: null,
      authors: ['Unknown Author'],
      venue: 'Unknown Venue'
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(paperData));
  }
  else if (parsedUrl.pathname === '/analyze_paper') {
    const paperId = parsedUrl.query.paper_id;
    console.log('🚀 BACKEND LIVE: Full paper analysis for', paperId, '- API WORKING!');

    // Different analysis data per paper
    const analysisData = {
      'paper1': {
        title: 'Attention Is All You Need',
        abstract: 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.',
        contributions: [
          'Introduced the Transformer architecture',
          'Eliminated recurrence in favor of attention',
          'Achieved state-of-the-art translation results',
          'Enabled parallel training'
        ],
        methodology: ['Multi-head attention mechanism', 'Positional encoding', 'Layer normalization', 'Residual connections'],
        impact_score: 9.8,
        relevance_score: 9.5
      },
      'paper2': {
        title: 'BERT: Pre-training Deep Bidirectional Representations',
        abstract: 'We introduce BERT, designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.',
        contributions: [
          'Introduced bidirectional pre-training',
          'Achieved new SOTA on GLUE benchmark',
          'Demonstrated transfer learning effectiveness',
          'Influenced numerous follow-up models'
        ],
        methodology: ['Masked language modeling', 'Next sentence prediction', 'Bidirectional encoding', 'Fine-tuning approach'],
        impact_score: 9.6,
        relevance_score: 9.2
      },
      'paper3': {
        title: 'GPT-3: Language Models are Few-Shot Learners',
        abstract: 'We train GPT-3, an autoregressive language model with 175 billion parameters, demonstrating strong few-shot learning capabilities.',
        contributions: [
          'Demonstrated emergent few-shot learning',
          'Scaled to 175B parameters',
          'Showed in-context learning capabilities',
          'Influenced large language model development'
        ],
        methodology: ['Autoregressive generation', 'Few-shot prompting', 'Massive scale training', 'In-context learning'],
        impact_score: 9.4,
        relevance_score: 8.9
      }
    };

    const paperData = analysisData[paperId] || analysisData['paper1'];

    const mockAnalysis = {
      paper_id: paperId,
      openalex_data: {
        openalex_id: paperId,
        title: paperData.title,
        abstract: paperData.abstract,
        year: paperId === 'paper1' ? 2017 : paperId === 'paper2' ? 2018 : 2020,
        citations: paperId === 'paper1' ? 70000 : paperId === 'paper2' ? 45000 : 25000,
        doi: `10.5555/${paperId}.example`,
        url: `https://proceedings.neurips.cc/paper/${paperId}/`,
        authors: paperId === 'paper1' ? ['Vaswani et al.'] : paperId === 'paper2' ? ['Devlin et al.'] : ['Brown et al.'],
        venue: 'NeurIPS'
      },
      benchmark_metrics: {
        accuracy: paperId === 'paper1' ? 92.5 : paperId === 'paper2' ? 88.9 : 85.2,
        f1_score: paperId === 'paper1' ? 89.3 : paperId === 'paper2' ? 91.1 : 87.4,
        bleu_score: paperId === 'paper1' ? 34.2 : paperId === 'paper2' ? null : 28.9,
        perplexity: paperId === 'paper1' ? 1.24 : paperId === 'paper2' ? 1.18 : 1.32,
        inference_time: paperId === 'paper1' ? 0.045 : paperId === 'paper2' ? 0.052 : 0.089,
        model_size: paperId === 'paper1' ? 65.0 : paperId === 'paper2' ? 340.0 : 175000.0,
        dataset: paperId === 'paper1' ? 'WMT 2014 EN-DE' : paperId === 'paper2' ? 'GLUE Benchmark' : 'Common Crawl',
        benchmark_suite: paperId === 'paper1' ? 'Machine Translation' : paperId === 'paper2' ? 'GLUE' : 'Few-shot Learning',
        evaluation_date: paperId === 'paper1' ? '2017-06-12' : paperId === 'paper2' ? '2018-10-11' : '2020-05-28'
      },
      analysis_summary: {
        key_contributions: paperData.contributions,
        methodology: paperData.methodology,
        strengths: paperId === 'paper1' ?
          ['Highly parallelizable', 'Better long-range dependencies', 'Strong empirical results', 'Computational efficiency'] :
          paperId === 'paper2' ?
          ['Bidirectional context', 'Transfer learning', 'SOTA results', 'Widely applicable'] :
          ['Few-shot learning', 'Massive scale', 'Emergent abilities', 'General purpose'],
        limitations: paperId === 'paper1' ?
          ['Quadratic complexity', 'Requires large data', 'Memory intensive'] :
          paperId === 'paper2' ?
          ['Requires fine-tuning', 'Computationally expensive', 'Limited context length'] :
          ['Enormous compute requirements', 'Hallucination issues', 'Limited reasoning'],
        impact_score: paperData.impact_score,
        relevance_score: paperData.relevance_score
      },
      processing_metadata: {
        extraction_confidence: 0.96,
        last_updated: '2024-01-15T10:30:00Z',
        data_sources: ['OpenAlex', 'Semantic Scholar', 'ArXiv']
      }
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(mockAnalysis));
  }
  else if (parsedUrl.pathname === '/get_benchmarks') {
    const paperId = parsedUrl.query.paper_id;
    console.log('⚡ REAL-TIME: Benchmark metrics loaded for', paperId, '- BACKEND RESPONDING!');

    const mockMetrics = {
      accuracy: paperId === 'paper1' ? 92.5 : paperId === 'paper2' ? 88.9 : 85.2,
      f1_score: paperId === 'paper1' ? 89.3 : paperId === 'paper2' ? 91.1 : 87.4,
      bleu_score: paperId === 'paper1' ? 34.2 : paperId === 'paper2' ? null : 28.9,
      rouge_score: paperId === 'paper1' ? 58.7 : paperId === 'paper2' ? 62.3 : 55.1,
      perplexity: paperId === 'paper1' ? 1.24 : paperId === 'paper2' ? 1.18 : 1.32,
      inference_time: paperId === 'paper1' ? 0.045 : paperId === 'paper2' ? 0.052 : 0.089,
      model_size: paperId === 'paper1' ? 65.0 : paperId === 'paper2' ? 340.0 : 175000.0,
      training_time: paperId === 'paper1' ? 3.2 : paperId === 'paper2' ? 12.5 : 1200.0,
      dataset: paperId === 'paper1' ? 'WMT 2014 EN-DE' : paperId === 'paper2' ? 'GLUE Benchmark' : 'Common Crawl',
      benchmark_suite: paperId === 'paper1' ? 'Machine Translation' : paperId === 'paper2' ? 'GLUE' : 'Few-shot Learning',
      evaluation_date: paperId === 'paper1' ? '2017-06-12' : paperId === 'paper2' ? '2018-10-11' : '2020-05-28'
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(mockMetrics));
  }
  else {
    res.writeHead(404);
    res.end('Not found');
  }
});

// Generate mock graph data based on search query
function generateMockGraph(query, systemid) {
  console.log('📊 Generating dynamic graph for query:', query);

  // Query-specific research topics
  const getTopicsForQuery = (searchQuery) => {
    const lowerQuery = searchQuery.toLowerCase();

    if (lowerQuery.includes('machine learning') || lowerQuery.includes('ml')) {
      return [
        { title: `Machine Learning Foundations for ${searchQuery}`, keywords: ['ml', 'supervised-learning', 'algorithms'] },
        { title: `Deep Learning Approaches in ${searchQuery}`, keywords: ['deep-learning', 'neural-networks', 'backpropagation'] },
        { title: `Reinforcement Learning Applications for ${searchQuery}`, keywords: ['reinforcement-learning', 'policy-gradient', 'reward-function'] },
        { title: `Unsupervised Learning Methods in ${searchQuery}`, keywords: ['clustering', 'dimensionality-reduction', 'autoencoder'] },
        { title: `Meta-Learning and Few-Shot Learning for ${searchQuery}`, keywords: ['meta-learning', 'few-shot', 'transfer-learning'] }
      ];
    } else if (lowerQuery.includes('transformer') || lowerQuery.includes('attention')) {
      return [
        { title: 'Attention Is All You Need', keywords: ['transformer', 'attention', 'self-attention'] },
        { title: 'BERT: Pre-training Deep Bidirectional Transformers', keywords: ['bert', 'bidirectional', 'pre-training'] },
        { title: 'GPT: Generative Pre-training of a Language Model', keywords: ['gpt', 'generative', 'autoregressive'] },
        { title: 'Vision Transformer: An Image is Worth 16x16 Words', keywords: ['vision-transformer', 'image-patches', 'computer-vision'] },
        { title: 'Switch Transformer: Scaling to Trillion Parameter Models', keywords: ['switch-transformer', 'mixture-of-experts', 'scaling'] }
      ];
    } else if (lowerQuery.includes('nlp') || lowerQuery.includes('language')) {
      return [
        { title: `Natural Language Processing for ${searchQuery}`, keywords: ['nlp', 'tokenization', 'language-model'] },
        { title: `Sentiment Analysis in ${searchQuery}`, keywords: ['sentiment-analysis', 'classification', 'opinion-mining'] },
        { title: `Named Entity Recognition for ${searchQuery}`, keywords: ['ner', 'entity-extraction', 'sequence-labeling'] },
        { title: `Question Answering Systems in ${searchQuery}`, keywords: ['qa', 'reading-comprehension', 'information-retrieval'] },
        { title: `Machine Translation Approaches for ${searchQuery}`, keywords: ['machine-translation', 'sequence-to-sequence', 'alignment'] }
      ];
    } else {
      // Generic topics based on query
      return [
        { title: `Novel Approaches to ${searchQuery}`, keywords: ['methodology', 'innovation', searchQuery.toLowerCase().replace(/\s+/g, '-')] },
        { title: `Comprehensive Survey of ${searchQuery}`, keywords: ['survey', 'review', searchQuery.toLowerCase().replace(/\s+/g, '-')] },
        { title: `Deep Learning Applications in ${searchQuery}`, keywords: ['deep-learning', 'applications', searchQuery.toLowerCase().replace(/\s+/g, '-')] },
        { title: `Empirical Studies on ${searchQuery}`, keywords: ['empirical', 'experimental', searchQuery.toLowerCase().replace(/\s+/g, '-')] },
        { title: `Future Directions for ${searchQuery} Research`, keywords: ['future-work', 'research-direction', searchQuery.toLowerCase().replace(/\s+/g, '-')] }
      ];
    }
  };

  const topics = getTopicsForQuery(query);

  // Generate 8-15 papers for more realistic graph
  const numPapers = Math.floor(Math.random() * 8) + 8;
  const nodes = [];
  const edges = [];

  // Create main papers from topics
  topics.forEach((topic, i) => {
    const year = 2015 + Math.floor(Math.random() * 10);
    const paperId = `paper${i + 1}`;

    nodes.push({
      id: paperId,
      label: topic.title,
      data: {
        id: paperId,
        title: topic.title,
        authors: [`Researcher${i + 1} et al.`],
        year: year,
        abstract: `This paper presents ${topic.title.toLowerCase()}, contributing novel insights to the field through innovative methodologies and comprehensive experimental validation.`,
        citations: Math.floor(Math.random() * 15000) + 1000,
        cluster: getClusterFromQuery(query),
        confidence: 80 + Math.floor(Math.random() * 20),
        summary: `Comprehensive research on ${topic.title.toLowerCase()}, providing significant contributions to the ${query.toLowerCase()} domain.`,
        metrics: {},
        embedding: []
      }
    });
  });

  // Generate additional papers to reach target number
  for (let i = topics.length; i < numPapers; i++) {
    const year = 2010 + Math.floor(Math.random() * 15);
    const paperId = `paper${i + 1}`;

    nodes.push({
      id: paperId,
      label: `Advanced ${query} Techniques and Applications`,
      data: {
        id: paperId,
        title: `Advanced ${query} Techniques and Applications ${i - topics.length + 1}`,
        authors: [`Author${i + 1} et al.`],
        year: year,
        abstract: `This work explores advanced techniques in ${query.toLowerCase()}, presenting novel algorithms and methodologies.`,
        citations: Math.floor(Math.random() * 8000) + 200,
        cluster: getClusterFromQuery(query),
        confidence: 70 + Math.floor(Math.random() * 25),
        summary: `Research focusing on ${query.toLowerCase()} with emphasis on practical applications and theoretical foundations.`,
        metrics: {},
        embedding: []
      }
    });
  }

  // Generate edges (citations) between papers
  nodes.forEach((node, i) => {
    const numConnections = Math.floor(Math.random() * 3) + 1;
    const potentialTargets = nodes.filter((otherNode, j) =>
      j !== i && Math.abs(otherNode.data.year - node.data.year) <= 4
    );

    const selectedTargets = potentialTargets
      .sort(() => Math.random() - 0.5)
      .slice(0, numConnections);

    selectedTargets.forEach(target => {
      const edgeId = `citation-${node.id}-${target.id}`;
      if (!edges.find(e => e.id === edgeId || e.id === `citation-${target.id}-${node.id}`)) {
        edges.push({
          id: edgeId,
          source: node.id,
          target: target.id,
          type: 'citation',
          data: {
            relation: 'cites',
            confidence: 0.7 + Math.random() * 0.3
          }
        });
      }
    });
  });

  return {
    systemid: systemid,
    graph: {
      nodes: nodes,
      edges: edges,
      metadata: {
        query: query,
        total_papers: nodes.length,
        processing_time: 15.0 + Math.random() * 10,
        insights: {
          dominant_cluster: getClusterFromQuery(query),
          average_year: Math.round(nodes.reduce((sum, n) => sum + n.data.year, 0) / nodes.length),
          total_citations: nodes.reduce((sum, n) => sum + n.data.citations, 0)
        },
        seed_papers: nodes.slice(0, Math.min(3, nodes.length)).map(n => n.id)
      }
    }
  };
}

function getClusterFromQuery(query) {
  const lowerQuery = query.toLowerCase();
  if (lowerQuery.includes('nlp') || lowerQuery.includes('language')) return 'Natural Language Processing';
  if (lowerQuery.includes('vision') || lowerQuery.includes('image')) return 'Computer Vision';
  if (lowerQuery.includes('ml') || lowerQuery.includes('machine learning')) return 'Machine Learning';
  if (lowerQuery.includes('ai') || lowerQuery.includes('artificial')) return 'Artificial Intelligence';
  if (lowerQuery.includes('data')) return 'Data Science';
  return 'Computer Science';
}

// Simulate realistic processing with status updates
function startProcessingSimulation(systemid, query) {
  const statuses = [
    { status: 'finding seed papers', duration: 3000 },
    { status: 'building citation graph', duration: 4000 },
    { status: 'analyzing papers with AI agents', duration: 5000 },
    { status: 'generating insights', duration: 3000 },
    { status: 'finalizing graph', duration: 2000 },
    { status: 'done', duration: 0 }
  ];

  // Initialize job
  jobs.set(systemid, {
    query: query,
    status: 'started',
    startTime: Date.now()
  });

  console.log('🚀 Starting processing for:', systemid);

  // Progress through statuses
  let currentStep = 0;

  function progressToNextStatus() {
    if (currentStep >= statuses.length) return;

    const currentStatus = statuses[currentStep];
    const job = jobs.get(systemid);

    if (job) {
      job.status = currentStatus.status;
      jobs.set(systemid, job);
      console.log('⏳ Status update:', systemid, '→', currentStatus.status);
    }

    currentStep++;

    if (currentStep < statuses.length) {
      setTimeout(progressToNextStatus, currentStatus.duration);
    } else {
      console.log('✅ Processing complete for:', systemid);
    }
  }

  // Start the progression
  setTimeout(progressToNextStatus, 1000); // Start after 1 second
}

server.listen(8000, () => {
  console.log('🧪 Test server running on http://localhost:8000');
  console.log('📡 Frontend should now show "Live Data" mode!');
  console.log('⏱️  Processing simulation: ~17 seconds total');
});