// D3 Many-Body Force Implementation
// Adapted from the provided D3 source code

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  index: number;
}

interface QuadNode {
  x: number;
  y: number;
  value: number;
  length?: number;
  data?: Node;
  next?: QuadNode;
  [0]?: QuadNode;
  [1]?: QuadNode;
  [2]?: QuadNode;
  [3]?: QuadNode;
}

// Simple quadtree implementation
function quadtree(nodes: Node[]): QuadNode {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  
  // Find bounds
  for (const node of nodes) {
    if (node.x < x0) x0 = node.x;
    if (node.x > x1) x1 = node.x;
    if (node.y < y0) y0 = node.y;
    if (node.y > y1) y1 = node.y;
  }
  
  // Create root quad
  const root: QuadNode = {
    x: (x0 + x1) / 2,
    y: (y0 + y1) / 2,
    value: 0,
    length: 0
  };
  
  // Add nodes to quadtree (simplified implementation)
  for (const node of nodes) {
    addNode(root, node, x0, y0, x1, y1);
  }
  
  return root;
}

function addNode(quad: QuadNode, node: Node, x0: number, y0: number, x1: number, y1: number) {
  if (!quad.length) {
    if (!quad.data) {
      quad.data = node;
      quad.x = node.x;
      quad.y = node.y;
    } else {
      // Split node
      const oldData = quad.data;
      quad.data = undefined;
      quad.length = 4;
      
      // Add both nodes to appropriate quadrants
      const xm = (x0 + x1) / 2;
      const ym = (y0 + y1) / 2;
      
      insertNodeInQuadrant(quad, oldData, x0, y0, x1, y1, xm, ym);
      insertNodeInQuadrant(quad, node, x0, y0, x1, y1, xm, ym);
    }
  } else {
    // Internal node, add to appropriate quadrant
    const xm = (x0 + x1) / 2;
    const ym = (y0 + y1) / 2;
    insertNodeInQuadrant(quad, node, x0, y0, x1, y1, xm, ym);
  }
}

function insertNodeInQuadrant(quad: QuadNode, node: Node, x0: number, y0: number, x1: number, y1: number, xm: number, ym: number) {
  const right = node.x >= xm;
  const bottom = node.y >= ym;
  const quadrant = (bottom ? 2 : 0) + (right ? 1 : 0);
  
  if (!quad[quadrant as 0 | 1 | 2 | 3]) {
    quad[quadrant as 0 | 1 | 2 | 3] = {
      x: node.x,
      y: node.y,
      value: 0,
      length: 0
    };
  }
  
  const childQuad = quad[quadrant as 0 | 1 | 2 | 3]!;
  const newX0 = right ? xm : x0;
  const newY0 = bottom ? ym : y0;
  const newX1 = right ? x1 : xm;
  const newY1 = bottom ? y1 : ym;
  
  addNode(childQuad, node, newX0, newY0, newX1, newY1);
}

// Jiggle function for handling coincident nodes
function jiggle(random: () => number): number {
  return (random() - 0.5) * 1e-6;
}

export class D3ManyBodyForce {
  private nodes: Node[] = [];
  private random: () => number = Math.random;
  private alpha: number = 1;
  private strength: number = -30;
  private strengths: number[] = [];
  private distanceMin2: number = 1;
  private distanceMax2: number = Infinity;
  private theta2: number = 0.81;

  initialize(nodes: Node[], randomFn?: () => number): void {
    this.nodes = nodes;
    if (randomFn) this.random = randomFn;
    
    this.strengths = new Array(nodes.length);
    for (let i = 0; i < nodes.length; i++) {
      this.strengths[nodes[i].index] = this.strength;
    }
  }

  force(alpha: number): void {
    this.alpha = alpha;
    const tree = quadtree(this.nodes);
    this.visitAfter(tree, this.accumulate.bind(this));
    
    for (let i = 0; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      this.visit(tree, node, this.apply.bind(this));
    }
  }

  private accumulate(quad: QuadNode): void {
    let strength = 0;
    let weight = 0;
    let x = 0;
    let y = 0;

    // For internal nodes, accumulate forces from child quadrants
    if (quad.length) {
      for (let i = 0; i < 4; i++) {
        const q = quad[i as 0 | 1 | 2 | 3];
        if (q && Math.abs(q.value)) {
          const c = Math.abs(q.value);
          strength += q.value;
          weight += c;
          x += c * q.x;
          y += c * q.y;
        }
      }
      quad.x = weight ? x / weight : 0;
      quad.y = weight ? y / weight : 0;
    } else {
      // For leaf nodes
      let q: QuadNode | undefined = quad;
      quad.x = quad.data?.x || quad.x;
      quad.y = quad.data?.y || quad.y;
      
      do {
        if (q.data) {
          strength += this.strengths[q.data.index];
        }
      } while ((q = q.next));
    }

    quad.value = strength;
  }

  private apply(quad: QuadNode, node: Node, x1: number, x2: number): boolean {
    if (!quad.value) return true;

    let x = quad.x - node.x;
    let y = quad.y - node.y;
    const w = x2 - x1;
    let l = x * x + y * y;

    // Apply Barnes-Hut approximation if possible
    if (w * w / this.theta2 < l) {
      if (l < this.distanceMax2) {
        if (x === 0) {
          x = jiggle(this.random);
          l += x * x;
        }
        if (y === 0) {
          y = jiggle(this.random);
          l += y * y;
        }
        if (l < this.distanceMin2) {
          l = Math.sqrt(this.distanceMin2 * l);
        }
        
        node.vx += x * quad.value * this.alpha / l;
        node.vy += y * quad.value * this.alpha / l;
      }
      return true;
    }

    // Otherwise, process points directly
    if (quad.length || l >= this.distanceMax2) return false;

    // Limit forces for very close nodes
    if (quad.data !== node || quad.next) {
      if (x === 0) {
        x = jiggle(this.random);
        l += x * x;
      }
      if (y === 0) {
        y = jiggle(this.random);
        l += y * y;
      }
      if (l < this.distanceMin2) {
        l = Math.sqrt(this.distanceMin2 * l);
      }
    }

    let q: QuadNode | undefined = quad;
    do {
      if (q.data !== node && q.data) {
        const w = this.strengths[q.data.index] * this.alpha / l;
        node.vx += x * w;
        node.vy += y * w;
      }
    } while ((q = q.next));

    return false;
  }

  private visit(quad: QuadNode, node: Node, callback: (quad: QuadNode, node: Node, x1: number, x2: number) => boolean): void {
    const quads: Array<{ quad: QuadNode; x0: number; y0: number; x1: number; y1: number }> = [];
    let x0 = -Infinity, y0 = -Infinity, x1 = Infinity, y1 = Infinity;
    
    quads.push({ quad, x0, y0, x1, y1 });
    
    while (quads.length) {
      const { quad: q, x0: qx0, y0: qy0, x1: qx1, y1: qy1 } = quads.pop()!;
      
      if (!callback(q, node, qx0, qx1)) {
        const xm = (qx0 + qx1) / 2;
        const ym = (qy0 + qy1) / 2;
        
        if (q.length) {
          if (q[3]) quads.push({ quad: q[3], x0: xm, y0: ym, x1: qx1, y1: qy1 });
          if (q[2]) quads.push({ quad: q[2], x0: qx0, y0: ym, x1: xm, y1: qy1 });
          if (q[1]) quads.push({ quad: q[1], x0: xm, y0: qy0, x1: qx1, y1: ym });
          if (q[0]) quads.push({ quad: q[0], x0: qx0, y0: qy0, x1: xm, y1: ym });
        }
      }
    }
  }

  private visitAfter(quad: QuadNode, callback: (quad: QuadNode) => void): void {
    const quads: QuadNode[] = [];
    const next: QuadNode[] = [];
    
    quads.push(quad);
    
    while (quads.length) {
      const q = quads.pop()!;
      if (q.length) {
        for (let i = 0; i < 4; i++) {
          const child = q[i as 0 | 1 | 2 | 3];
          if (child) quads.push(child);
        }
      }
      next.push(q);
    }
    
    while (next.length) {
      callback(next.pop()!);
    }
  }

  setStrength(strength: number): void {
    this.strength = strength;
    if (this.nodes.length) {
      for (let i = 0; i < this.nodes.length; i++) {
        this.strengths[this.nodes[i].index] = strength;
      }
    }
  }

  setDistanceMin(distance: number): void {
    this.distanceMin2 = distance * distance;
  }

  setDistanceMax(distance: number): void {
    this.distanceMax2 = distance * distance;
  }

  setTheta(theta: number): void {
    this.theta2 = theta * theta;
  }
}