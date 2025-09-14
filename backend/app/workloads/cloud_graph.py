import asyncio
from collections import deque
from typing import Dict, List, Set, Tuple, Optional, NamedTuple

import networkx as nx
from pyathena import connect
from pyathena.error import ProgrammingError

# todo fix the raw sql
def norm_wid(s: str) -> str:
    s = s.strip()
    if s.startswith("http"):
        s = s.rstrip("/").split("/")[-1]
    return s

def athena_conn(region, s3_staging_dir, workgroup, database):
    return connect(
        s3_staging_dir=s3_staging_dir,
        region_name=region,
        work_group=workgroup,
        schema_name=database,
    )

def table_has(cur, fq_table: str, col: str) -> bool:
    try:
        cur.execute(f"SELECT {col} FROM {fq_table} LIMIT 0")
        return True
    except ProgrammingError:
        return False

def edge_exists(cur, fq_table: str, a: str, b: str) -> bool:
    sql = f"SELECT 1 FROM {fq_table} WHERE src_work=%(a)s AND dst_work=%(b)s LIMIT 1"
    cur.execute(sql, {"a": a, "b": b})
    return cur.fetchone() is not None

def outs(cur, fq_table: str, w: str, limit: int | None) -> Set[str]:
    sql = f"SELECT dst_work FROM {fq_table} WHERE src_work=%(w)s"
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, {"w": w})
    return {r[0] for r in cur.fetchall() if r[0]}

def ins(cur, fq_table: str, w: str, limit: int | None, use_prune: bool) -> Set[str]:
    if use_prune:
        sql = f"""
            SELECT src_work
            FROM {fq_table}
            WHERE dst_pfx=%(pfx)s AND dst_work=%(w)s
        """
        params = {"pfx": w[:3], "w": w}
    else:
        sql = f"SELECT src_work FROM {fq_table} WHERE dst_work=%(w)s"
        params = {"w": w}
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, params)
    return {r[0] for r in cur.fetchall() if r[0]}

def get_neighbors(cur, fq_table: str, w: str, directions: str,
                  limit: int | None, use_prune: bool) -> Tuple[Set[str], List[Tuple[str, str]]]:
    nbrs, edges = set(), []
    if directions in ("out", "both"):
        for n in outs(cur, fq_table, w, limit):
            nbrs.add(n); edges.append((w, n))
    if directions in ("in", "both"):
        for n in ins(cur, fq_table, w, limit, use_prune):
            nbrs.add(n); edges.append((n, w))
    return nbrs, edges

def bidir(cur, fq_table: str, s: str, t: str, max_depth: int, directions: str,
          limit: int | None, use_prune: bool) -> Tuple[nx.DiGraph, List[str]]:
    G = nx.DiGraph()
    s, t = norm_wid(s), norm_wid(t)
    G.add_nodes_from([s, t])
    if s == t:
        return G, [s]

    if directions in ("out", "both") and edge_exists(cur, fq_table, s, t):
        G.add_edge(s, t); return G, [s, t]
    if directions in ("in", "both") and edge_exists(cur, fq_table, t, s):
        G.add_edge(t, s); return G, [s, t]

    q_s, q_t = deque([s]), deque([t])
    dist_s, dist_t = {s: 0}, {t: 0}
    parent_s: Dict[str, str] = {}
    parent_t: Dict[str, str] = {}
    meet = ""

    while q_s or q_t:
        # stop if hop budget exceeded (sum of radii)
        if max(dist_s.values(), default=0) + max(dist_t.values(), default=0) >= max_depth:
            break

        expand_s_side = (len(q_s) <= len(q_t) and q_s) or not q_t
        if expand_s_side and q_s:
            for _ in range(len(q_s)):
                u = q_s.popleft()
                nbrs, edges = get_neighbors(cur, fq_table, u, directions, limit, use_prune)
                for a, b in edges: G.add_edge(a, b)
                for v in nbrs:
                    if v not in dist_s:
                        dist_s[v] = dist_s[u] + 1
                        parent_s[v] = u
                        q_s.append(v)
                        if v in dist_t: meet = v; break
                if meet: break
        elif q_t:
            rev = {"out": "in", "in": "out", "both": "both"}[directions]
            for _ in range(len(q_t)):
                u = q_t.popleft()
                nbrs, edges = get_neighbors(cur, fq_table, u, rev, limit, use_prune)
                for a, b in edges: G.add_edge(a, b)
                for v in nbrs:
                    if v not in dist_t:
                        dist_t[v] = dist_t[u] + 1
                        parent_t[v] = u
                        q_t.append(v)
                        if v in dist_s: meet = v; break
                if meet: break
        if meet: break

    path: List[str] = []
    if meet:
        left = [meet]
        x = meet
        while x in parent_s:
            x = parent_s[x]
            left.append(x)
        left.reverse()

        right: List[str] = []
        x = meet
        while x in parent_t:
            x = parent_t[x]
            right.append(x)

        path = left + right

    return G, path

class GraphResult(NamedTuple):
    """Result from Seed2Graph.get_edges containing the graph data and path."""
    nodes: List[Dict[str, str]]
    edges: List[Dict[str, str]]
    path: List[str] 
    start: str
    end: str


class Seed2Graph:
    """Async class to find citation paths between OpenAlex works using Athena."""

    # Hardcoded AWS configuration
    REGION = "us-east-1"
    DATABASE = "openalex"
    TABLE = "edges"
    WORKGROUP = "primary"
    S3_STAGING_DIR = "s3://your-athena-staging-bucket/queries/"

    def __init__(self, use_prune: bool = True):
        """Initialize the Seed2Graph client.

        Args:
            use_prune: Whether to use dst_pfx pruning for inbound queries (default True)
        """
        self.use_prune = use_prune
        self._fq_table = f"{self.DATABASE}.{self.TABLE}"

    async def get_edges(self,
                       start_work_id: str,
                       end_work_id: str,
                       direction: str = "both",
                       max_depth: int = 4,
                       limit_per_hop: Optional[int] = None) -> GraphResult:
        """Async method to find citation edges between two OpenAlex works.

        Args:
            start_work_id: Starting OpenAlex work ID
            end_work_id: Ending OpenAlex work ID
            direction: Search direction - "out", "in", or "both" (default "both")
            max_depth: Maximum search depth (default 4)
            limit_per_hop: Limit neighbors per node, None for no limit (default None)

        Returns:
            GraphResult containing nodes, edges, path, start, and end
        """
        # Normalize work IDs
        start = norm_wid(start_work_id)
        end = norm_wid(end_work_id)

        # Run the sync search in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        graph, path = await loop.run_in_executor(
            None,
            self._search_sync,
            start, end, direction, max_depth, limit_per_hop
        )

        # Convert to Python data structures
        nodes = [{"id": n} for n in graph.nodes()]
        edges = [{"src": u, "dst": v} for u, v in graph.edges()]

        return GraphResult(
            nodes=nodes,
            edges=edges,
            path=path,
            start=start,
            end=end
        )

    def _search_sync(self, start: str, end: str, direction: str,
                    max_depth: int, limit_per_hop: Optional[int]) -> Tuple[nx.DiGraph, List[str]]:
        """Synchronous search implementation that runs in thread pool."""
        conn = athena_conn(self.REGION, self.S3_STAGING_DIR, self.WORKGROUP, self.DATABASE)
        cur = conn.cursor()

        # Check if pruning is available and enabled
        use_prune = self.use_prune and table_has(cur, self._fq_table, "dst_pfx")

        # Perform bidirectional search
        graph, path = bidir(cur, self._fq_table, start, end, max_depth,
                           direction, limit_per_hop, use_prune)

        cur.close()
        conn.close()
        return graph, path
