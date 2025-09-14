import argparse, json, sys
from collections import deque
from typing import Dict, List, Set, Tuple

import networkx as nx
from pyathena import connect
from pyathena.error import ProgrammingError

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

# --------------------- search ---------------------

def bidir(cur, fq_table: str, s: str, t: str, max_depth: int, directions: str,
          limit: int | None, use_prune: bool) -> Tuple[nx.DiGraph, List[str]]:
    G = nx.DiGraph()
    s, t = norm_wid(s), norm_wid(t)
    G.add_nodes_from([s, t])
    if s == t:
        return G, [s]

    # quick direct check in both directions
    if directions in ("out", "both") and edge_exists(cur, fq_table, s, t):
        G.add_edge(s, t); return G, [s, t]
    if directions in ("in", "both") and edge_exists(cur, fq_table, t, s):
        G.add_edge(t, s); return G, [s, t]

    # bidirectional BFS
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

def main():
    ap = argparse.ArgumentParser(description="OpenAlex connection graph via Athena edges table (dict params)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--directions", choices=["out", "in", "both"], default="both")
    ap.add_argument("--limit-per-hop", type=int, default=0, help="cap neighbors per node (0 = no cap)")
    ap.add_argument("--database", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--workgroup", default="primary")
    ap.add_argument("--s3-staging-dir", required=True)
    ap.add_argument("--no-prune", action="store_true", help="disable dst_pfx pruning for inbound")
    ap.add_argument("--out-json", default="graph.json")
    ap.add_argument("--out-dot", default="graph.dot")
    args = ap.parse_args()

    s = norm_wid(args.start)
    t = norm_wid(args.end)
    limit = args.limit_per_hop if args.limit_per_hop > 0 else None

    conn = athena_conn(args.region, args.s3_staging_dir, args.workgroup, args.database)
    cur = conn.cursor()
    fq_table = f"{args.database}.{args.table}"

    use_prune = (not args.no_prune) and table_has(cur, fq_table, "dst_pfx")

    G, path = bidir(cur, fq_table, s, t, args.max_depth, args.directions, limit, use_prune)

    out = {
        "start": s, "end": t, "max_depth": args.max_depth, "directions": args.directions,
        "path": path,
        "nodes": [{"id": n} for n in G.nodes()],
        "edges": [{"src": u, "dst": v} for u, v in G.edges()],
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    try:
        from networkx.drawing.nx_pydot import write_dot
        write_dot(G, args.out_dot)
    except Exception:
        pass
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
