# file: connect_openalex_seeds.py
import os
import ray
import ray.data as rd
import xxhash
from collections import defaultdict, deque

# ---------- CONFIG ----------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # make sure creds/region are set in env/instance profile

BY_SRC = "s3://supermassive22/openalex_edges_by_src/"
BY_DST = "s3://supermassive22/openalex_edges_by_dst/"

OUT_NODES = "s3://supermassive22/openalex_result/nodes/"
OUT_EDGES = "s3://supermassive22/openalex_result/edges/"

N_SHARDS = 256                           # must match what you built in Athena
SEEDS = ["W1471265382", "W101137784"]    # <-- put your seeds here (2+ ok)
MAX_DEPTH = 2
FANOUT_CAP = 500                        # limit explosive hubs per level (None to disable)
UNDIRECTED = True                        # explore both directions
# -----------------------------

def shard_id(wid: str, n=N_SHARDS) -> int:
    # match Athena: abs(xxhash64(utf8)) % N
    h = xxhash.xxh64(wid.encode("utf-8")).intdigest()
    if h < 0:
        h = -h
    return h % n

def shard_paths(base_prefix: str, shard_col: str, shard_values: set[int]) -> list[str]:
    # Parquet partition layout is .../<shard_col>=<value>/...
    return [f"{base_prefix}{shard_col}={v}/" for v in shard_values]

def neighbors_for_ids(base_prefix: str, shard_col: str, key_col: str, nbr_col: str,
                      ids: list[str], cap: int | None):
    """
    Read only needed shard folders, then filter to the exact ids and return pairs (key_id, neighbor).
    """
    ids_set = set(ids)
    shards = {shard_id(i) for i in ids_set}
    paths = shard_paths(base_prefix, shard_col, shards)
    if not paths:
        return []

    ds = rd.read_parquet(paths)  # partition-pruned read (only the shards we need)
    # small broadcast filter
    ds = ds.filter(lambda r: r[key_col] in ids_set)
    if cap:
        ds = ds.groupby(key_col).limit_per_group(cap)
    return ds.select_columns([key_col, nbr_col]).to_pandas().itertuples(index=False, name=None)

def run_bi_bfs(seeds: list[str]):
    ray.init()  # or ray.init(address="auto") on a cluster

    parents: dict[str, tuple[str | None, str, str]] = {} # node -> (parent, origin_seed, dir)
    visited: dict[str, str] = {}                         # node -> origin_seed
    q = deque()

    for s in seeds:
        visited[s] = s
        parents[s] = (None, s, "seed")
        q.append((s, s, 0))

    # poor-man’s disjoint set for “are seeds connected yet?”
    comp = {s: s for s in seeds}
    def find(x):
        while comp[x] != x:
            comp[x] = comp[comp[x]]
            x = comp[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: comp[rb] = ra

    all_connected = False
    depth = 0
    while q and depth <= MAX_DEPTH and not all_connected:
        # gather level by origin
        level_by_origin = defaultdict(list)
        for _ in range(len(q)):
            node, origin, d = q.popleft()
            level_by_origin[origin].append(node)

        next_items = []

        for origin, nodes in level_by_origin.items():
            # FORWARD (by_src): src_id -> dst neighbors
            for (src, dst) in neighbors_for_ids(
                BY_SRC, "src_shard", "src_id", "dst_id", nodes, FANOUT_CAP
            ):
                if dst not in visited:
                    visited[dst] = origin
                    parents[dst] = (src, origin, "fwd")
                    next_items.append((dst, origin))
                elif visited[dst] != origin:
                    union(visited[dst], origin)

            if UNDIRECTED:
                # BACKWARD (by_dst): dst_id -> citing src neighbors
                for (dst, src) in neighbors_for_ids(
                    BY_DST, "dst_shard", "dst_id", "src_id", nodes, FANOUT_CAP
                ):
                    if src not in visited:
                        visited[src] = origin
                        parents[src] = (dst, origin, "bwd")
                        next_items.append((src, origin))
                    elif visited[src] != origin:
                        union(visited[src], origin)

        for n, o in next_items:
            q.append((n, o, depth + 1))
        depth += 1

        roots = {find(s) for s in seeds}
        all_connected = (len(roots) == 1)

    # backtrack minimal connecting forest
    seeds_set = set(seeds)
    result_nodes = set(seeds_set)
    result_edges = set()

    touched_by = defaultdict(set)
    for n, o in visited.items():
        touched_by[n].add(o)
    connectors = [n for n, os in touched_by.items() if len(os) > 1] or list(seeds_set)

    def walk_back(n: str):
        while n is not None and n in parents:
            p, origin, _ = parents[n]
            result_nodes.add(n)
            if p is not None:
                result_nodes.add(p)
                result_edges.add((p, n))
            n = p

    for c in connectors:
        walk_back(c)

    # write tiny result graph
    rd.from_items([{"id": n} for n in result_nodes]).write_parquet(OUT_NODES)
    rd.from_items([{"src_id": s, "dst_id": d} for (s, d) in result_edges]).write_parquet(OUT_EDGES)

    print(f"✅ Subgraph written: {len(result_nodes)} nodes, {len(result_edges)} edges")
    print("Nodes path:", OUT_NODES)
    print("Edges path:", OUT_EDGES)

if __name__ == "__main__":
    run_bi_bfs(SEEDS)
