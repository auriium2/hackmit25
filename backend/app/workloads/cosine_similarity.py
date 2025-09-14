import requests
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# ----------- CONFIG -----------
OPENALEX_URL = "https://api.openalex.org/works/"
SLEEP = 0.2  # politeness for API requests
# Weights for similarity computation
ALPHA, BETA, GAMMA = 0.6, 0.3, 0.1
# -------------------------------

# ========== HELPERS ==========

def fetch_work(wid, fields=None):
    """Fetch a single work from OpenAlex."""
    params = {}
    if fields:
        params['fields'] = ",".join(fields)
    r = requests.get(OPENALEX_URL + wid, params=params)
    r.raise_for_status()
    return r.json()

# ========== GRAPH CONSTRUCTION ==========

def build_closed_citation_graph(work_ids):
    """Build a citation graph restricted to a given set of OpenAlex IDs."""
    fields = ["id","title","abstract","referenced_works"]
    G = nx.DiGraph()

    # Fetch all works in the set
    works = {}
    for wid in work_ids:
        w = fetch_work(wid, fields=fields)
        works[wid] = w
        G.add_node(
            wid,
            title=w.get("title",""),
            abstract=w.get("abstract",""),
            references=set(w.get("referenced_works", [])),
            citers=set()
        )

    # Add edges only if both citing and cited papers are in the set
    for wid, w in works.items():
        for rid in w.get("referenced_works", []):
            if rid in works:
                G.add_edge(wid, rid)
                G.nodes[rid]["citers"].add(wid)

    return G

# ========== SIMILARITIES ==========

def compute_bibliographic_coupling(G, i, j):
    refs_i = G.nodes[i].get("references", set())
    refs_j = G.nodes[j].get("references", set())
    inter = len(refs_i & refs_j)
    denom = np.sqrt(len(refs_i) * len(refs_j))
    return inter/denom if denom > 0 else 0.0

def compute_co_citation(G, i, j):
    citers_i = G.nodes[i].get("citers", set())
    citers_j = G.nodes[j].get("citers", set())
    inter = len(citers_i & citers_j)
    denom = np.sqrt(len(citers_i) * len(citers_j))
    return inter/denom if denom > 0 else 0.0

def compute_text_similarity(G, model=None):
    nodes = list(G.nodes())
    texts = [((G.nodes[n].get("title","") or "") + " " + (G.nodes[n].get("abstract","") or "")).strip() for n in nodes]

    if model:
        embeddings = model.encode(texts, convert_to_tensor=False)
        sim_matrix = cosine_similarity(embeddings)
    else:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf)

    return nodes, sim_matrix

# ========== GRAPH PLOTTING ==========

def plot_graph(G, highlight_ids=None):
    plt.figure(figsize=(10,8))
    pos = nx.spring_layout(G, seed=42, k=0.5)

    if highlight_ids is None:
        highlight_ids = []

    node_colors = ["red" if n in highlight_ids else "skyblue" for n in G.nodes()]
    node_sizes = [300 + 50*G.degree(n) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", alpha=0.3)

    labels = {n: (G.nodes[n].get("title","")[:30] + "…") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.title("Closed Citation Graph")
    plt.axis("off")
    plt.show()

# ========== MAIN PIPELINE ==========

def run(work_ids, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    G = build_closed_citation_graph(work_ids)
    print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

    # Compute text similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    nodes, text_sim = compute_text_similarity(G, model=model)
    node_index = {n:i for i,n in enumerate(nodes)}

    # Compute weighted similarity for all pairs
    similarities = {}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            bc = compute_bibliographic_coupling(G, ni, nj)
            cc = compute_co_citation(G, ni, nj)
            ts = text_sim[node_index[ni], node_index[nj]]
            weight = alpha*bc + beta*cc + gamma*ts
            similarities[(ni,nj)] = {"BC": bc, "CC": cc, "TEXT": ts, "weight": weight}

    return G, similarities

# ========== RUN ==========

if __name__ == "__main__":
    work_ids = [
        "W4283734789",
        "W4378696928",
        "W4397049037",
        "W1030513622",
        "W1512719907",
        "W1518641734",
        "W1562460111",
        "W1980569135",
        "W2000018820",
        "W2000018820"
    ]

    G, sims = run(work_ids)

    print("\n=== Sample Similarity Values ===")
    for pair, vals in list(sims.items())[:10]:
        print(f"{pair}: {vals}")

    # Plot graph
    plot_graph(G, highlight_ids=work_ids)
