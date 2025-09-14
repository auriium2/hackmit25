import os
import requests

# Create output directory
os.makedirs("sample_papers", exist_ok=True)

# A small, diverse set of arXiv IDs (mix of NLP, CV, ML, RL, speech, etc.)
arxiv_ids = [
    # NLP
    "1706.03762",  # Attention is All You Need
    "1810.04805",  # BERT
    "1907.10529",  # RoBERTa
    "2005.14165",  # GPT-3
    "2104.08773",  # Switch Transformers

    # Vision
    "1512.03385",  # ResNet
    "1409.1556",   # VGG
    "1405.0312",   # AlexNet
    "1512.00567",  # Faster R-CNN
    "2010.11929",  # Vision Transformer (ViT)

    # Speech & Audio
    "1303.7887",   # Deep Speech
    "1703.08581",  # WaveNet
    "1710.08969",  # Tacotron 2

    # Reinforcement Learning
    "1312.5602",   # DQN
    "1509.02971",  # A3C
    "1707.06347",  # PPO
    "1802.09477",  # Rainbow

    # ML theory & optimization
    "1608.08225",  # Adam optimizer
    "1412.6980",   # Batch Norm
    "1206.5538",   # Dropout

    # Benchmarks
    "1804.07461",  # GLUE benchmark
    "2005.14165",  # GPT-3 (includes benchmarks)
    "1910.10683",  # SuperGLUE
    "2009.03300",  # CLIP
    "2107.03374",  # Codex
]

for arxiv_id in arxiv_ids:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out_path = os.path.join("sample_papers", f"{arxiv_id}.pdf")

    if os.path.exists(out_path):
        print(f"Already downloaded {arxiv_id}")
        continue

    try:
        print(f"Downloading {arxiv_id}...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print(f"Failed {arxiv_id}: {e}")

print("Done. PDFs are in ./sample_papers/")
