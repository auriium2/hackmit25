import os, ray, httpx, json
from typing import List, Dict, Any
from ray.data import ActorPoolStrategy

ray.init()  # or ray.init(address="auto")

# ---------- Stage 1: massively parallel downloader ----------
def download_batch(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # One HTTP client per task (connection pooling inside task).
    client = httpx.Client(
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=True,
        headers={"User-Agent": "ray-docling-pipeline/1.0"},
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=50),
    )
    out = []
    for r in rows:
        url = r["url"]
        try:
            # Optional: HEAD to skip non-PDFs or huge files
            # head = client.head(url)
            # if 'application/pdf' not in head.headers.get('content-type',''): continue

            resp = client.get(url)
            resp.raise_for_status()
            out.append({
                "url": url,
                "bytes": resp.content,            # PDF bytes (goes to object store)
                "status": resp.status_code,
                "content_type": resp.headers.get("content-type", "")
            })
        except Exception as e:
            out.append({"url": url, "error": str(e)})
    client.close()
    return out

# ---------- Stage 2: Docling parser as warm actor UDF ----------
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import PdfPipelineOptions

# The function below runs *inside* persistent Ray actors when compute=ActorPoolStrategy(...)
def parse_batch(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Lazy, per-actor singleton model init
    global _docling_pipe
    if "_docling_pipe" not in globals():
        opts = PdfPipelineOptions(
            do_ocr=False,                              # flip True if you truly need OCR
        )
        _docling_pipe = StandardPdfPipeline(opts)

    out = []
    for r in rows:
        if "error" in r:
            out.append({"url": r["url"], "error": r["error"]})
            continue
        try:
            res = _docling_pipe.
            res = _docling_pipe.run_from_bytes(r["bytes"], origin=r["url"])
            out.append({
                "url": r["url"],
                "markdown": res.document.export_to_markdown(),
                "tables": [t.model_dump() for t in res.tables],
            })
        except Exception as e:
            out.append({"url": r["url"], "error": str(e)})
    return out

# ---------- Wire it together ----------
def run_pipeline(urls: List[str], num_download_blocks: int = 256, num_parser_actors: int = 8):
    # Build dataset of URLs and fan out many small blocks to drive parallelism
    ds = ray.data.from_items([{"url": u} for u in urls]).repartition(num_download_blocks)

    # Stage 1: downloading uses Ray tasks (cheap to scale out)
    downloaded = ds.map_batches(
        download_batch,
        batch_size=8,            # per-task batch; tune based on URL sizes
    )

    # Stage 2: parsing uses persistent actors so models stay warm
    parsed = downloaded.map_batches(
        parse_batch,
        batch_size=2,            # small batches to keep latency down per actor
        compute=ActorPoolStrategy(size=num_parser_actors),
    )

    # Persist results
    # Parquet is great for tables; JSONL is simple for quick checks
    #parsed.write_parquet("/data/docling_out/")  # or s3://bucket/prefix

    # Optionally return a small sample to the driver
    return parsed.take(5)

# Example usage:
urls = ["https://arxiv.org/pdf/2212.06817"]   # hundreds or thousands
sample = run_pipeline(urls, num_download_blocks=256, num_parser_actors=8)
print(json.dumps(sample, indent=2)[:2000])
