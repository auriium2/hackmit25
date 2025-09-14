import os, time, json, ray
from typing import List, Dict, Any
from ray.data import ActorPoolStrategy

# ---------- Stage 1: "download" from local files (read bytes) ----------
def read_local_batch(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        path = r["path"]
        try:
            with open(path, "rb") as f:
                data = f.read()
            out.append({
                "url": f"file://{os.path.abspath(path)}",
                "bytes": data,
                "status": 200,
                "content_type": "application/pdf",
                "size": len(data),
            })
        except Exception as e:
            out.append({"url": path, "error": str(e)})
    return out

# ---------- Stage 2: parse with Docling (warm actors) ----------
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline, StandardPdfPipelineOptions
from docling.datamodel.pipeline_options import TableFormerMode

def parse_batch(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # lazy singleton per actor process
    global _pipe
    if "_pipe" not in globals():
        opts = StandardPdfPipelineOptions(
            do_ocr=False,                              # flip True if scanned PDFs
            table_structure=TableFormerMode.ACCURATE,  # FAST for speed, ACCURATE for fidelity
        )
        _pipe = StandardPdfPipeline(opts)

    out = []
    for r in rows:
        if "error" in r:
            out.append({"url": r["url"], "error": r["error"]})
            continue
        try:
            res = _pipe.run_from_bytes(r["bytes"], origin=r["url"])
            out.append({
                "url": r["url"],
                "tables": [t.model_dump() for t in res.tables],
                "markdown": res.document.export_to_markdown(),  # or export_to_json()
            })
        except Exception as e:
            out.append({"url": r.get("url"), "error": str(e)})
    return out

def run_pipeline_local(pdf_dir: str,
                       num_download_blocks: int = 32,
                       num_parser_actors: int = 4,
                       dl_batch_size: int = 4,
                       parse_batch_size: int = 2,
                       out_dir: str = "./docling_out"):
    # gather files
    files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir)
             if f.lower().endswith(".pdf")]
    if not files:
        raise SystemExit(f"No PDFs found under: {pdf_dir}")

    # build dataset and fan out small blocks to drive parallelism
    ds = ray.data.from_items([{"path": p} for p in files]).repartition(num_download_blocks)

    start = time.time()

    # Stage 1: local "download" (file read) -> bytes into object store
    downloaded = ds.map_batches(
        read_local_batch,
        batch_size=dl_batch_size,
        compute="tasks",
    )

    # Stage 2: Docling parsing with warm actor pool
    parsed = downloaded.map_batches(
        parse_batch,
        batch_size=parse_batch_size,
        compute=ActorPoolStrategy(size=num_parser_actors)
    )

    # Persist results
    os.makedirs(out_dir, exist_ok=True)
    parsed.write_parquet(out_dir)

    # Small sample back to driver
    sample = parsed.take(3)
    dur = time.time() - start
    print(f"\nProcessed {len(files)} PDFs in {dur:.2f}s "
          f"({len(files)/max(dur,1e-9):.2f} PDFs/sec aggregate).")
    print("\nSample output (truncated):")
    print(json.dumps(sample, indent=2)[:1200])
    print(f"\nWrote results to: {out_dir}")

if __name__ == "__main__":
    
    ray.init(address="ray://<HEAD_IP>:10001")
    
    run_pipeline_local(
        pdf_dir="./sample_pdfs",
        num_download_blocks=32,     # increase for more fan-out
        num_parser_actors=4,        # match your CPU/GPU capacity
        dl_batch_size=4,
        parse_batch_size=2,
        out_dir="./docling_out"
    )