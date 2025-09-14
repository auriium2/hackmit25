import ray
import ray.data as rd

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
import hashlib
import os


# Define model and output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend directory
OUT_DIR = os.path.join(BASE_DIR, "out")

# Print paths to verify
print(f"Base directory: {BASE_DIR}")
print(f"Output directory: {OUT_DIR}")

class DoclingBatchConverter:
    def __init__(self, models_dir: str, out_dir: str):
        pipeline_options = PdfPipelineOptions(
            artifacts_path=models_dir,
        )

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        self.out_dir = out_dir

    def _convert_one(self, src: str):
        try:
            doc = self.converter.convert(src).document
            md = MarkdownDocSerializer(doc=doc).serialize().text

            name = src
            if not name.endswith(".pdf"):
                name = hashlib.sha1(src.encode("utf-8")).hexdigest() + ".pdf"

            return {"src": src, "ok": True, "data": md, "error": None}
        except Exception as e:
            return {"src": src, "ok": False, "data": None, "error": repr(e)}

    def __call__(self, batch):
        # Handle different batch formats that Ray might provide
        # Could be a pyarrow.Table, pandas.DataFrame, or dict of columns
        if hasattr(batch, "to_pydict"):  # pyarrow.Table
            batch_dict = batch.to_pydict()
        elif hasattr(batch, "to_dict"):  # pandas.DataFrame
            batch_dict = batch.to_dict(orient="list")
        else:  # Already a dict of columns
            batch_dict = batch

        # Now we have a dict with column names as keys
        srcs = batch_dict["src"]
        results = [self._convert_one(s) for s in srcs]
        # Return as columnar dict (Ray Datasets friendly)
        return {
            "src": [r["src"] for r in results],
            "ok": [r["ok"] for r in results],
            "data": [r["data"] for r in results],
            "error": [r["error"] for r in results],
        }



ray.init(
    #local_mode=True,
    ignore_reinit_error=True,
    runtime_env={
        #"uris": ["s3://your-bucket/data_item.zip"],
        'working_dir': None
    },
)
print("Ray initialized in LOCAL mode - models accessed directly from filesystem")

# Create the output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Ray initialized. Address: {ray.get_runtime_context().gcs_address}")
print(f"Writing output to: {OUT_DIR}")

# Build a dataset of inputs
sources = [
    "https://arxiv.org/pdf/2309.15462",
    "https://arxiv.org/pdf/2410.09871.pdf",
    "https://arxiv.org/pdf/2306.09226.pdf",
    "https://arxiv.org/pdf/2505.01435.pdf"
    #"https://arxiv.org/pdf/2401.xxxxx",
]
ds = rd.from_items([{"src": s} for s in sources])

NUM_ACTORS = 4

# Pass the class and constructor args to map_batches
# This way each worker will create its own instance with the correct paths
result_ds = ds.map_batches(
    DoclingBatchConverter("models", OUT_DIR),
    batch_size=1,                      # tune based on memory/throughput; 1 is fine too
    concurrency=4,
    # Set resource requirements for workers
    # ray_remote_args={"num_gpus": 1, "num_cpus": 1}  # one GPU per actor
)

# Trigger execution and collect a small sample (or write out somewhere)
results = result_ds.take_all()
# Or `result_ds.show(limit=20)` to peek

# Simple summary
ok = sum(1 for r in results if r["ok"])
fail = [r for r in results if not r["ok"]]
print(f"Converted: {ok} ✓, Failed: {len(fail)} ✗")
for r in fail:
    print(f"- {r['src']}: {r['error']}")
