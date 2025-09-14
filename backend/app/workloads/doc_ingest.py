#!/usr/bin/env python3
"""
Ray Data pipeline component for distributed document processing using Docling.
Processes research papers (PDFs) into markdown format with comprehensive
error handling, monitoring, and performance optimization.
"""

import os
import hashlib
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import shutil

import ray
import ray.data
from ray.data import Dataset
import pandas as pd
import pyarrow as pa

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enum for document processing status."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class DocumentResult:
    """Result of document processing containing all relevant metadata."""
    source_url: str
    status: ProcessingStatus
    markdown_content: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    worker_id: Optional[str] = None
    processed_at: Optional[str] = None


@dataclass
class ProcessingStats:
    """Statistics for batch processing results."""
    total_documents: int
    successful: int
    failed: int
    skipped: int
    total_processing_time: float
    average_processing_time: float
    throughput_docs_per_second: float
    error_summary: Dict[str, int] = field(default_factory=dict)


class DocumentProcessorConfig:
    """Configuration for the document processor pipeline."""

    def __init__(self,
                 models_dir: str = "models",
                 output_dir: Optional[str] = None,
                 batch_size: int = 1,
                 concurrency: int = 4,
                 max_retries: int = 2,
                 timeout_seconds: int = 300,
                 enable_caching: bool = True,
                 ray_remote_args: Optional[Dict[str, Any]] = None):
        """
        Initialize document processor configuration.

        Args:
            models_dir: Directory containing Docling models
            output_dir: Optional output directory for processed files
            batch_size: Number of documents to process per batch
            concurrency: Number of concurrent Ray actors
            max_retries: Maximum retry attempts for failed documents
            timeout_seconds: Timeout for individual document processing
            enable_caching: Whether to cache processed results
            ray_remote_args: Additional arguments for Ray remote functions
        """
        self.models_dir = models_dir
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="docling_out_")
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_caching = enable_caching
        self.ray_remote_args = ray_remote_args or {}

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)


class DoclingBatchProcessor:
    """Ray Data batch processor for converting documents using Docling."""

    def __init__(self, config: DocumentProcessorConfig):
        """Initialize the batch processor with configuration."""
        self.config = config
        self._init_converter()
        self._cache = {} if config.enable_caching else None
        self.worker_id = f"worker_{os.getpid()}"

        logger.info(f"Initialized DoclingBatchProcessor {self.worker_id} with config: "
                   f"models_dir={config.models_dir}, batch_size={config.batch_size}")

    def _init_converter(self):
        """Initialize the Docling document converter."""
        try:
            pipeline_options = PdfPipelineOptions(
                artifacts_path=self.config.models_dir,
            )

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info(f"Successfully initialized Docling converter with models from {self.config.models_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            raise

    def _process_single_document(self, source_url: str, retry_count: int = 0) -> DocumentResult:
        """Process a single document and return structured result."""
        start_time = time.time()

        try:
            # Check cache if enabled
            if self._cache is not None:
                cache_key = hashlib.sha256(source_url.encode()).hexdigest()
                if cache_key in self._cache:
                    cached_result = self._cache[cache_key]
                    logger.debug(f"Retrieved cached result for {source_url}")
                    return cached_result

            # Process document
            logger.info(f"Processing document: {source_url}")
            doc_result = self.converter.convert(source_url)
            document = doc_result.document

            # Serialize to markdown
            serializer = MarkdownDocSerializer(doc=document)
            markdown_content = serializer.serialize().text

            # Calculate content hash
            content_hash = hashlib.sha256(markdown_content.encode()).hexdigest()

            # Extract metadata
            metadata = {
                "title": getattr(document, 'title', None),
                "language": getattr(document, 'language', None),
                "creation_date": getattr(document, 'creation_date', None),
            }

            # Get document stats
            page_count = len(document.pages) if hasattr(document, 'pages') else None

            processing_time = time.time() - start_time

            result = DocumentResult(
                source_url=source_url,
                status=ProcessingStatus.SUCCESS,
                markdown_content=markdown_content,
                processing_time=processing_time,
                file_size=len(markdown_content.encode()),
                page_count=page_count,
                content_hash=content_hash,
                metadata=metadata,
                retry_count=retry_count,
                worker_id=self.worker_id,
                processed_at=pd.Timestamp.now().isoformat()
            )

            # Cache result if enabled
            if self._cache is not None:
                self._cache[cache_key] = result

            logger.info(f"Successfully processed {source_url} in {processing_time:.2f}s")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Failed to process {source_url}: {error_msg}")

            return DocumentResult(
                source_url=source_url,
                status=ProcessingStatus.FAILED,
                error_message=error_msg,
                processing_time=processing_time,
                retry_count=retry_count,
                worker_id=self.worker_id,
                processed_at=pd.Timestamp.now().isoformat()
            )

    def __call__(self, batch: Union[pa.Table, pd.DataFrame, Dict[str, List]]) -> Dict[str, List]:
        """
        Process a batch of documents using Ray Data.

        Args:
            batch: Batch of documents in various formats

        Returns:
            Dictionary with processed results in columnar format
        """
        # Normalize batch to dictionary format
        if hasattr(batch, "to_pydict"):  # pyarrow.Table
            batch_dict = batch.to_pydict()
        elif hasattr(batch, "to_dict"):  # pandas.DataFrame
            batch_dict = batch.to_dict(orient="list")
        else:  # Already a dict of columns
            batch_dict = batch

        # Extract source URLs
        sources = batch_dict.get("source_url", batch_dict.get("src", []))
        if not sources:
            logger.warning("No source URLs found in batch")
            return self._empty_result()

        logger.info(f"Processing batch of {len(sources)} documents on worker {self.worker_id}")

        # Process each document
        results = []
        for source_url in sources:
            result = self._process_single_document(source_url)

            # Retry logic for failed documents
            if (result.status == ProcessingStatus.FAILED and
                result.retry_count < self.config.max_retries):
                logger.info(f"Retrying {source_url} (attempt {result.retry_count + 1})")
                result = self._process_single_document(source_url, result.retry_count + 1)

            results.append(result)

        # Convert results to columnar format for Ray Data
        return self._results_to_columns(results)

    def _results_to_columns(self, results: List[DocumentResult]) -> Dict[str, List]:
        """Convert list of DocumentResult to columnar dictionary."""
        return {
            "source_url": [r.source_url for r in results],
            "status": [r.status.value for r in results],
            "markdown_content": [r.markdown_content for r in results],
            "error_message": [r.error_message for r in results],
            "processing_time": [r.processing_time for r in results],
            "file_size": [r.file_size for r in results],
            "page_count": [r.page_count for r in results],
            "content_hash": [r.content_hash for r in results],
            "metadata": [r.metadata for r in results],
            "retry_count": [r.retry_count for r in results],
            "worker_id": [r.worker_id for r in results],
            "processed_at": [r.processed_at for r in results]
        }

    def _empty_result(self) -> Dict[str, List]:
        """Return empty result structure."""
        return {
            "source_url": [],
            "status": [],
            "markdown_content": [],
            "error_message": [],
            "processing_time": [],
            "file_size": [],
            "page_count": [],
            "content_hash": [],
            "metadata": [],
            "retry_count": [],
            "worker_id": [],
            "processed_at": []
        }


class DocumentProcessingPipeline:
    """Main Ray Data pipeline for document processing."""

    def __init__(self, config: DocumentProcessorConfig):
        """Initialize the processing pipeline."""
        self.config = config
        logger.info(f"Initialized DocumentProcessingPipeline with config: {config.__dict__}")

    def process_documents(self, source_urls: List[str]) -> Dataset:
        """
        Process a list of document URLs using Ray Data pipeline.

        Args:
            source_urls: List of document URLs to process

        Returns:
            Ray Dataset with processed results
        """
        if not source_urls:
            raise ValueError("No source URLs provided")

        logger.info(f"Starting processing pipeline for {len(source_urls)} documents")

        # Create Ray Dataset from source URLs
        dataset = ray.data.from_items([{"source_url": url} for url in source_urls])

        # Process documents with batch processor
        processed_dataset = dataset.map_batches(
            DoclingBatchProcessor(self.config),
            batch_size=self.config.batch_size,
            concurrency=self.config.concurrency,
            ray_remote_args=self.config.ray_remote_args
        )

        logger.info("Document processing pipeline completed")
        return processed_dataset

    def process_and_collect(self, source_urls: List[str]) -> Tuple[List[DocumentResult], ProcessingStats]:
        """
        Process documents and collect results with statistics.

        Args:
            source_urls: List of document URLs to process

        Returns:
            Tuple of (results, statistics)
        """
        start_time = time.time()

        # Process documents
        dataset = self.process_documents(source_urls)
        raw_results = dataset.take_all()

        # Convert to DocumentResult objects
        results = []
        for i, row in enumerate(raw_results):
            result = DocumentResult(
                source_url=row["source_url"],
                status=ProcessingStatus(row["status"]),
                markdown_content=row["markdown_content"],
                error_message=row["error_message"],
                processing_time=row["processing_time"],
                file_size=row["file_size"],
                page_count=row["page_count"],
                content_hash=row["content_hash"],
                metadata=row["metadata"] or {},
                retry_count=row["retry_count"],
                worker_id=row["worker_id"],
                processed_at=row["processed_at"]
            )
            results.append(result)

        # Generate statistics
        stats = self._generate_stats(results, time.time() - start_time)

        logger.info(f"Pipeline completed: {stats.successful}/{stats.total_documents} successful, "
                   f"{stats.failed} failed in {stats.total_processing_time:.2f}s")

        return results, stats

    def _generate_stats(self, results: List[DocumentResult], total_time: float) -> ProcessingStats:
        """Generate processing statistics from results."""
        successful = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
        failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
        skipped = sum(1 for r in results if r.status == ProcessingStatus.SKIPPED)

        processing_times = [r.processing_time for r in results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        throughput = len(results) / total_time if total_time > 0 else 0

        # Error summary
        error_summary = {}
        for result in results:
            if result.error_message:
                error_type = type(result.error_message).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

        return ProcessingStats(
            total_documents=len(results),
            successful=successful,
            failed=failed,
            skipped=skipped,
            total_processing_time=total_time,
            average_processing_time=avg_processing_time,
            throughput_docs_per_second=throughput,
            error_summary=error_summary
        )

    def save_results(self, results: List[DocumentResult], output_path: str):
        """Save processing results to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        successful_results = [r for r in results if r.status == ProcessingStatus.SUCCESS]

        logger.info(f"Saving {len(successful_results)} successful results to {output_dir}")

        for result in successful_results:
            if result.markdown_content:
                # Create filename from source URL
                url_hash = hashlib.sha256(result.source_url.encode()).hexdigest()[:12]
                filename = f"doc_{url_hash}.md"
                filepath = output_dir / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Document: {result.source_url}\n\n")
                    f.write(f"Processed: {result.processed_at}\n")
                    f.write(f"Worker: {result.worker_id}\n")
                    f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                    f.write(f"Pages: {result.page_count}\n\n")
                    f.write("---\n\n")
                    f.write(result.markdown_content)

                logger.debug(f"Saved {result.source_url} to {filename}")


# Convenience functions for easy usage

def create_default_config(**kwargs) -> DocumentProcessorConfig:
    """Create a default configuration with optional overrides."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_models_dir = os.path.join(base_dir, "models")
    default_output_dir = os.path.join(base_dir, "out", "processed_docs")

    defaults = {
        "models_dir": default_models_dir,
        "output_dir": default_output_dir,
        "batch_size": 1,
        "concurrency": 4,
        "max_retries": 2,
        "timeout_seconds": 300,
        "enable_caching": True
    }

    defaults.update(kwargs)
    return DocumentProcessorConfig(**defaults)


def process_documents_ray(source_urls: List[str],
                         config: Optional[DocumentProcessorConfig] = None,
                         save_results: bool = True) -> Tuple[List[DocumentResult], ProcessingStats]:
    """
    Process documents using Ray Data pipeline with sensible defaults.

    Args:
        source_urls: List of document URLs to process
        config: Optional custom configuration
        save_results: Whether to save results to disk

    Returns:
        Tuple of (results, statistics)
    """
    # Use default config if none provided
    if config is None:
        config = create_default_config()

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.info("Initialized Ray for document processing")

    # Create and run pipeline
    pipeline = DocumentProcessingPipeline(config)
    results, stats = pipeline.process_and_collect(source_urls)

    # Save results if requested
    if save_results:
        pipeline.save_results(results, config.output_dir)

    return results, stats


# Example usage and testing
if __name__ == "__main__":
    import sys

    # Sample document URLs for testing
    sample_urls = [
        "https://arxiv.org/pdf/2309.15462",
        "https://arxiv.org/pdf/2410.09871.pdf",
        "https://arxiv.org/pdf/2306.09226.pdf",
        "https://arxiv.org/pdf/2505.01435.pdf"
    ]

    print("🚀 Testing Ray Data Document Processing Pipeline")
    print("=" * 70)

    try:
        # Create custom configuration
        config = create_default_config(
            batch_size=2,
            concurrency=4,
            max_retries=1,
            enable_caching=True
        )

        print(f"📁 Models directory: {config.models_dir}")
        print(f"📁 Output directory: {config.output_dir}")
        print(f"⚙️  Configuration: batch_size={config.batch_size}, concurrency={config.concurrency}")
        print()

        # Process documents
        print(f"🔄 Processing {len(sample_urls)} documents...")
        results, stats = process_documents_ray(
            source_urls=sample_urls,
            config=config,
            save_results="--save" in sys.argv
        )

        # Display results
        print(f"\n📊 Processing Statistics:")
        print(f"   Total documents: {stats.total_documents}")
        print(f"   ✅ Successful: {stats.successful}")
        print(f"   ❌ Failed: {stats.failed}")
        print(f"   ⏱️  Total time: {stats.total_processing_time:.2f}s")
        print(f"   📈 Throughput: {stats.throughput_docs_per_second:.2f} docs/sec")
        print(f"   ⚡ Avg processing time: {stats.average_processing_time:.2f}s")

        if stats.error_summary:
            print(f"\n🚨 Error Summary:")
            for error_type, count in stats.error_summary.items():
                print(f"   • {error_type}: {count}")

        # Show sample results
        print(f"\n📄 Sample Results:")
        for i, result in enumerate(results[:3]):  # Show first 3
            status_emoji = "✅" if result.status == ProcessingStatus.SUCCESS else "❌"
            print(f"   {status_emoji} {result.source_url}")
            print(f"      Status: {result.status.value}")
            print(f"      Processing time: {result.processing_time:.2f}s")
            if result.markdown_content:
                content_preview = result.markdown_content[:100].replace('\n', ' ')
                print(f"      Content preview: {content_preview}...")
            if result.error_message:
                print(f"      Error: {result.error_message}")
            print()

        print("=" * 70)
        print("✅ Document processing pipeline completed successfully!")

        if "--save" in sys.argv:
            print(f"💾 Results saved to: {config.output_dir}")
        else:
            print("💡 Use '--save' flag to save results to disk")

        # Output JSON if requested
        if "--json" in sys.argv:
            import json
            import dataclasses

            print("\n📄 JSON Output:")
            results_dict = [dataclasses.asdict(result) for result in results]
            stats_dict = dataclasses.asdict(stats)

            output = {
                "results": results_dict,
                "statistics": stats_dict
            }

            print(json.dumps(output, indent=2, default=str))

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    finally:
        # Cleanup Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown completed")