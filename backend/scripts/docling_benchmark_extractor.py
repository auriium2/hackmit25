import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time

import ray
from pydantic import BaseModel, Field
import openai
import json as json_lib
import pandas as pd

# Import existing models
from app.models.research import Benchmark, MetricValue, PaperAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedExtractionPrompt(BaseModel):
    """Enhanced prompts for processing structured document content."""

    structured_extraction: str = Field(default="""
You are analyzing a research paper that has been processed to preserve table structure and layout.

The content includes:
- Main text paragraphs
- Tables with preserved structure
- Figure captions
- Section headers

Extract ALL benchmarks, datasets, and performance metrics. Pay special attention to tables which often contain the most important benchmark results.

Return ONLY valid JSON:
{{
  "benchmarks": [
    {{
      "name": "exact dataset/benchmark name",
      "category": "computer_vision|nlp|audio|robotics|general",
      "standardized_key": "lowercase_with_underscores",
      "metric_type": "accuracy|f1|bleu|rmse|map|iou|top1|top5|em|perplexity",
      "description": "brief description of what it measures",
      "source_section": "section where found (e.g., 'Table 1', 'Results')"
    }}
  ],
  "metric_values": [
    {{
      "benchmark": "standardized_key_matching_benchmark",
      "value": "numerical result as string",
      "unit": "%|null",
      "context": "model/method name or experimental setup",
      "source_section": "where this result was found",
      "comparison_baseline": "baseline being compared against, if any"
    }}
  ]
}}

Focus especially on:
- Performance comparison tables (these are goldmines)
- Results sections with numerical comparisons
- Abstract mentions of key performance improvements
- Method names and their corresponding results

Document content:
{document_content}

Return only JSON.
""")


class DoclingPDFProcessor:
    """Advanced PDF processor using Docling for structured extraction."""

    def __init__(self):
        self.converter = None
        self._initialize_docling()

    def _initialize_docling(self):
        """Initialize Docling converter."""
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            # Configure pipeline for research papers
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,  # Enable OCR for scanned papers
                do_table_structure=True,  # Extract table structure
                table_structure_options={
                    "do_cell_matching": True,
                    "mode": "accurate"  # More accurate but slower
                }
            )

            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }

            self.converter = DocumentConverter(
                format_options=format_options
            )
            logger.info("Initialized Docling converter with table structure extraction")

        except ImportError as e:
            logger.error(f"Failed to import Docling: {e}")
            logger.error("Install with: pip install docling")
            self.converter = None

    def extract_structured_content(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract structured content from PDF including tables and layout.

        Returns:
            Dict with structured content including tables, text, and metadata
        """
        if not self.converter:
            logger.error("Docling converter not initialized")
            return {"error": "Docling not available"}

        pdf_path = Path(pdf_path)

        try:
            logger.info(f"Processing PDF with Docling: {pdf_path}")

            # Convert document
            result = self.converter.convert(pdf_path)
            doc = result.document

            # Extract structured content
            structured_content = {
                "title": doc.name or pdf_path.stem,
                "main_text": doc.export_to_text(),
                "tables": [],
                "figures": [],
                "sections": [],
                "metadata": {
                    "page_count": len(doc.pages) if hasattr(doc, 'pages') else 0,
                    "processing_time": time.time(),
                    "extraction_method": "docling"
                }
            }

            # Extract tables with structure preserved
            if hasattr(doc, 'tables') and doc.tables:
                for i, table in enumerate(doc.tables):
                    table_data = {
                        "table_id": f"table_{i+1}",
                        "caption": getattr(table, 'caption', ''),
                        "data": self._extract_table_data(table),
                        "location": f"Page {getattr(table, 'prov', [{}])[0].get('page_no', 'unknown')}"
                    }
                    structured_content["tables"].append(table_data)

            # Extract figures and captions
            if hasattr(doc, 'pictures') and doc.pictures:
                for i, figure in enumerate(doc.pictures):
                    figure_data = {
                        "figure_id": f"figure_{i+1}",
                        "caption": getattr(figure, 'caption', ''),
                        "location": f"Page {getattr(figure, 'prov', [{}])[0].get('page_no', 'unknown')}"
                    }
                    structured_content["figures"].append(figure_data)

            # Create formatted content for LLM
            formatted_content = self._format_for_llm(structured_content)
            structured_content["formatted_for_llm"] = formatted_content

            logger.info(f"Extracted {len(structured_content['tables'])} tables and "
                       f"{len(structured_content['figures'])} figures")

            return structured_content

        except Exception as e:
            logger.error(f"Error processing PDF with Docling: {e}")
            return {"error": str(e)}

    def _extract_table_data(self, table) -> List[List[str]]:
        """Extract table data as list of rows."""
        try:
            # Try to get table data in various formats
            if hasattr(table, 'data'):
                return table.data
            elif hasattr(table, 'to_dataframe'):
                df = table.to_dataframe()
                return [df.columns.tolist()] + df.values.tolist()
            elif hasattr(table, 'cells'):
                # Extract from cell structure
                rows = {}
                for cell in table.cells:
                    row_idx = getattr(cell, 'row_idx', 0)
                    col_idx = getattr(cell, 'col_idx', 0)
                    content = getattr(cell, 'content', str(cell))

                    if row_idx not in rows:
                        rows[row_idx] = {}
                    rows[row_idx][col_idx] = content

                # Convert to list of lists
                table_data = []
                for row_idx in sorted(rows.keys()):
                    row = []
                    for col_idx in sorted(rows[row_idx].keys()):
                        row.append(str(rows[row_idx][col_idx]))
                    table_data.append(row)

                return table_data
            else:
                return [["Table structure not extractable"]]

        except Exception as e:
            logger.warning(f"Failed to extract table data: {e}")
            return [["Error extracting table"]]

    def _format_for_llm(self, structured_content: Dict[str, Any]) -> str:
        """Format structured content for LLM processing."""
        formatted = []

        # Add title
        if structured_content.get("title"):
            formatted.append(f"TITLE: {structured_content['title']}")
            formatted.append("")

        # Add main text (truncated for LLM limits)
        main_text = structured_content.get("main_text", "")
        if main_text:
            # Limit text length
            max_text_chars = 30000  # Leave room for tables
            if len(main_text) > max_text_chars:
                main_text = main_text[:max_text_chars] + "...[truncated]"
            formatted.append("MAIN TEXT:")
            formatted.append(main_text)
            formatted.append("")

        # Add tables (these are critical for benchmarks)
        for table in structured_content.get("tables", []):
            formatted.append(f"TABLE: {table['table_id']}")
            if table.get("caption"):
                formatted.append(f"Caption: {table['caption']}")

            # Format table data
            if table.get("data"):
                try:
                    # Create a simple table format
                    table_text = []
                    for row in table["data"]:
                        table_text.append(" | ".join(str(cell) for cell in row))
                    formatted.append("\n".join(table_text))
                except:
                    formatted.append("Table data format error")

            formatted.append(f"Location: {table.get('location', 'unknown')}")
            formatted.append("")

        # Add figure captions (may contain benchmark info)
        for figure in structured_content.get("figures", []):
            formatted.append(f"FIGURE: {figure['figure_id']}")
            if figure.get("caption"):
                formatted.append(f"Caption: {figure['caption']}")
            formatted.append(f"Location: {figure.get('location', 'unknown')}")
            formatted.append("")

        return "\n".join(formatted)


@ray.remote
class DoclingBenchmarkExtractor:
    """Ray actor for LLM-based benchmark extraction using Docling-processed content."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", gpu_id: Optional[int] = None):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.prompts = EnhancedExtractionPrompt()

        # Initialize Docling processor
        self.pdf_processor = DoclingPDFProcessor()

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key)

        logger.info(f"Initialized DoclingBenchmarkExtractor on GPU {gpu_id} with model {model_name}")

    def extract_from_pdf(self, pdf_path: Union[str, Path],
                        domain_hint: Optional[str] = None) -> Dict[str, Any]:
        """Extract benchmarks from PDF using Docling + LLM."""

        try:
            # Step 1: Extract structured content with Docling
            logger.info(f"Step 1: Extracting structured content from {pdf_path}")
            structured_content = self.pdf_processor.extract_structured_content(pdf_path)

            if "error" in structured_content:
                return structured_content

            # Step 2: Process with LLM
            logger.info("Step 2: Processing with LLM for benchmark extraction")
            llm_input = structured_content["formatted_for_llm"]

            if self.client:
                extraction_result = self._call_llm_api(llm_input)
            else:
                logger.warning("No OpenAI client, using mock extraction")
                logger.warning("MOCK WARNING: Results are simulated and not based on actual document content")
                extraction_result = self._mock_extraction(llm_input)

            # Step 3: Combine results
            final_result = {
                **extraction_result,
                "document_structure": {
                    "tables_found": len(structured_content.get("tables", [])),
                    "figures_found": len(structured_content.get("figures", [])),
                    "page_count": structured_content["metadata"]["page_count"]
                },
                "paper_metadata": {
                    "file_path": str(pdf_path),
                    "file_name": Path(pdf_path).name,
                    "processing_timestamp": time.time(),
                    "extraction_method": "docling_llm"
                }
            }

            return final_result

        except Exception as e:
            logger.error(f"Error in PDF extraction: {e}")
            return {"error": str(e)}

    def _call_llm_api(self, formatted_content: str) -> Dict[str, Any]:
        """Call OpenAI API for benchmark extraction."""
        try:
            prompt = self.prompts.structured_extraction.format(
                document_content=formatted_content
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting benchmarks and performance metrics from research papers. Pay special attention to tables and structured data."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000
            )

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                result = json_lib.loads(content)
                return {
                    "benchmarks": result.get("benchmarks", []),
                    "metric_values": result.get("metric_values", []),
                    "extraction_metadata": {
                        "model_used": self.model_name,
                        "tokens_used": response.usage.total_tokens,
                        "timestamp": time.time()
                    }
                }
            except json_lib.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                # Try to extract JSON with regex
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        result = json_lib.loads(json_match.group())
                        return {
                            "benchmarks": result.get("benchmarks", []),
                            "metric_values": result.get("metric_values", []),
                            "extraction_metadata": {
                                "model_used": self.model_name,
                                "timestamp": time.time(),
                                "parsing_warning": "Had to extract JSON with regex"
                            }
                        }
                    except:
                        pass

                return {
                    "benchmarks": [],
                    "metric_values": [],
                    "error": f"Failed to parse LLM response: {content[:500]}..."
                }

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return {
                "benchmarks": [],
                "metric_values": [],
                "error": f"API call failed: {str(e)}"
            }

    def _mock_extraction(self, content: str) -> Dict[str, Any]:
        """Mock extraction for testing without API key."""
        # Simple pattern matching on the formatted content
        content_lower = content.lower()

        logger.warning("MOCK WARNING: Using pattern matching instead of LLM for extraction")

        benchmarks = []
        metrics = []

        # Look for common benchmark patterns in tables and text
        patterns = {
            'imagenet': 'computer_vision',
            'cifar': 'computer_vision',
            'mnist': 'computer_vision',
            'coco': 'computer_vision',
            'squad': 'nlp',
            'glue': 'nlp',
            'bleu': 'nlp'
        }

        for pattern, category in patterns.items():
            if pattern in content_lower:
                benchmarks.append({
                    "name": pattern.title(),
                    "category": category,
                    "standardized_key": pattern.lower(),
                    "metric_type": "accuracy" if category == 'computer_vision' else "f1",
                    "description": f"Mock extraction of {pattern}",
                    "source_section": "Mock"
                })

                metrics.append({
                    "benchmark": pattern.lower(),
                    "value": f"{80 + hash(pattern) % 20:.1f}",
                    "unit": "%",
                    "context": "Mock method",
                    "source_section": "Mock table"
                })

        return {
            "benchmarks": benchmarks,
            "metric_values": metrics,
            "extraction_metadata": {
                "model_used": "mock",
                "timestamp": time.time(),
                "mock_warning": "These results are simulated and may not accurately reflect document content"
            },
            "warning": "MOCK DATA: Results generated without LLM processing"
        }


@ray.remote
def process_paper_batch(paper_paths: List[Union[str, Path]],
                       domain_hint: Optional[str] = None,
                       model_name: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Process a batch of papers in parallel on a Ray cluster.

    This function can handle massive parallelization across cluster nodes.
    """
    logger.info(f"Processing batch of {len(paper_paths)} papers")

    # Create extractor actors (one per available GPU if possible)
    num_actors = min(len(paper_paths), 8)  # Adjust based on cluster size

    extractors = []
    for i in range(num_actors):
        # Try to assign different GPUs if available
        gpu_id = i % torch.cuda.device_count() if 'torch' in sys.modules and torch.cuda.is_available() else None
        extractor = DoclingBenchmarkExtractor.remote(model_name=model_name, gpu_id=gpu_id)
        extractors.append(extractor)

    # Distribute papers across extractors
    tasks = []
    for i, paper_path in enumerate(paper_paths):
        extractor = extractors[i % len(extractors)]
        task = extractor.extract_from_pdf.remote(paper_path, domain_hint)
        tasks.append((str(paper_path), task))

    # Collect results
    results = {}
    global_benchmarks = {}
    all_metrics = []
    mock_used = False

    for paper_path, task in tasks:
        try:
            result = ray.get(task)
            paper_id = Path(paper_path).stem
            results[paper_id] = result

            # Aggregate benchmarks
            # Check if mock was used
            if result.get("extraction_metadata", {}).get("model_used") == "mock":
                mock_used = True
                logger.warning(f"MOCK WARNING: Paper {paper_id} processed with mock extraction")

            for benchmark in result.get("benchmarks", []):
                key = benchmark.get("standardized_key", benchmark.get("name", "unknown")).lower()
                if key not in global_benchmarks:
                    global_benchmarks[key] = benchmark.copy()
                    global_benchmarks[key]["papers"] = []
                global_benchmarks[key]["papers"].append(paper_id)

            # Collect metrics
            all_metrics.extend(result.get("metric_values", []))

        except Exception as e:
            paper_id = Path(paper_path).stem
            logger.error(f"Failed to process {paper_path}: {e}")
            results[paper_id] = {"error": str(e)}

    return {
        "results": results,
        "global_benchmarks": global_benchmarks,
        "all_metrics": all_metrics,
        "processing_metadata": {
            "papers_processed": len([r for r in results.values() if "error" not in r]),
            "papers_failed": len([r for r in results.values() if "error" in r]),
            "total_benchmarks": len(global_benchmarks),
            "total_metrics": len(all_metrics),
            "cluster_info": ray.cluster_resources(),
            "timestamp": time.time(),
            "mock_extraction_us ed": mock_used,
            "mock_warning": "Some results were generated with mock extraction and may not be accurate" if mock_used else None
        }
    }


def main():
    """CLI for massively parallel benchmark extraction."""
    parser = argparse.ArgumentParser(description="Docling-based benchmark extraction with Ray")
    parser.add_argument("--paper_path", type=str, help="Single PDF path")
    parser.add_argument("--batch_dir", type=str, help="Directory of PDFs to process")
    parser.add_argument("--ray_address", type=str, help="Ray cluster address")
    parser.add_argument("--domain_hint", choices=["cv", "nlp", "general"], default="general")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--output", default="docling_results.json", help="Output file")
    parser.add_argument("--max_papers", type=int, help="Max papers to process")

    args = parser.parse_args()

    # Initialize Ray (connect to cluster if specified)
    if args.ray_address:
        ray.init(address=args.ray_address)
        logger.info(f"Connected to Ray cluster at {args.ray_address}")
    else:
        ray.init()
        logger.info("Started local Ray cluster")

    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")

    async def run():
        if args.paper_path:
            # Single paper processing
            extractor = DoclingBenchmarkExtractor.remote(model_name=args.model)
            result = await extractor.extract_from_pdf.remote(args.paper_path, args.domain_hint)

            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            logger.info(f"Processed {args.paper_path}")
            logger.info(f"Benchmarks: {len(result.get('benchmarks', []))}")
            logger.info(f"Metrics: {len(result.get('metric_values', []))}")

            # Check if mock was used
            if result.get("extraction_metadata", {}).get("model_used") == "mock":
                logger.warning("MOCK WARNING: Results were generated with mock extraction and may not be accurate")

        elif args.batch_dir:
            # Batch processing
            batch_dir = Path(args.batch_dir)
            pdf_files = list(batch_dir.glob("*.pdf"))

            if args.max_papers:
                pdf_files = pdf_files[:args.max_papers]

            logger.info(f"Found {len(pdf_files)} PDFs to process")

            # Process in parallel across the cluster
            result = await process_paper_batch.remote(
                pdf_files, args.domain_hint, args.model
            )

            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            metadata = result["processing_metadata"]
            logger.info(f"Processed {metadata['papers_processed']} papers successfully")
            logger.info(f"Found {metadata['total_benchmarks']} unique benchmarks")
            logger.info(f"Extracted {metadata['total_metrics']} total metrics")

            # Check if mock was used in batch processing
            if metadata.get("mock_extraction_used"):
                logger.warning("MOCK WARNING: Some papers were processed with mock extraction")
                logger.warning("Results from mock extraction may not accurately reflect document content")

        else:
            parser.print_help()

    # Run async
    asyncio.run(run())

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
