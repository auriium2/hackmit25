#!/usr/bin/env python3
"""
Benchmark extraction script using LLMs to extract relevant benchmarks from research papers.

This script processes PDF research papers and uses language models to identify and extract
benchmarks, metrics, and performance results. It integrates with the existing Ray/FastAPI
architecture and follows the established data models.

Usage:
    python extract_benchmarks.py --paper_path sample_papers/1234.5678.pdf
    python extract_benchmarks.py --batch_process sample_papers/
    python extract_benchmarks.py --interactive
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time

import PyPDF2
import ray
from pydantic import BaseModel, Field
import openai
import json as json_lib

# Import existing models
from app.models.research import Benchmark, MetricValue, PaperAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BenchmarkExtractionPrompt(BaseModel):
    """Structured prompts for different types of benchmark extraction."""
    general_extraction: str = Field(default="""
You are a research paper analysis expert. Analyze the following paper text and extract ALL benchmarks, datasets, and evaluation metrics mentioned.

Return ONLY valid JSON with this exact structure:
{{
  "benchmarks": [
    {{
      "name": "exact name from paper",
      "category": "computer_vision|nlp|audio|robotics|general",
      "standardized_key": "lowercase_with_underscores",
      "metric_type": "accuracy|f1|bleu|rmse|map|etc",
      "description": "brief description"
    }}
  ],
  "metric_values": [
    {{
      "benchmark": "standardized_key_matching_benchmark",
      "value": "numerical result as string",
      "unit": "% or null",
      "context": "model/method name or configuration"
    }}
  ]
}}

Look for:
- Dataset names (ImageNet, CIFAR, MNIST, SQuAD, GLUE, etc.)
- Performance metrics (accuracy %, F1 scores, BLEU scores, etc.)
- Benchmark results with numerical values
- Method comparisons with quantitative results

Paper text: {paper_text}

Return only the JSON, no other text.
""")

    computer_vision_focused: str = Field(default="""
Extract computer vision benchmarks and metrics. Return ONLY valid JSON:

{{
  "benchmarks": [
    {{
      "name": "dataset name",
      "category": "computer_vision",
      "standardized_key": "lowercase_key",
      "metric_type": "accuracy|map|iou|top1|top5",
      "description": "brief description"
    }}
  ],
  "metric_values": [
    {{
      "benchmark": "matching_key",
      "value": "numerical_result",
      "unit": "%|null",
      "context": "method_name"
    }}
  ]
}}

Look for: ImageNet, CIFAR, MNIST, COCO, Pascal VOC, Cityscapes, etc.

Paper text: {paper_text}

Return only JSON.
""")

    nlp_focused: str = Field(default="""
Extract NLP benchmarks and metrics. Return ONLY valid JSON:

{{
  "benchmarks": [
    {{
      "name": "dataset name",
      "category": "nlp",
      "standardized_key": "lowercase_key",
      "metric_type": "f1|bleu|rouge|accuracy|em",
      "description": "brief description"
    }}
  ],
  "metric_values": [
    {{
      "benchmark": "matching_key",
      "value": "numerical_result",
      "unit": "%|null",
      "context": "method_name"
    }}
  ]
}}

Look for: SQuAD, GLUE, Penn Treebank, WMT, BLEU, ROUGE, etc.

Paper text: {paper_text}

Return only JSON.
""")


class PDFProcessor:
    """Handles PDF text extraction and preprocessing."""

    @staticmethod
    def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess extracted text."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove common PDF artifacts
        text = text.replace("•", "").replace("", "")

        # Limit text length for LLM processing (approximate token limit)
        max_chars = 50000  # Roughly 12-15k tokens
        if len(text) > max_chars:
            # Try to cut at a sentence boundary
            text = text[:max_chars]
            last_period = text.rfind('.')
            if last_period > max_chars * 0.8:  # If we can find a reasonable cut point
                text = text[:last_period + 1]

        return text


@ray.remote
class BenchmarkExtractor:
    """Ray actor for LLM-based benchmark extraction."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.prompts = BenchmarkExtractionPrompt()

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Set it to use real LLM extraction.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key)
            logger.info(f"Initialized BenchmarkExtractor with model: {model_name}")

        logger.info(f"Initialized BenchmarkExtractor with model: {model_name}")

    async def extract_benchmarks(self, paper_text: str, domain_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract benchmarks from paper text using LLM.

        Args:
            paper_text: Full text content of the paper
            domain_hint: Optional domain hint (cv, nlp, etc.) for focused extraction

        Returns:
            Dictionary with extracted benchmarks and metrics
        """
        try:
            # Select appropriate prompt based on domain hint
            if domain_hint == "cv" or domain_hint == "computer_vision":
                prompt = self.prompts.computer_vision_focused.format(paper_text=paper_text)
            elif domain_hint == "nlp":
                prompt = self.prompts.nlp_focused.format(paper_text=paper_text)
            else:
                prompt = self.prompts.general_extraction.format(paper_text=paper_text)

            # Call real LLM API
            if self.client:
                result = await self._call_openai_api(prompt)
            else:
                # Fallback to mock if no API key
                logger.warning("No OpenAI API key found, using mock extraction")
                result = await self._mock_llm_call(prompt, paper_text)

            # Validate and structure the response
            return self._validate_extraction_result(result)

        except Exception as e:
            logger.error(f"Error in benchmark extraction: {e}")
            return {"benchmarks": [], "metric_values": [], "error": str(e)}

    async def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API for benchmark extraction."""
        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a research paper analysis expert. Extract benchmarks and metrics from research papers and return them in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # Parse JSON response
            try:
                result = json_lib.loads(content)
                return result
            except json_lib.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM JSON response: {e}")
                logger.debug(f"Raw response: {content}")
                # Try to extract JSON from the response if it's wrapped in other text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        result = json_lib.loads(json_match.group())
                        return result
                    except:
                        pass

                # If JSON parsing fails, return error
                return {
                    "benchmarks": [],
                    "metric_values": [],
                    "error": f"Could not parse LLM response as JSON: {content[:500]}..."
                }

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return {
                "benchmarks": [],
                "metric_values": [],
                "error": f"API call failed: {str(e)}"
            }

    async def _mock_llm_call(self, prompt: str, paper_text: str) -> Dict[str, Any]:
        """Mock LLM call that simulates intelligent benchmark extraction."""
        # Simulate processing time
        await asyncio.sleep(2.0)

        # Mock intelligent extraction based on common patterns
        mock_benchmarks = []
        mock_metrics = []

        # Look for common benchmark names in text (simplified pattern matching)
        text_lower = paper_text.lower()

        benchmark_patterns = {
            'imagenet': {'category': 'computer_vision', 'metric': 'accuracy'},
            'cifar-10': {'category': 'computer_vision', 'metric': 'accuracy'},
            'cifar-100': {'category': 'computer_vision', 'metric': 'accuracy'},
            'mnist': {'category': 'computer_vision', 'metric': 'accuracy'},
            'coco': {'category': 'computer_vision', 'metric': 'map'},
            'pascal voc': {'category': 'computer_vision', 'metric': 'map'},
            'squad': {'category': 'nlp', 'metric': 'f1'},
            'glue': {'category': 'nlp', 'metric': 'score'},
            'bleu': {'category': 'nlp', 'metric': 'bleu'},
            'penn treebank': {'category': 'nlp', 'metric': 'perplexity'},
            'wmt': {'category': 'nlp', 'metric': 'bleu'},
        }

        for benchmark_name, info in benchmark_patterns.items():
            if benchmark_name in text_lower:
                mock_benchmarks.append({
                    "name": benchmark_name.title().replace(' ', ''),
                    "category": info['category'],
                    "standardized_key": benchmark_name.replace(' ', '_').replace('-', '_'),
                    "metric_type": info['metric'],
                    "description": f"Standard {info['category']} benchmark"
                })

                # Mock some metric values
                if info['metric'] == 'accuracy':
                    value = f"{85 + (hash(benchmark_name) % 15):.1f}%"
                elif info['metric'] == 'map':
                    value = f"{40 + (hash(benchmark_name) % 30):.1f}"
                elif info['metric'] == 'f1':
                    value = f"{80 + (hash(benchmark_name) % 20):.1f}"
                else:
                    value = f"{hash(benchmark_name) % 100:.2f}"

                mock_metrics.append({
                    "benchmark": benchmark_name.replace(' ', '_').replace('-', '_'),
                    "value": value,
                    "unit": "%" if 'accuracy' in info['metric'] else None,
                    "context": "Proposed method"
                })

        return {
            "benchmarks": mock_benchmarks,
            "metric_values": mock_metrics
        }

    def _validate_extraction_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extraction results."""
        validated_benchmarks = []
        validated_metrics = []

        # Validate benchmarks
        for bench in result.get("benchmarks", []):
            try:
                benchmark = Benchmark(
                    name=bench.get("name", "Unknown"),
                    category=bench.get("category", "general"),
                    standardized_key=bench.get("standardized_key", bench.get("name", "").lower()),
                    metric_type=bench.get("metric_type", "unknown"),
                    description=bench.get("description")
                )
                validated_benchmarks.append(benchmark.model_dump())
            except Exception as e:
                logger.warning(f"Invalid benchmark data: {bench}, error: {e}")

        # Validate metrics
        for metric in result.get("metric_values", []):
            try:
                metric_value = MetricValue(
                    benchmark=metric.get("benchmark", "unknown"),
                    value=str(metric.get("value", "")),
                    unit=metric.get("unit"),
                    context=metric.get("context")
                )
                validated_metrics.append(metric_value.model_dump())
            except Exception as e:
                logger.warning(f"Invalid metric data: {metric}, error: {e}")

        return {
            "benchmarks": validated_benchmarks,
            "metric_values": validated_metrics,
            "extraction_metadata": {
                "timestamp": time.time(),
                "model_used": self.model_name,
                "benchmarks_found": len(validated_benchmarks),
                "metrics_found": len(validated_metrics)
            }
        }


class BenchmarkExtractionPipeline:
    """Main pipeline for benchmark extraction from research papers."""

    def __init__(self, num_extractors: int = 2):
        self.pdf_processor = PDFProcessor()
        self.num_extractors = num_extractors
        self.extractors = []

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create extractor actors
        self.extractors = [BenchmarkExtractor.remote() for _ in range(num_extractors)]
        logger.info(f"Initialized pipeline with {num_extractors} extractors")

    async def process_paper(self, paper_path: Union[str, Path],
                           domain_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single paper and extract benchmarks.

        Args:
            paper_path: Path to PDF file
            domain_hint: Optional domain hint for focused extraction

        Returns:
            Extracted benchmarks and metrics
        """
        logger.info(f"Processing paper: {paper_path}")

        # Extract text from PDF
        text = self.pdf_processor.extract_text_from_pdf(paper_path)
        if not text:
            return {"error": f"Could not extract text from {paper_path}"}

        # Clean and preprocess text
        clean_text = self.pdf_processor.clean_text(text)

        # Extract benchmarks using LLM
        extractor = self.extractors[0]  # Use first available extractor
        result = await extractor.extract_benchmarks.remote(clean_text, domain_hint)

        # Add paper metadata
        result["paper_metadata"] = {
            "file_path": str(paper_path),
            "file_name": Path(paper_path).name,
            "text_length": len(clean_text),
            "processing_timestamp": time.time()
        }

        logger.info(f"Extracted {len(result.get('benchmarks', []))} benchmarks "
                   f"and {len(result.get('metric_values', []))} metrics from {paper_path}")

        return result

    async def batch_process(self, papers_dir: Union[str, Path],
                           domain_hint: Optional[str] = None,
                           max_papers: Optional[int] = None) -> Dict[str, Any]:
        """
        Process multiple papers in parallel.

        Args:
            papers_dir: Directory containing PDF files
            domain_hint: Optional domain hint for all papers
            max_papers: Maximum number of papers to process

        Returns:
            Combined results from all papers
        """
        papers_dir = Path(papers_dir)
        pdf_files = list(papers_dir.glob("*.pdf"))

        if max_papers:
            pdf_files = pdf_files[:max_papers]

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Process papers in parallel using Ray
        tasks = []
        for i, pdf_file in enumerate(pdf_files):
            extractor = self.extractors[i % len(self.extractors)]

            # Extract text
            text = self.pdf_processor.extract_text_from_pdf(pdf_file)
            clean_text = self.pdf_processor.clean_text(text)

            # Create extraction task
            task = extractor.extract_benchmarks.remote(clean_text, domain_hint)
            tasks.append((pdf_file, task))

        # Wait for all tasks to complete
        results = {}
        all_benchmarks = {}
        all_metrics = []

        for pdf_file, task in tasks:
            try:
                result = await task

                # Add paper metadata
                result["paper_metadata"] = {
                    "file_path": str(pdf_file),
                    "file_name": pdf_file.name,
                    "processing_timestamp": time.time()
                }

                results[pdf_file.stem] = result

                # Aggregate benchmarks
                for benchmark in result.get("benchmarks", []):
                    key = benchmark["standardized_key"]
                    if key not in all_benchmarks:
                        all_benchmarks[key] = benchmark.copy()
                        all_benchmarks[key]["papers"] = []
                    all_benchmarks[key]["papers"].append(pdf_file.stem)

                # Collect all metrics
                all_metrics.extend(result.get("metric_values", []))

            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                results[pdf_file.stem] = {"error": str(e)}

        # Compile summary
        summary = {
            "papers_processed": len([r for r in results.values() if "error" not in r]),
            "papers_failed": len([r for r in results.values() if "error" in r]),
            "total_unique_benchmarks": len(all_benchmarks),
            "total_metrics_extracted": len(all_metrics),
            "processing_timestamp": time.time()
        }

        return {
            "summary": summary,
            "papers": results,
            "global_benchmarks": all_benchmarks,
            "all_metrics": all_metrics
        }

    def save_results(self, results: Dict[str, Any], output_file: Union[str, Path]):
        """Save extraction results to JSON file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Extract benchmarks from research papers")
    parser.add_argument("--paper_path", type=str, help="Path to single PDF file")
    parser.add_argument("--batch_process", type=str, help="Directory containing PDF files")
    parser.add_argument("--domain_hint", type=str, choices=["cv", "nlp", "general"],
                       default="general", help="Domain hint for focused extraction")
    parser.add_argument("--max_papers", type=int, help="Maximum number of papers to process")
    parser.add_argument("--output", type=str, default="benchmark_extraction_results.json",
                       help="Output file for results")
    parser.add_argument("--num_extractors", type=int, default=2,
                       help="Number of parallel extractors")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive mode for testing")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = BenchmarkExtractionPipeline(num_extractors=args.num_extractors)

    async def run():
        if args.interactive:
            # Interactive mode for testing and exploration
            print("Benchmark Extraction - Interactive Mode")
            print("Available sample papers:")
            sample_dir = Path("sample_papers")
            if sample_dir.exists():
                for i, pdf_file in enumerate(sample_dir.glob("*.pdf"), 1):
                    print(f"  {i}. {pdf_file.name}")

                choice = input("\nEnter paper number to process (or 'all' for batch): ")
                if choice.lower() == 'all':
                    results = await pipeline.batch_process(sample_dir, args.domain_hint, args.max_papers)
                else:
                    try:
                        idx = int(choice) - 1
                        pdf_files = list(sample_dir.glob("*.pdf"))
                        if 0 <= idx < len(pdf_files):
                            results = await pipeline.process_paper(pdf_files[idx], args.domain_hint)
                        else:
                            print("Invalid choice")
                            return
                    except ValueError:
                        print("Invalid input")
                        return

                print(f"\nResults preview:")
                print(f"Benchmarks found: {len(results.get('benchmarks', results.get('global_benchmarks', {})))}")
                print(f"Metrics extracted: {len(results.get('metric_values', results.get('all_metrics', [])))}")

                save = input("Save results? (y/n): ")
                if save.lower() == 'y':
                    pipeline.save_results(results, args.output)
            else:
                print("sample_papers directory not found")

        elif args.paper_path:
            # Single paper processing
            results = await pipeline.process_paper(args.paper_path, args.domain_hint)
            pipeline.save_results(results, args.output)
            print(f"Processed {args.paper_path}")
            print(f"Benchmarks: {len(results.get('benchmarks', []))}")
            print(f"Metrics: {len(results.get('metric_values', []))}")

        elif args.batch_process:
            # Batch processing
            results = await pipeline.batch_process(args.batch_process, args.domain_hint, args.max_papers)
            pipeline.save_results(results, args.output)
            print(f"Processed {results['summary']['papers_processed']} papers")
            print(f"Found {results['summary']['total_unique_benchmarks']} unique benchmarks")
            print(f"Extracted {results['summary']['total_metrics_extracted']} metrics")

        else:
            parser.print_help()

    # Run the async pipeline
    asyncio.run(run())


if __name__ == "__main__":
    main()