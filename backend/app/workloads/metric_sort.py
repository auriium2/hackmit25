#!/usr/bin/env python3
"""
Ray Data pipeline stage for metric consolidation and standardization.
Takes paper metrics from Ray datasets and creates standardized, comparable metrics
across research papers using distributed processing.
"""

import json
import re
import logging
from typing import Dict, List, Any, Tuple, Set, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field

import ray
import ray.data
from ray.data import Dataset
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PaperMetrics:
    """Structured representation of a paper's metrics for Ray Data processing."""
    paper_id: str
    title: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    standardized_metrics: Dict[str, Any] = field(default_factory=dict)
    processing_errors: List[str] = field(default_factory=list)


@dataclass
class MetricConsolidationResult:
    """Result of metric consolidation across multiple papers."""
    papers: List[PaperMetrics]
    metric_summary: Dict[str, Any]
    comparison_matrix: Dict[str, List[Dict[str, Any]]]
    all_metric_names: List[str]
    paper_count: int


class RayMetricProcessor:
    """Ray Data pipeline processor for standardizing research paper metrics."""

    def __init__(self, standardizer: Optional['MetricStandardizer'] = None):
        """Initialize with optional custom standardizer."""
        self.standardizer = standardizer or MetricStandardizer()

    def process_dataset(self, dataset: Dataset) -> Dataset:
        """
        Main pipeline method to process a Ray Dataset of papers with metrics.

        Args:
            dataset: Ray Dataset with records containing paper data and metrics

        Returns:
            Processed Dataset with standardized metrics
        """
        logger.info(f"Processing dataset with {dataset.count()} papers")

        # Transform each paper's metrics
        standardized_ds = dataset.map(
            self._standardize_paper_metrics,
            concurrency=4  # Parallel processing
        )

        # Add cross-paper analysis
        analyzed_ds = standardized_ds.map(
            self._add_cross_paper_analysis,
            concurrency=2
        )

        logger.info("Metric standardization pipeline completed")
        return analyzed_ds

    def consolidate_metrics(self, dataset: Dataset) -> MetricConsolidationResult:
        """
        Consolidate metrics across all papers in the dataset.

        Args:
            dataset: Ray Dataset with standardized paper metrics

        Returns:
            MetricConsolidationResult with consolidated data
        """
        # Collect all papers
        papers_data = dataset.take_all()
        papers = [PaperMetrics(**paper) for paper in papers_data]

        # Extract all metrics
        all_metrics = set()
        for paper in papers:
            all_metrics.update(paper.standardized_metrics.keys())

        # Create summary and comparison matrix
        metric_summary = self._create_distributed_summary(papers, all_metrics)
        comparison_matrix = self._create_comparison_matrix(papers, all_metrics)

        return MetricConsolidationResult(
            papers=papers,
            metric_summary=metric_summary,
            comparison_matrix=comparison_matrix,
            all_metric_names=sorted(all_metrics),
            paper_count=len(papers)
        )

    def _standardize_paper_metrics(self, paper_record: Dict[str, Any]) -> Dict[str, Any]:
        """Ray map function to standardize metrics for a single paper."""
        try:
            # Extract paper data
            paper_id = paper_record.get('paper_id', paper_record.get('id', 'unknown'))
            metrics = paper_record.get('metrics', {})

            # Initialize paper metrics object
            paper = PaperMetrics(
                paper_id=paper_id,
                title=paper_record.get('title'),
                metrics=metrics,
                metadata=paper_record.get('metadata', {})
            )

            # Standardize metrics
            standardized = {}
            for key, value in metrics.items():
                if key.startswith('_'):
                    continue

                try:
                    standard_key = self.standardizer._normalize_key(key)
                    normalized_value = self.standardizer._normalize_value(standard_key, value)

                    standardized[standard_key] = {
                        'value': normalized_value,
                        'original_key': key,
                        'original_value': value,
                        'confidence': self._calculate_confidence(key, value)
                    }
                except Exception as e:
                    paper.processing_errors.append(f"Error processing {key}: {str(e)}")
                    logger.warning(f"Failed to process metric {key} for paper {paper_id}: {e}")

            paper.standardized_metrics = standardized

            # Convert to dict for Ray serialization
            return {
                'paper_id': paper.paper_id,
                'title': paper.title,
                'metrics': paper.metrics,
                'metadata': paper.metadata,
                'standardized_metrics': paper.standardized_metrics,
                'processing_errors': paper.processing_errors,
                'metric_count': len(standardized)
            }

        except Exception as e:
            logger.error(f"Failed to process paper {paper_record.get('paper_id', 'unknown')}: {e}")
            return {
                'paper_id': paper_record.get('paper_id', 'error'),
                'processing_errors': [f"Critical processing error: {str(e)}"],
                'standardized_metrics': {},
                'metric_count': 0
            }

    def _add_cross_paper_analysis(self, paper_record: Dict[str, Any]) -> Dict[str, Any]:
        """Add cross-paper analysis metrics to each paper."""
        # Add analysis metadata
        paper_record['analysis'] = {
            'processed_at': pd.Timestamp.now().isoformat(),
            'pipeline_version': '1.0',
            'has_errors': len(paper_record.get('processing_errors', [])) > 0,
            'standardized_metric_count': paper_record.get('metric_count', 0)
        }

        return paper_record

    def _calculate_confidence(self, key: str, value: Any) -> float:
        """Calculate confidence score for metric standardization."""
        confidence = 1.0

        # Reduce confidence for ambiguous keys
        if len(key.split()) > 3:
            confidence -= 0.1

        # Reduce confidence for non-numeric values
        if isinstance(value, str) and not re.search(r'\d+', str(value)):
            confidence -= 0.2

        # Boost confidence for exact matches
        standard_key = self.standardizer._normalize_key(key)
        if key.lower() == standard_key:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _create_distributed_summary(self, papers: List[PaperMetrics],
                                  all_metrics: Set[str]) -> Dict[str, Any]:
        """Create metric summary using distributed processing."""
        summary = {}

        for metric_name in all_metrics:
            values = []
            papers_with_metric = []

            for paper in papers:
                if metric_name in paper.standardized_metrics:
                    metric_data = paper.standardized_metrics[metric_name]
                    values.append(metric_data['value'])
                    papers_with_metric.append(paper.paper_id)

            if values:
                summary[metric_name] = {
                    'appears_in_papers': len(papers_with_metric),
                    'unique_values': len(set(values)),
                    'all_values': values,
                    'papers_with_metric': papers_with_metric,
                    'coverage_percentage': len(papers_with_metric) / len(papers) * 100
                }

                # Add statistical analysis for numeric values
                numeric_values = self._extract_numeric_values(values)
                if numeric_values:
                    summary[metric_name].update({
                        'min_value': min(numeric_values),
                        'max_value': max(numeric_values),
                        'avg_value': sum(numeric_values) / len(numeric_values),
                        'numeric_count': len(numeric_values)
                    })

        return summary

    def _create_comparison_matrix(self, papers: List[PaperMetrics],
                                all_metrics: Set[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Create comparison matrix for papers and metrics."""
        matrix = {}

        for metric_name in all_metrics:
            matrix[metric_name] = []

            for paper in papers:
                if metric_name in paper.standardized_metrics:
                    metric_data = paper.standardized_metrics[metric_name]
                    matrix[metric_name].append({
                        'paper_id': paper.paper_id,
                        'value': metric_data['value'],
                        'confidence': metric_data.get('confidence', 1.0),
                        'original_key': metric_data['original_key']
                    })
                else:
                    matrix[metric_name].append({
                        'paper_id': paper.paper_id,
                        'value': None,
                        'confidence': 0.0,
                        'original_key': None
                    })

        return matrix

    def _extract_numeric_values(self, values: List[Any]) -> List[float]:
        """Extract numeric values from mixed list for statistical analysis."""
        numeric_values = []

        for val in values:
            match = re.search(r'(\d+\.?\d*)', str(val))
            if match:
                try:
                    numeric_values.append(float(match.group(1)))
                except ValueError:
                    continue

        return numeric_values


class MetricStandardizer:
    """Standardizes and consolidates metrics from multiple research papers."""

    def __init__(self):
        # Standard metric name mappings
        self.standard_names = {
            # Performance metrics
            'accuracy': ['accuracy', 'acc', 'accuracy_score', 'test_accuracy', 'validation_accuracy'],
            'precision': ['precision', 'prec', 'precision_score'],
            'recall': ['recall', 'rec', 'recall_score', 'sensitivity'],
            'f1_score': ['f1', 'f1_score', 'f1score', 'fscore', 'f_measure'],
            'bleu_score': ['bleu', 'bleu_score', 'bleuscore', 'bleu4'],
            'rouge_score': ['rouge', 'rouge_score', 'rouge_l', 'rouge_1', 'rouge_2'],
            'perplexity': ['perplexity', 'ppl', 'perp'],
            'top1_accuracy': ['top1', 'top_1', 'top1_accuracy', 'top1_acc'],
            'top5_accuracy': ['top5', 'top_5', 'top5_accuracy', 'top5_acc'],
            'map_score': ['map', 'mean_ap', 'mean_average_precision', 'map_score'],
            'ap_score': ['ap', 'average_precision', 'ap_score'],

            # Speed and timing metrics
            'training_time': ['training_time', 'train_time', 'training_duration', 'train_duration'],
            'inference_time': ['inference_time', 'inference_speed', 'prediction_time', 'test_time'],
            'fps': ['fps', 'frames_per_second', 'frame_rate'],
            'processing_speed': ['speed', 'processing_speed', 'throughput'],

            # Model characteristics
            'model_parameters': ['parameters', 'params', 'model_size', 'num_parameters', 'model_parameters'],
            'model_size': ['size', 'model_size', 'file_size', 'model_weight'],
            'epochs': ['epochs', 'num_epochs', 'training_epochs'],
            'batch_size': ['batch_size', 'batch', 'batchsize'],
            'learning_rate': ['lr', 'learning_rate', 'learn_rate', 'alpha'],

            # Hardware and system metrics
            'gpu_memory': ['gpu_memory', 'memory_usage', 'vram', 'gpu_ram'],
            'cpu_time': ['cpu_time', 'cpu_usage', 'computation_time'],
            'memory_usage': ['memory', 'ram_usage', 'memory_consumption'],

            # Robotics specific metrics
            'max_speed': ['speed', 'max_speed', 'maximum_speed', 'top_speed'],
            'control_frequency': ['frequency', 'control_freq', 'update_rate', 'hz'],
            'joint_count': ['joints', 'actuated_joints', 'joint_count', 'num_joints'],

            # Dataset metrics
            'dataset_size': ['dataset_size', 'data_size', 'num_samples', 'sample_count'],
            'error_rate': ['error', 'error_rate', 'err_rate', 'mistake_rate']
        }

        # Unit standardization patterns
        self.unit_patterns = {
            'percentage': [r'(\d+\.?\d*)\s*%', r'(\d+\.?\d*)\s*percent'],
            'seconds': [r'(\d+\.?\d*)\s*(?:sec|seconds?|s)(?:\b|$)', r'(\d+\.?\d*)\s*(?:hrs?|hours?)'],
            'milliseconds': [r'(\d+\.?\d*)\s*(?:ms|milliseconds?)(?:\b|$)'],
            'hertz': [r'(\d+\.?\d*)\s*(?:hz|hertz)(?:\b|$)'],
            'bytes': [r'(\d+\.?\d*)\s*(?:mb|gb|kb|bytes?)(?:\b|$)'],
            'parameters': [r'(\d+\.?\d*)\s*(?:[bmk]|million|billion|thousand)?(?:\s*parameters?)?(?:\b|$)'],
        }

        # Value standardization
        self.value_normalizers = {
            'percentage': self._normalize_percentage,
            'parameters': self._normalize_parameters,
            'time': self._normalize_time,
            'frequency': self._normalize_frequency,
            'size': self._normalize_size
        }

    def _normalize_key(self, key: str) -> str:
        """Normalize a metric key to standard form."""
        # Convert to lowercase and replace separators
        key_lower = key.lower().replace('-', '_').replace(' ', '_')

        # Remove common prefixes/suffixes
        key_clean = re.sub(r'^(test_|val_|validation_|train_|training_)', '', key_lower)
        key_clean = re.sub(r'(_score|_metric|_value)$', '', key_clean)

        # Find standard name
        for standard, variants in self.standard_names.items():
            if key_clean in [v.lower() for v in variants]:
                return standard

        return key_clean

    def _normalize_percentage(self, value: str) -> str:
        """Normalize percentage values."""
        # Extract number and ensure % suffix
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            num = match.group(1)
            return f"{num}%"
        return value

    def _normalize_parameters(self, value: str) -> str:
        """Normalize parameter counts (1M, 1B, etc.)."""
        value_lower = value.lower()

        # Extract number
        match = re.search(r'(\d+\.?\d*)', value_lower)
        if not match:
            return value

        num = float(match.group(1))

        # Determine scale
        if 'b' in value_lower or 'billion' in value_lower:
            return f"{num}B"
        elif 'm' in value_lower or 'million' in value_lower:
            return f"{num}M"
        elif 'k' in value_lower or 'thousand' in value_lower:
            return f"{num}K"
        else:
            return f"{int(num)}"

    def _normalize_time(self, value: str) -> str:
        """Normalize time values."""
        value_lower = value.lower()

        # Extract number
        match = re.search(r'(\d+\.?\d*)', value_lower)
        if not match:
            return value

        num = float(match.group(1))

        # Convert to standard units
        if 'hour' in value_lower or 'hr' in value_lower:
            return f"{num} hours"
        elif 'min' in value_lower:
            return f"{num} minutes"
        elif 'ms' in value_lower or 'millisecond' in value_lower:
            return f"{num}ms"
        else:
            return f"{num}s"

    def _normalize_frequency(self, value: str) -> str:
        """Normalize frequency values."""
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            return f"{match.group(1)} Hz"
        return value

    def _normalize_size(self, value: str) -> str:
        """Normalize size values."""
        value_lower = value.lower()
        match = re.search(r'(\d+\.?\d*)', value_lower)

        if not match:
            return value

        num = float(match.group(1))

        if 'gb' in value_lower:
            return f"{num}GB"
        elif 'mb' in value_lower:
            return f"{num}MB"
        elif 'kb' in value_lower:
            return f"{num}KB"
        else:
            return f"{num}B"

    def _normalize_value(self, key: str, value: str) -> str:
        """Normalize a metric value based on its key."""
        if not isinstance(value, str):
            return str(value)

        value_str = str(value).strip()

        # Apply appropriate normalizer based on key type
        if 'accuracy' in key or 'precision' in key or 'recall' in key or '%' in value_str:
            return self._normalize_percentage(value_str)
        elif 'parameter' in key or 'size' in key:
            if 'mb' in value_str.lower() or 'gb' in value_str.lower():
                return self._normalize_size(value_str)
            else:
                return self._normalize_parameters(value_str)
        elif 'time' in key or 'duration' in key:
            return self._normalize_time(value_str)
        elif 'frequency' in key or 'hz' in value_str.lower():
            return self._normalize_frequency(value_str)
        elif 'size' in key and ('mb' in value_str.lower() or 'gb' in value_str.lower()):
            return self._normalize_size(value_str)

        return value_str

    def consolidate_metrics(self, metric_dicts: List[Dict[str, Any]], paper_ids: List[str] = None) -> Dict[str, Any]:
        """
        Consolidate multiple metric dictionaries into a standardized format.

        Args:
            metric_dicts: List of metric dictionaries from different papers
            paper_ids: Optional list of paper identifiers

        Returns:
            Consolidated metrics dictionary with standardized keys and values
        """
        if paper_ids is None:
            paper_ids = [f"paper_{i}" for i in range(len(metric_dicts))]

        # Group metrics by standardized key
        metric_groups = defaultdict(dict)
        all_metrics = set()

        # Process each paper's metrics
        for i, metrics in enumerate(metric_dicts):
            paper_id = paper_ids[i]

            for key, value in metrics.items():
                # Skip metadata fields
                if key.startswith('_'):
                    continue

                # Normalize key and value
                standard_key = self._normalize_key(key)
                normalized_value = self._normalize_value(standard_key, value)

                # Store in groups
                metric_groups[standard_key][paper_id] = {
                    'value': normalized_value,
                    'original_key': key,
                    'original_value': value
                }

                all_metrics.add(standard_key)

        # Create consolidated result
        consolidated = {
            'standardized_metrics': dict(metric_groups),
            'metric_summary': self._create_metric_summary(metric_groups),
            'all_metric_names': sorted(all_metrics),
            'paper_count': len(metric_dicts),
            'papers': paper_ids
        }

        logger.info(f"Consolidated {len(all_metrics)} unique metrics from {len(metric_dicts)} papers")
        return consolidated

    def _create_metric_summary(self, metric_groups: Dict[str, Dict]) -> Dict[str, Any]:
        """Create a summary of metrics across all papers."""
        summary = {}

        for metric_name, papers in metric_groups.items():
            values = [data['value'] for data in papers.values()]

            summary[metric_name] = {
                'appears_in_papers': len(papers),
                'unique_values': len(set(values)),
                'all_values': values,
                'papers_with_metric': list(papers.keys())
            }

            # Add statistical info for numeric metrics
            numeric_values = []
            for val in values:
                # Extract numeric part
                match = re.search(r'(\d+\.?\d*)', str(val))
                if match:
                    numeric_values.append(float(match.group(1)))

            if numeric_values:
                summary[metric_name].update({
                    'min_value': min(numeric_values),
                    'max_value': max(numeric_values),
                    'avg_value': sum(numeric_values) / len(numeric_values)
                })

        return summary

    def get_comparison_matrix(self, consolidated_metrics: Dict[str, Any]) -> Dict[str, List]:
        """
        Create a comparison matrix showing which papers have which metrics.

        Args:
            consolidated_metrics: Output from consolidate_metrics()

        Returns:
            Matrix showing paper-metric relationships
        """
        papers = consolidated_metrics['papers']
        metrics = consolidated_metrics['standardized_metrics']

        # Create matrix
        matrix = {}
        for metric_name in metrics:
            matrix[metric_name] = []
            for paper_id in papers:
                if paper_id in metrics[metric_name]:
                    matrix[metric_name].append({
                        'paper': paper_id,
                        'value': metrics[metric_name][paper_id]['value']
                    })
                else:
                    matrix[metric_name].append({
                        'paper': paper_id,
                        'value': None
                    })

        return matrix


def consolidate_paper_metrics(metric_dicts: List[Dict[str, Any]], paper_ids: List[str] = None) -> Dict[str, Any]:
    """
    Legacy function to consolidate metrics from multiple papers (non-Ray version).

    Args:
        metric_dicts: List of metric dictionaries from LLM processors
        paper_ids: Optional list of paper identifiers

    Returns:
        Consolidated and standardized metrics

    Example:
        >>> metrics1 = {"Accuracy": "95.2%", "Training Time": "12 hours"}
        >>> metrics2 = {"accuracy": "87.1%", "train_time": "8 hrs"}
        >>> result = consolidate_paper_metrics([metrics1, metrics2])
        >>> print(result['standardized_metrics']['accuracy'])
    """
    standardizer = MetricStandardizer()
    return standardizer.consolidate_metrics(metric_dicts, paper_ids)


def create_ray_dataset_from_papers(papers_data: List[Dict[str, Any]]) -> Dataset:
    """
    Create a Ray Dataset from paper data for processing.

    Args:
        papers_data: List of paper dictionaries with metrics

    Returns:
        Ray Dataset ready for metric processing
    """
    return ray.data.from_items(papers_data)


def process_papers_with_ray(papers_data: List[Dict[str, Any]],
                           num_cpus: Optional[int] = None) -> MetricConsolidationResult:
    """
    Process papers using Ray Data pipeline for distributed metric standardization.

    Args:
        papers_data: List of paper dictionaries containing metrics
        num_cpus: Optional number of CPUs to use for processing

    Returns:
        MetricConsolidationResult with standardized metrics

    Example:
        >>> papers = [
        ...     {"paper_id": "paper1", "metrics": {"Accuracy": "95.2%"}},
        ...     {"paper_id": "paper2", "metrics": {"accuracy": "87.1%"}}
        ... ]
        >>> result = process_papers_with_ray(papers)
        >>> print(result.metric_summary)
    """
    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)

    # Create Ray dataset
    dataset = create_ray_dataset_from_papers(papers_data)

    # Process with Ray pipeline
    processor = RayMetricProcessor()
    processed_dataset = processor.process_dataset(dataset)

    # Consolidate results
    result = processor.consolidate_metrics(processed_dataset)

    logger.info(f"Processed {result.paper_count} papers with {len(result.all_metric_names)} unique metrics")
    return result


# Main execution for testing Ray pipeline
if __name__ == "__main__":
    import sys
    import asyncio

    # Test with sample data structured for Ray processing
    sample_papers = [
        {
            "paper_id": "GPT-4",
            "title": "Language Models are Few-Shot Learners",
            "metrics": {
                "Accuracy": "95.2%",
                "Training Time": "12 hours",
                "Model Parameters": "175B",
                "F1-Score": "0.94",
                "Max Speed": "3.7 m/s"
            },
            "metadata": {"venue": "NeurIPS", "year": 2020}
        },
        {
            "paper_id": "BERT-Large",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "metrics": {
                "accuracy": "87.1%",
                "train_time": "8 hrs",
                "parameters": "110M",
                "f1": "0.89",
                "maximum_speed": "2.1 m/s"
            },
            "metadata": {"venue": "NAACL", "year": 2019}
        },
        {
            "paper_id": "RoBERTa",
            "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "metrics": {
                "test_accuracy": "91.5%",
                "training_duration": "240 minutes",
                "model_size": "340M",
                "fscore": "0.91",
                "speed": "4.2 m/s"
            },
            "metadata": {"venue": "arXiv", "year": 2019}
        }
    ]

    print("🚀 Testing Ray Data Pipeline for Metric Standardization")
    print("=" * 70)

    try:
        # Test Ray pipeline
        if "--ray" in sys.argv:
            print("🔄 Processing with Ray Data pipeline...")
            result = process_papers_with_ray(sample_papers)

            print(f"📊 Ray processed {result.paper_count} papers with {len(result.all_metric_names)} unique metrics")
            print(f"📋 Papers: {[p.paper_id for p in result.papers]}")
            print()

            # Show standardized metrics
            print("📈 Standardized Metrics (Ray Pipeline):")
            for metric_name in sorted(result.all_metric_names):
                print(f"\n🔍 {metric_name}:")
                if metric_name in result.metric_summary:
                    summary = result.metric_summary[metric_name]
                    print(f"   Coverage: {summary['coverage_percentage']:.1f}% of papers")
                    print(f"   Unique values: {summary['unique_values']}")

                    if 'avg_value' in summary:
                        print(f"   Average: {summary['avg_value']:.2f}")
                        print(f"   Range: {summary['min_value']:.2f} - {summary['max_value']:.2f}")

            # Show comparison matrix sample
            print(f"\n📊 Comparison Matrix Sample:")
            sample_metric = list(result.comparison_matrix.keys())[0]
            print(f"Metric: {sample_metric}")
            for entry in result.comparison_matrix[sample_metric]:
                confidence = entry['confidence']
                status = "✅" if confidence > 0.8 else "⚠️" if confidence > 0.5 else "❌"
                print(f"   {status} {entry['paper_id']}: {entry['value']} (confidence: {confidence:.2f})")

        else:
            # Test legacy function
            print("🔄 Processing with legacy function...")
            legacy_metrics = [paper["metrics"] for paper in sample_papers]
            legacy_ids = [paper["paper_id"] for paper in sample_papers]
            result = consolidate_paper_metrics(legacy_metrics, legacy_ids)

            print(f"📊 Legacy processed {result['paper_count']} papers with {len(result['all_metric_names'])} unique metrics")
            print(f"📋 Papers: {', '.join(result['papers'])}")

        print("\n" + "=" * 70)
        print("✅ Metric standardization completed successfully!")
        print("\n💡 Use '--ray' flag to test Ray Data pipeline")
        print("💡 Use '--json' flag to output JSON results")

        # Output as JSON if requested
        if "--json" in sys.argv:
            print("\n📄 JSON Output:")
            if hasattr(result, '__dict__'):
                # Convert dataclass to dict for JSON serialization
                import dataclasses
                result_dict = dataclasses.asdict(result)
                print(json.dumps(result_dict, indent=2, default=str))
            else:
                print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

    finally:
        # Cleanup Ray if initialized
        if ray.is_initialized():
            ray.shutdown()