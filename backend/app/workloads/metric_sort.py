#!/usr/bin/env python3
"""
Metric consolidation and standardization system.
Takes multiple key-value metric dictionaries from LLM processors and creates
a unified, standardized set of metrics for research paper comparison.
"""

import json
import re
from typing import Dict, List, Any, Tuple, Set
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    Main function to consolidate metrics from multiple papers.

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


# Main execution for testing
if __name__ == "__main__":
    import sys

    # Test with sample data
    sample_metrics = [
        {
            "Accuracy": "95.2%",
            "Training Time": "12 hours",
            "Model Parameters": "175B",
            "F1-Score": "0.94",
            "Max Speed": "3.7 m/s"
        },
        {
            "accuracy": "87.1%",
            "train_time": "8 hrs",
            "parameters": "110M",
            "f1": "0.89",
            "maximum_speed": "2.1 m/s"
        },
        {
            "test_accuracy": "91.5%",
            "training_duration": "240 minutes",
            "model_size": "340M",
            "fscore": "0.91",
            "speed": "4.2 m/s"
        }
    ]

    paper_names = ["GPT-4", "BERT-Large", "RoBERTa"]

    print("🔧 Testing Metric Standardization System")
    print("=" * 60)

    # Consolidate metrics
    result = consolidate_paper_metrics(sample_metrics, paper_names)

    print(f"📊 Consolidated {len(result['all_metric_names'])} unique metrics from {result['paper_count']} papers")
    print(f"📋 Papers: {', '.join(result['papers'])}")
    print()

    print("📈 Standardized Metrics:")
    for metric_name in sorted(result['all_metric_names']):
        print(f"\n🔍 {metric_name}:")
        papers_data = result['standardized_metrics'][metric_name]
        for paper_id, data in papers_data.items():
            print(f"   • {paper_id}: {data['value']} (was: {data['original_key']} = {data['original_value']})")

    print("\n" + "=" * 60)
    print("✅ Metric standardization completed successfully!")

    # Output as JSON if requested
    if "--json" in sys.argv:
        print("\n📄 JSON Output:")
        print(json.dumps(result, indent=2))