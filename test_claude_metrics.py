#!/usr/bin/env python3
"""
Simple test for Claude LLM metric extraction without Ray dependencies.
"""

import sys
import os
import time
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ClaudeLLMClient:
    """Client for Claude API to perform metric extraction from research papers."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            print("⚠️  Claude API key not found. Falling back to mock implementation.")
            self.enabled = False
        else:
            self.enabled = True
            print("✅ Claude LLM client initialized successfully")

    def extract_metrics_from_text(self, paper_text, paper_title=""):
        """Extract key-value pairs of metrics from research paper text using Claude LLM."""
        if not self.enabled:
            return self._fallback_metric_extraction(paper_text, paper_title)

        # Truncate text if too long
        max_chars = 15000
        if len(paper_text) > max_chars:
            paper_text = paper_text[:max_chars] + "...[truncated]"

        prompt = f"""
You are a research paper analysis expert. Extract ALL quantitative metrics and performance measurements from the following research paper text.

Paper Title: {paper_title}

Paper Text:
{paper_text}

Extract metrics as key-value pairs. Include:
- Performance metrics (accuracy, precision, recall, F1-score, BLEU, ROUGE, etc.)
- Computational metrics (training time, inference time, memory usage, FLOPs)
- Dataset statistics (dataset size, number of parameters, epochs)
- Benchmarks scores (on specific datasets like ImageNet, CIFAR, etc.)
- Error rates (RMSE, MAE, cross-entropy loss, perplexity, etc.)
- Model characteristics (model size, depth, width, etc.)

Return ONLY a valid JSON object with metric names as keys and their values as strings. Include the units when available.
Example: {{"accuracy": "94.2%", "training_time": "12 hours", "model_parameters": "175B", "BLEU_score": "42.1"}}

JSON:"""

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            response.raise_for_status()

            content = response.json()["content"][0]["text"].strip()

            try:
                metrics = json.loads(content)
                print(f"✅ Successfully extracted {len(metrics)} metrics using Claude LLM")
                return metrics if isinstance(metrics, dict) else {}
            except json.JSONDecodeError:
                print("⚠️  Failed to parse Claude response as JSON, attempting to extract from text")
                return self._parse_metrics_from_text(content)

        except Exception as e:
            print(f"❌ Error calling Claude API: {e}")
            return self._fallback_metric_extraction(paper_text, paper_title)

    def _parse_metrics_from_text(self, text):
        """Attempt to extract metrics from raw text response."""
        metrics = {}
        lines = text.split('\n')

        for line in lines:
            if ':' in line and any(char.isdigit() for char in line):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('"').strip("'")
                    value = parts[1].strip().strip(',').strip('"').strip("'")
                    if key and value:
                        metrics[key] = value

        return metrics

    def _fallback_metric_extraction(self, paper_text, paper_title):
        """Simple regex-based extraction when Claude API is not available."""
        import re

        metrics = {}
        text = paper_text.lower()

        # Accuracy patterns
        accuracy_matches = re.findall(r'accuracy[:\s]+(\d+\.?\d*%?)', text)
        if accuracy_matches:
            metrics["accuracy"] = accuracy_matches[0] + ("%" if "%" not in accuracy_matches[0] else "")

        # F1 Score patterns
        f1_matches = re.findall(r'f1[:\s-]+(\d+\.?\d*)', text)
        if f1_matches:
            metrics["f1_score"] = f1_matches[0]

        # BLEU score patterns
        bleu_matches = re.findall(r'bleu[:\s-]+(\d+\.?\d*)', text)
        if bleu_matches:
            metrics["bleu_score"] = bleu_matches[0]

        # Parameters
        param_matches = re.findall(r'(\d+\.?\d*[bmk]?)\s*parameters', text)
        if param_matches:
            metrics["model_parameters"] = param_matches[0]

        print(f"📊 Fallback extraction found {len(metrics)} metrics")
        return metrics

def test_claude_metric_extraction():
    """Test the Claude metric extraction with sample papers."""
    print("🧪 Testing Claude LLM Metric Extraction")
    print("=" * 60)

    client = ClaudeLLMClient()

    # Sample research papers
    sample_papers = [
        {
            "title": "Attention Is All You Need",
            "text": """
            We propose the Transformer, a novel neural network architecture based solely on attention mechanisms.

            Our model achieves 28.4 BLEU score on the WMT 2014 English-German translation task, establishing a new state-of-the-art.
            Training time was reduced to 12 hours using 8 P100 GPUs.

            The model has 65M parameters for the base model and 213M parameters for the large model.
            We achieved 99.6% accuracy on the Penn Treebank parsing task.

            Training took 3.5 days on 8 P100 GPUs for the large model.
            Our approach achieves perplexity of 1.73 on the Penn Treebank test set.
            Inference time is 20ms per sequence.
            """
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "text": """
            We introduce BERT, a new method for pre-training language representations.

            BERT achieves new state-of-the-art results on eleven natural language processing tasks.
            On the GLUE benchmark, BERT Base achieves a score of 78.3% and BERT Large achieves 80.5%.

            Our model contains 110M parameters for BERT-Base and 340M parameters for BERT-Large.
            Training took 4 days on 4 Cloud TPUs (16 TPU chips total).

            On SQuAD v1.1, BERT achieves 93.2 F1 score and 88.5 exact match.
            The model achieves 87.4% accuracy on MNLI matched and 86.7% on MNLI mismatched.
            Fine-tuning takes only 1 hour on a single Cloud TPU.
            """
        },
        {
            "title": "YOLOv5: Real-Time Object Detection",
            "text": """
            YOLOv5 achieves outstanding performance in real-time object detection.

            On COCO test-dev, YOLOv5s achieves 37.4% AP at 140 FPS.
            YOLOv5m reaches 45.4% AP with inference time of 8.1ms.
            YOLOv5l achieves 49.0% AP and YOLOv5x reaches 50.7% AP.

            Model sizes: YOLOv5s (14.1MB), YOLOv5m (42.2MB), YOLOv5l (89.3MB), YOLOv5x (166.4MB).
            Training on COCO dataset took 300 epochs, approximately 8 hours on V100 GPU.

            The models achieve real-time performance with batch size 32 on Tesla V100.
            Memory usage during training: YOLOv5s (16GB), YOLOv5x (32GB).
            """
        }
    ]

    for i, paper in enumerate(sample_papers, 1):
        print(f"\n📄 Test {i}: {paper['title']}")
        print("-" * 50)

        start_time = time.time()
        metrics = client.extract_metrics_from_text(paper['text'], paper['title'])
        extraction_time = time.time() - start_time

        print(f"⏱️  Extraction time: {extraction_time:.2f} seconds")
        print(f"📊 Extracted {len(metrics)} metrics:")

        for key, value in metrics.items():
            print(f"   • {key}: {value}")

        print()

    print("=" * 60)
    print("🎉 Metric extraction testing completed!")
    print("\n✅ Features demonstrated:")
    print("   • Real-time Claude LLM metric extraction")
    print("   • Key-value pair format output")
    print("   • Performance metrics (accuracy, F1, BLEU, AP)")
    print("   • Computational metrics (training time, inference time)")
    print("   • Model characteristics (parameters, model size)")
    print("   • Robust fallback mechanisms")

if __name__ == "__main__":
    test_claude_metric_extraction()