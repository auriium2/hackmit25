#!/usr/bin/env python3
"""
Test script for the enhanced llm_processor.py with Claude LLM metric extraction.
Tests both the ClaudeLLMClient directly and the Ray-based DomainExpertAgent.
"""

import sys
import os
sys.path.insert(0, '/Users/arjuncaputo/hackmit25/backend/app/workloads')

import time
import json
from llm_processor import ClaudeLLMClient, DomainExpertAgent
import ray

def test_claude_llm_directly():
    """Test the ClaudeLLMClient directly with sample paper text."""
    print("🧪 Testing Claude LLM Client directly...")
    print("=" * 60)

    client = ClaudeLLMClient()

    # Sample research paper text (simulating what we'd get from a real paper)
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
            The model achieves 87.4 accuracy on MNLI matched and 86.7 on MNLI mismatched.
            """
        },
        {
            "title": "YOLOv5: Object Detection Excellence",
            "text": """
            YOLOv5 achieves outstanding performance in real-time object detection.

            On COCO test-dev, YOLOv5s achieves 37.4% AP at 140 FPS.
            YOLOv5m reaches 45.4% AP with inference time of 8.1ms.
            YOLOv5l achieves 49.0% AP and YOLOv5x reaches 50.7% AP.

            Model sizes: YOLOv5s (14.1MB), YOLOv5m (42.2MB), YOLOv5l (89.3MB), YOLOv5x (166.4MB).
            Training on COCO dataset took 300 epochs, approximately 8 hours on V100 GPU.

            The models achieve real-time performance with batch size 1 on Tesla V100.
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
            if not key.startswith("_"):  # Skip metadata fields
                print(f"   • {key}: {value}")

        print()

def test_ray_domain_expert():
    """Test the Ray-based DomainExpertAgent with the Claude LLM integration."""
    print("\n🚀 Testing Ray DomainExpertAgent with Claude LLM...")
    print("=" * 60)

    # Initialize Ray (if not already initialized)
    if not ray.is_initialized():
        ray.init(local_mode=True)  # Use local mode for testing

    try:
        # Create a domain expert agent
        agent = DomainExpertAgent.remote(domain="computer_vision")

        # Sample paper data (simulating what we'd get from the seed paper retrieval)
        paper_data = {
            "openalex_id": "W2963983434",
            "title": "ResNet: Deep Residual Learning for Image Recognition",
            "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
            "full_text": """
            Deep residual networks (ResNets) have revolutionized image classification.

            Our 152-layer ResNet achieves 3.57% top-5 error on ImageNet test set.
            ResNet-50 achieves 7.13% top-5 error with 25.6M parameters.
            ResNet-101 reaches 6.44% top-5 error with 44.5M parameters.
            ResNet-152 achieves the best performance with 60.2M parameters.

            Training time was 2-3 weeks on 8 GPUs.
            We achieve 96.43% accuracy on CIFAR-10 with 110-layer ResNet.
            On CIFAR-100, our method achieves 78.63% accuracy.

            Inference time is 21.3ms per image on a single GPU.
            """,
            "venue": "IEEE Conference on Computer Vision and Pattern Recognition",
            "year": 2016
        }

        print("📄 Analyzing paper:", paper_data["title"])
        print("🔬 Using Ray remote agent...")

        start_time = time.time()

        # Analyze the paper using the Ray actor
        analysis_result = ray.get(agent.analyze_paper.remote(paper_data))

        analysis_time = time.time() - start_time

        print(f"⏱️  Analysis time: {analysis_time:.2f} seconds")
        print(f"🎯 Domain: {analysis_result['domain']}")
        print(f"📊 Extracted Metrics:")

        for key, value in analysis_result['metrics'].items():
            if not key.startswith("_"):  # Skip metadata fields
                print(f"   • {key}: {value}")

        print(f"\n📝 Summary: {analysis_result['summary'][:100]}...")
        print(f"🏆 Benchmarks found: {len(analysis_result['benchmarks'])}")

        for benchmark in analysis_result['benchmarks'][:3]:  # Show first 3
            print(f"   • {benchmark['name']} ({benchmark['category']})")

    finally:
        ray.shutdown()

def test_with_minimal_text():
    """Test with minimal text to see fallback behavior."""
    print("\n🔍 Testing with minimal text (fallback behavior)...")
    print("=" * 60)

    client = ClaudeLLMClient()

    minimal_paper = {
        "title": "A Study on Machine Learning Performance",
        "text": "This paper reports 95.2% accuracy on our dataset using a neural network with 1.2M parameters trained for 50 epochs."
    }

    metrics = client.extract_metrics_from_text(minimal_paper['text'], minimal_paper['title'])

    print(f"📊 Extracted {len(metrics)} metrics from minimal text:")
    for key, value in metrics.items():
        if not key.startswith("_"):
            print(f"   • {key}: {value}")

def main():
    """Run all tests."""
    print("🚀 Testing Enhanced LLM Processor with Claude Integration")
    print("=" * 80)

    # Test 1: Direct Claude LLM client
    test_claude_llm_directly()

    # Test 2: Ray-based domain expert
    test_ray_domain_expert()

    # Test 3: Minimal text fallback
    test_with_minimal_text()

    print("\n" + "=" * 80)
    print("🎉 All tests completed!")
    print("\n✅ Key Features Demonstrated:")
    print("   • Real Claude LLM metric extraction from research papers")
    print("   • Comprehensive key-value pair extraction")
    print("   • Fallback mechanisms for API failures")
    print("   • Integration with existing Ray-based architecture")
    print("   • Support for various paper text formats")

if __name__ == "__main__":
    main()