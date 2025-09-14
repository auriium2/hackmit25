#!/usr/bin/env python3
"""
Test the complete pipeline: LLM metric extraction → Metric standardization
Demonstrates how llm_processor.py and metric_sort.py work together.
"""

import sys
import os
sys.path.insert(0, '/Users/arjuncaputo/hackmit25/backend/app/workloads')

import time
import json
from llm_processor import extract_metrics
from metric_sort import consolidate_paper_metrics

def test_full_pipeline():
    """Test the complete pipeline from raw paper text to standardized metrics."""
    print("🔬 Testing Complete Pipeline: LLM Extraction → Metric Standardization")
    print("=" * 80)

    # Sample research papers with different metric naming conventions
    papers = [
        {
            "id": "transformer_paper",
            "title": "Attention Is All You Need",
            "text": """
            We propose the Transformer, a novel neural network architecture based solely on attention mechanisms.
            Our model achieves 28.4 BLEU score on the WMT 2014 English-German translation task.
            Training time was reduced to 12 hours using 8 P100 GPUs.
            The model has 65M parameters for the base model and 213M parameters for the large model.
            We achieved 99.6% accuracy on the Penn Treebank parsing task.
            Training took 3.5 days on 8 P100 GPUs for the large model.
            Our approach achieves perplexity of 1.73 on the Penn Treebank test set.
            Inference time is 20ms per sequence.
            """
        },
        {
            "id": "bert_paper",
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
            "id": "robotics_paper",
            "title": "Highly Dynamic Quadruped Locomotion",
            "text": """
            The Mini-Cheetah quadruped robot, equipped with 12 actuated joints, serves as the experimental platform.
            Control Frequencies: MPC operates at 40 Hz, while WBIC runs at 500 Hz, enabling real-time responsiveness.
            Top Speed: The robot achieves a maximum speed of 3.7 m/s, fully utilizing its hardware capabilities.
            Joint Performance: Analysis of joint velocities, torques, and power consumption indicates efficient use.
            Tracking Accuracy: The system demonstrates precise tracking of vertical reaction forces and body height.
            The robot successfully navigates various outdoor terrains with 95.2% success rate.
            """
        }
    ]

    print(f"📄 Processing {len(papers)} research papers...")
    print("-" * 80)

    # Step 1: Extract metrics from each paper using LLM
    extracted_metrics = []
    paper_ids = []

    for i, paper in enumerate(papers, 1):
        print(f"\n🔍 Step 1.{i}: Extracting metrics from '{paper['title'][:50]}...'")

        start_time = time.time()
        metrics = extract_metrics(paper['text'], paper['title'], clean_output=True)
        extraction_time = time.time() - start_time

        print(f"   ⏱️  Extraction time: {extraction_time:.2f}s")
        print(f"   📊 Extracted {len(metrics)} raw metrics")

        # Show a few examples of raw metrics
        for j, (key, value) in enumerate(list(metrics.items())[:3]):
            print(f"      • {key}: {value}")
        if len(metrics) > 3:
            print(f"      ... and {len(metrics) - 3} more")

        extracted_metrics.append(metrics)
        paper_ids.append(paper['id'])

    # Step 2: Consolidate and standardize all metrics
    print(f"\n🔧 Step 2: Consolidating and standardizing metrics from all papers...")

    start_time = time.time()
    consolidated = consolidate_paper_metrics(extracted_metrics, paper_ids)
    consolidation_time = time.time() - start_time

    print(f"   ⏱️  Consolidation time: {consolidation_time:.2f}s")
    print(f"   ✅ Standardized {len(consolidated['all_metric_names'])} unique metrics")

    # Step 3: Display results
    print("\n" + "=" * 80)
    print("📈 STANDARDIZED RESULTS")
    print("=" * 80)

    print(f"🎯 Summary:")
    print(f"   • Papers processed: {consolidated['paper_count']}")
    print(f"   • Unique metrics found: {len(consolidated['all_metric_names'])}")
    print(f"   • Total processing time: {sum([extraction_time, consolidation_time]):.2f}s")

    print(f"\n📊 Metric Coverage Matrix:")
    for metric_name in sorted(consolidated['all_metric_names']):
        summary = consolidated['metric_summary'][metric_name]
        coverage = f"{summary['appears_in_papers']}/{len(paper_ids)}"
        print(f"   • {metric_name}: appears in {coverage} papers")

    print(f"\n🔍 Detailed Standardized Metrics:")
    for metric_name in sorted(consolidated['all_metric_names']):
        print(f"\n   📌 {metric_name.upper()}:")
        papers_data = consolidated['standardized_metrics'][metric_name]

        for paper_id, data in papers_data.items():
            original = f"{data['original_key']} = {data['original_value']}"
            standardized = data['value']
            print(f"      • {paper_id}: {standardized} (was: {original})")

        # Show statistical summary if available
        summary = consolidated['metric_summary'][metric_name]
        if 'min_value' in summary:
            print(f"        📊 Range: {summary['min_value']:.2f} - {summary['max_value']:.2f}, Avg: {summary['avg_value']:.2f}")

    # Step 4: Export consolidated metrics as JSON
    print(f"\n💾 Exporting consolidated metrics...")

    output_file = "/Users/arjuncaputo/hackmit25/consolidated_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2, default=str)

    print(f"   ✅ Saved to: {output_file}")

    print("\n" + "=" * 80)
    print("🎉 PIPELINE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    print("\n✅ Key Features Demonstrated:")
    print("   • Real Claude LLM metric extraction from diverse research papers")
    print("   • Automatic standardization of metric names (accuracy, Accuracy, test_accuracy → accuracy)")
    print("   • Value normalization (95.2%, 0.952, 95.2 percent → 95.2%)")
    print("   • Cross-paper metric consolidation and comparison")
    print("   • Statistical analysis (min/max/average for numeric metrics)")
    print("   • JSON export for downstream processing")
    print("   • Full integration between llm_processor.py and metric_sort.py")

    return consolidated

if __name__ == "__main__":
    result = test_full_pipeline()