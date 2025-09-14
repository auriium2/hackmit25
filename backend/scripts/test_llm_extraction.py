#!/usr/bin/env python3
"""
Simple test script to verify LLM-based benchmark extraction works.
Set OPENAI_API_KEY environment variable before running.
"""

import os
import asyncio
import ray
from extract_benchmarks import BenchmarkExtractionPipeline

async def test_single_paper():
    """Test extraction on a single sample paper."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it first:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return

    print("🚀 Testing LLM-based benchmark extraction...")

    # Initialize pipeline
    pipeline = BenchmarkExtractionPipeline(num_extractors=1)

    # Test with first sample paper
    sample_path = "sample_papers/1206.5538.pdf"

    print(f"📄 Processing: {sample_path}")

    try:
        result = await pipeline.process_paper(sample_path, domain_hint="cv")

        print("\n✅ Extraction completed!")
        print(f"📊 Benchmarks found: {len(result.get('benchmarks', []))}")
        print(f"📈 Metrics extracted: {len(result.get('metric_values', []))}")

        # Show some results
        if result.get('benchmarks'):
            print("\n🎯 Benchmarks:")
            for bench in result['benchmarks'][:3]:  # Show first 3
                print(f"  - {bench['name']} ({bench['category']}) - {bench['metric_type']}")

        if result.get('metric_values'):
            print("\n📊 Metrics:")
            for metric in result['metric_values'][:3]:  # Show first 3
                unit = f" {metric['unit']}" if metric.get('unit') else ""
                print(f"  - {metric['benchmark']}: {metric['value']}{unit} ({metric.get('context', 'N/A')})")

        # Save results
        pipeline.save_results(result, "llm_test_results.json")
        print("\n💾 Results saved to llm_test_results.json")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # Cleanup Ray
        ray.shutdown()

if __name__ == "__main__":
    asyncio.run(test_single_paper())