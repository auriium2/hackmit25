#!/usr/bin/env python3
"""
Test Claude LLM metric extraction with the specific robotics paper text.
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
- Robotics metrics (speed, frequency, joint count, tracking accuracy, etc.)
- System specifications (control frequencies, hardware details)

Return ONLY a valid JSON object with metric names as keys and their values as strings. Include the units when available.
Example: {{"accuracy": "94.2%", "training_time": "12 hours", "model_parameters": "175B", "max_speed": "3.7 m/s"}}

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

        # Speed patterns
        speed_matches = re.findall(r'(\d+\.?\d*)\s*m/s', text)
        if speed_matches:
            metrics["max_speed"] = f"{speed_matches[0]} m/s"

        # Frequency patterns
        freq_matches = re.findall(r'(\d+\.?\d*)\s*hz', text)
        if freq_matches:
            metrics["control_frequency"] = f"{freq_matches[0]} Hz"

        # Joint patterns
        joint_matches = re.findall(r'(\d+)\s*(?:actuated\s+)?joints?', text)
        if joint_matches:
            metrics["joint_count"] = joint_matches[0]

        print(f"📊 Fallback extraction found {len(metrics)} metrics")
        return metrics

def test_robotics_paper():
    """Test metric extraction with the specific robotics paper."""
    print("🤖 Testing LLM Processor with Robotics Paper")
    print("=" * 70)

    client = ClaudeLLMClient()

    paper_title = "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control"

    paper_text = """The paper titled "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control and Model Predictive Control" by Donghyun Kim et al. presents an advanced control framework that enables high-speed, agile locomotion for quadruped robots. The approach integrates Model Predictive Control (MPC) and Whole-Body Impulse Control (WBIC) to achieve dynamic and stable movement, even during aerial phases.

🧠 Key Contributions

Whole-Body Impulse Control (WBIC): The authors introduce a WBIC formulation that focuses on tracking reaction forces rather than predefined body trajectories. This method is particularly effective for handling dynamic locomotion scenarios with frequent aerial phases.

Model Predictive Control (MPC): MPC is employed to compute an optimal reaction force profile over a longer time horizon using a simplified model. This approach allows for the anticipation of future dynamics and enhances the robot's ability to adapt to changing conditions.

Integration of MPC and WBIC: The combination of MPC and WBIC provides a robust control architecture that leverages the strengths of both methods. MPC handles long-term planning, while WBIC addresses real-time execution, ensuring stability and agility.

⚙️ Experimental Setup

Robot Platform: The Mini-Cheetah quadruped robot, equipped with 12 actuated joints, serves as the experimental platform.

Control Frequencies: MPC operates at 40 Hz, while WBIC runs at 500 Hz, enabling real-time responsiveness.

Gait Variations: The system is tested across six different gait patterns to demonstrate versatility.

Environments: Experiments are conducted both indoors on a treadmill and outdoors on various terrains to assess robustness.

📊 Performance Metrics

Top Speed: The robot achieves a maximum speed of 3.7 m/s, fully utilizing its hardware capabilities.

Joint Performance: Analysis of joint velocities, torques, and power consumption indicates efficient use of the robot's actuators during high-speed running.

Tracking Accuracy: The system demonstrates precise tracking of vertical reaction forces and body height, maintaining stability during jumps and landings.

🏞️ Real-World Demonstrations

Outdoor Navigation: The robot successfully navigates various outdoor terrains, showcasing its ability to handle rough and uneven surfaces.

Push-Recovery: In scenarios where external disturbances are applied, the system effectively recovers, maintaining balance and stability.

This work represents a significant advancement in legged robotics, providing a practical solution for achieving high-speed, dynamic locomotion in real-world environments. The integration of MPC and WBIC offers a scalable and effective approach to complex locomotion tasks."""

    print(f"📄 Paper: {paper_title}")
    print(f"📝 Text length: {len(paper_text)} characters")
    print("🔍 Extracting metrics...")
    print("-" * 70)

    start_time = time.time()
    metrics = client.extract_metrics_from_text(paper_text, paper_title)
    extraction_time = time.time() - start_time

    print(f"⏱️  Extraction time: {extraction_time:.2f} seconds")
    print(f"📊 Extracted {len(metrics)} metrics:")
    print()

    # Organize metrics by category for better display
    performance_metrics = {}
    control_metrics = {}
    hardware_metrics = {}
    other_metrics = {}

    for key, value in metrics.items():
        if any(term in key.lower() for term in ['speed', 'accuracy', 'performance', 'tracking']):
            performance_metrics[key] = value
        elif any(term in key.lower() for term in ['frequency', 'hz', 'control', 'mpc', 'wbic']):
            control_metrics[key] = value
        elif any(term in key.lower() for term in ['joint', 'actuated', 'robot', 'platform']):
            hardware_metrics[key] = value
        else:
            other_metrics[key] = value

    if performance_metrics:
        print("🏃 Performance Metrics:")
        for key, value in performance_metrics.items():
            print(f"   • {key}: {value}")
        print()

    if control_metrics:
        print("🎛️  Control System Metrics:")
        for key, value in control_metrics.items():
            print(f"   • {key}: {value}")
        print()

    if hardware_metrics:
        print("⚙️  Hardware Metrics:")
        for key, value in hardware_metrics.items():
            print(f"   • {key}: {value}")
        print()

    if other_metrics:
        print("📋 Other Metrics:")
        for key, value in other_metrics.items():
            print(f"   • {key}: {value}")
        print()

    print("=" * 70)
    print("🎯 Summary:")
    print(f"   • Total metrics extracted: {len(metrics)}")
    print(f"   • Processing time: {extraction_time:.2f}s")
    print(f"   • Paper domain: Robotics/Control Systems")
    print(f"   • Key focus: Dynamic quadruped locomotion")

if __name__ == "__main__":
    test_robotics_paper()