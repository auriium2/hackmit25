#!/usr/bin/env python3
"""
Model Management Script for HackMIT 2025 Backend

This script helps manage the large model files in the project. It provides utilities to:
1. Download required model files from external sources
2. Create a models.zip archive for easy distribution
3. Extract models from a zip archive
4. Clean up model files for git commits

Usage:
    # Download all required model files
    python model_management.py download

    # Create a models.zip archive
    python model_management.py create_archive

    # Extract models from models.zip
    python model_management.py extract_archive

    # Clean up model files for git
    python model_management.py clean
"""

import os
import sys
import zipfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import json
import argparse
import logging
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_management")

# Constants
BACKEND_DIR = Path(__file__).parent.parent.absolute()
MODELS_DIR = BACKEND_DIR / "models"
MODELS_ZIP = BACKEND_DIR / "models.zip"

# Model configuration - add model download URLs here
MODEL_CONFIG = {
    "EasyOcr": {
        "craft_mlt_25k.pth": {
            "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/craft_mlt_25k.pth",
            "size_mb": 79.30
        },
        "english_g2.pth": {
            "url": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.pth",
            "size_mb": 46.0
        },
        "latin_g2.pth": {
            "url": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/latin_g2.zip",
            "size_mb": 85.0,
            "extract": True
        }
    },
    # Add more model configurations as needed
    # The real download links should be added for other models
    "ds4sd--docling-layout-heron": {
        "model.safetensors": {
            "url": None,  # Add real URL
            "size_mb": 163.71
        }
    },
    "ds4sd--docling-models/model_artifacts/tableformer/accurate": {
        "tableformer_accurate.safetensors": {
            "url": None,  # Add real URL
            "size_mb": 202.90
        }
    },
    "ds4sd--docling-models/model_artifacts/tableformer/fast": {
        "tableformer_fast.safetensors": {
            "url": None,  # Add real URL
            "size_mb": 138.72
        }
    },
    "ds4sd--CodeFormulaV2": {
        "model.safetensors": {
            "url": None,  # Add real URL
            "size_mb": 601.76
        }
    }
}


def download_file(url: str, destination: Path, desc: str = None) -> None:
    """
    Download a file from a URL with progress bar
    """
    if not url:
        logger.warning(f"No URL provided for {destination}")
        return
    
    # Create parent directories if they don't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Don't redownload if file exists and is of correct size
    if destination.exists():
        logger.info(f"File already exists: {destination}")
        return
    
    logger.info(f"Downloading {url} to {destination}")
    
    # Stream download with progress bar
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        desc = desc or os.path.basename(destination)
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as progress_bar:
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))


def download_models() -> None:
    """
    Download all model files from their sources
    """
    logger.info("Starting model downloads...")
    
    for model_dir, files in MODEL_CONFIG.items():
        for filename, config in files.items():
            if not config.get("url"):
                logger.warning(f"No download URL for {model_dir}/{filename}, skipping")
                continue
                
            dest_path = MODELS_DIR / model_dir / filename
            download_file(config["url"], dest_path, desc=f"{model_dir}/{filename}")
            
            # Handle zip files that need extraction
            if config.get("extract") and dest_path.suffix == ".zip":
                logger.info(f"Extracting {dest_path}")
                extract_dir = dest_path.parent
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                # Remove the zip file after extraction
                dest_path.unlink()
    
    logger.info("Model downloads completed.")


def create_models_archive() -> None:
    """
    Create a zip archive of all model files
    """
    logger.info(f"Creating models archive at {MODELS_ZIP}")
    
    if not MODELS_DIR.exists():
        logger.error(f"Models directory {MODELS_DIR} does not exist")
        return
    
    # Create a new zip file
    with zipfile.ZipFile(MODELS_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the models directory and add all files
        for root, _, files in os.walk(MODELS_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                # Get the relative path from the models directory
                rel_path = os.path.relpath(file_path, BACKEND_DIR)
                logger.debug(f"Adding {rel_path} to archive")
                zipf.write(file_path, rel_path)
    
    # Get the size of the archive
    size_mb = MODELS_ZIP.stat().st_size / (1024 * 1024)
    logger.info(f"Created models archive: {MODELS_ZIP} ({size_mb:.2f} MB)")


def extract_models_archive() -> None:
    """
    Extract the models archive to restore model files
    """
    if not MODELS_ZIP.exists():
        logger.error(f"Models archive {MODELS_ZIP} does not exist")
        return
    
    logger.info(f"Extracting models from {MODELS_ZIP}")
    
    # Create the extraction directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract the archive
    with zipfile.ZipFile(MODELS_ZIP, 'r') as zipf:
        zipf.extractall(BACKEND_DIR)
    
    logger.info(f"Models extracted to {MODELS_DIR}")


def clean_model_files() -> None:
    """
    Clean up model files to prepare for git commit
    Preserves the models.zip archive and removes the extracted files
    """
    if not MODELS_DIR.exists():
        logger.info(f"No models directory found at {MODELS_DIR}")
        return
    
    logger.info(f"Cleaning up model files in {MODELS_DIR}")
    
    # Check if we have a models.zip backup first
    if not MODELS_ZIP.exists():
        logger.warning("No models.zip backup found. Run 'create_archive' first to avoid data loss.")
        confirmation = input("Continue anyway? This will delete model files without backup! (y/N): ")
        if confirmation.lower() != 'y':
            logger.info("Cleanup cancelled.")
            return
    
    # Remove the models directory
    shutil.rmtree(MODELS_DIR)
    logger.info(f"Removed {MODELS_DIR}")
    
    # Create an empty models directory to preserve structure
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a README in the models directory explaining how to restore models
    readme_path = MODELS_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write("""# Model Files

The model files have been excluded from git due to their large size.

To restore the model files, run:
```
python scripts/model_management.py extract_archive
```

This will extract the models from the models.zip archive.
""")
    
    logger.info("Model cleanup completed. Created placeholder README in models directory.")


def main() -> None:
    """
    Main function to parse arguments and execute commands
    """
    parser = argparse.ArgumentParser(description="Model management utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download model files")
    
    # Create archive command
    create_archive_parser = subparsers.add_parser("create_archive", help="Create models.zip archive")
    
    # Extract archive command
    extract_archive_parser = subparsers.add_parser("extract_archive", help="Extract models from archive")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up model files")
    
    args = parser.parse_args()
    
    if args.command == "download":
        download_models()
    elif args.command == "create_archive":
        create_models_archive()
    elif args.command == "extract_archive":
        extract_models_archive()
    elif args.command == "clean":
        clean_model_files()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()