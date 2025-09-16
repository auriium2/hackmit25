# Model Files for HackMIT 2025 Backend

This directory contains large machine learning model files used by the application. These files are **not** tracked in Git due to their size exceeding GitHub's file size limits.

## Large Model Files

The following large model files are required for the application to function correctly:

- `EasyOcr/craft_mlt_25k.pth` (79.30 MB)
- `ds4sd--docling-layout-heron/model.safetensors` (163.71 MB)
- `ds4sd--docling-models/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors` (202.90 MB)
- `ds4sd--docling-models/model_artifacts/tableformer/fast/tableformer_fast.safetensors` (138.72 MB)
- `ds4sd--CodeFormulaV2/model.safetensors` (601.76 MB)

## Managing Model Files

We've created a model management script to help work with these large files. The script is located at `../scripts/model_management.py`.

### First-time Setup

To download model files from their original sources (when URLs are configured):

```bash
python scripts/model_management.py download
```

### Working with models.zip

We use a zip archive to store and distribute these model files outside of Git.

#### Creating the archive

Before pushing to GitHub, archive your model files:

```bash
python scripts/model_management.py create_archive
```

This creates `models.zip` in the backend directory, which contains all model files.

#### Extracting the archive

After cloning the repository or pulling changes, extract the models:

```bash
python scripts/model_management.py extract_archive
```

#### Cleaning up before git commits

To prepare for a git commit and remove large files:

```bash
python scripts/model_management.py clean
```

## Alternative Distribution Methods

For team collaboration, consider these alternatives for distributing model files:

1. **Cloud Storage**: Upload `models.zip` to Google Drive, Dropbox, or S3
2. **Direct Transfer**: Use tools like `scp` or `rsync` to transfer directly
3. **Hugging Face Hub**: Host models on Hugging Face and modify code to download them at runtime

## Adding New Models

When adding new models:

1. Add the model details to `MODEL_CONFIG` in `scripts/model_management.py`
2. Update this README with the new file sizes
3. Update `.gitignore` if necessary