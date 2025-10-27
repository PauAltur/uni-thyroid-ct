# Thyroid CT Preprocessing and Embedding Pipeline

A complete pipeline for preprocessing 3D CT volumes and extracting embeddings using the UNI foundation model.

## 🎯 Overview

This project processes thyroid CT scans to extract patch-level embeddings for downstream analysis. The pipeline handles:

- Loading 3D TIFF volumes with tissue and invasion masks
- Normalization and resampling to common resolution
- 2D patch extraction (224×224 for UNI model)
- Tissue-based filtering
- GPU-accelerated processing (optional)
- Embedding extraction using UNI foundation model
- Structured HDF5 output with comprehensive metadata

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate setup
python tests/test_setup.py

# 3. Update config file
# Edit config/preprocess_config.yaml with your paths

# 4. Run pipeline
python scripts/preprocess_pipeline.py

# 5. Open analysis notebook
# notebooks/03-analyze-embeddings.ipynb
```

## 📁 Project Structure

```
├── config/
│   ├── paths.yaml                    # Path configuration
│   └── preprocess_config.yaml        # Pipeline configuration
├── config/
│   ├── paths.yaml                    # Path configuration
│   └── preprocess_config.yaml        # Pipeline configuration
├── notebooks/
│   ├── 01-UNI-api-callback.ipynb    # UNI model setup
│   ├── 02-preprocessing-visualization.ipynb
│   └── 03-analyze-embeddings.ipynb   # Example analysis
├── scripts/
│   └── preprocess_pipeline.py        # Main pipeline script
├── src/
│   ├── preprocessing.py              # Core preprocessing functions
│   └── utils.py
├── tests/
│   ├── test_preprocessing.py         # Unit tests
│   └── test_setup.py                 # Setup validation
├── IMPLEMENTATION_SUMMARY.md         # 👈 START HERE
├── SETUP_GUIDE.md                    # Detailed setup guide
├── PIPELINE_README.md                # Pipeline documentation
└── requirements.txt
```

## 🎓 Getting Started

**New to this project?** Start with [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) for a complete overview.

Then:
1. Read [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for detailed setup instructions
2. Review [`PIPELINE_README.md`](PIPELINE_README.md) for pipeline details
3. Run `python tests/test_setup.py` to validate your setup

## 🔑 Key Features

### ✅ Configurable Pipeline
- Hydra-based configuration
- Override any parameter from command line
- Multiple configuration profiles

### ✅ GPU Acceleration
- CuPy for 5-20x faster resampling
- Batched model inference
- Automatic CPU fallback

### ✅ Robust Processing
- Validates all inputs
- Graceful error handling
- Detailed logging and progress tracking
- Continues if one sample fails

### ✅ Comprehensive Metadata
Each patch includes:
- Sample of origin
- 3D position and normalized coordinates
- Tissue percentage
- Invasion presence and type (0-4)
- Invasion percentage

### ✅ Cache Management
- HuggingFace cache: configurable
- CuPy cache: `/t/users/altp/.cupy`
- Prevents C: drive filling

## 📊 Input/Output

### Input Format
```
data_dir/
├── <sample_code>.tif           # CT volume
├── <sample_code>_tissue.tif    # Tissue mask
└── <sample_code>_mask.tif      # Invasion mask (integer: 0-4)
```

Plus a CSV file with columns: `filename`, `x_resolution`, `y_resolution`, `z_resolution`

### Output Format
```
output.h5
├── embeddings (N, 1024)              # UNI embeddings
├── sample_codes (N,)                  # Sample identifiers
└── metadata/
    ├── patch_index, position_d/h/w
    ├── coord_d/h/w_normalized [0,1]
    ├── tissue_percentage
    ├── has_invasion (bool)
    ├── invasion_type (0-4)
    └── invasion_percentage
```

## 🛠️ Configuration

Edit `config/preprocess_config.yaml`:

```yaml
paths:
  data_dir: "path/to/data"
  resolution_csv: "path/to/resolutions.csv"
  output_dir: "path/to/output"
  cupy_cache: "/t/users/altp/.cupy"

preprocessing:
  resampling:
    target_spacing: [1.0, 1.0, 1.0]  # mm
    use_gpu: true
  patching:
    patch_size: [1, 224, 224]
    stride: [1, 224, 224]  # non-overlapping
  tissue_filter:
    min_tissue_percentage: 0.0  # keep all

model:
  name: "hf-hub:MahmoodLab/UNI2-h"
  batch_size: 32
  device: "cuda"
```

Override from command line:
```bash
python scripts/preprocess_pipeline.py model.batch_size=64 preprocessing.resampling.use_gpu=false
```

## 📈 Example Usage

### Basic Pipeline Run
```bash
python scripts/preprocess_pipeline.py
```

### With Custom Settings
```bash
python scripts/preprocess_pipeline.py \
    model.batch_size=32 \
    preprocessing.tissue_filter.min_tissue_percentage=10.0 \
    preprocessing.patching.stride=[1,112,112]  # 50% overlap
```

### Inspect Results
```bash
# Summary
Open notebooks/03-analyze-embeddings.ipynb

# Filter by sample
Open notebooks/03-analyze-embeddings.ipynb
```

### Load in Python
```python
import h5py
import pandas as pd

with h5py.File("output.h5", "r") as f:
    embeddings = f["embeddings"][:]
    metadata = pd.DataFrame({
        "sample": f["sample_codes"][:].astype(str),
        "tissue_pct": f["metadata/tissue_percentage"][:],
        "invasion_type": f["metadata/invasion_type"][:],
    })
```

## 🧪 Testing

```bash
# Unit tests
pytest tests/

# Pipeline validation
python tests/test_setup.py

# Check specific module
pytest tests/test_preprocessing.py -v
```

## 📚 Documentation

- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Complete overview and quick start
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: Detailed setup instructions
- **[PIPELINE_README.md](PIPELINE_README.md)**: Pipeline documentation and troubleshooting
- **Notebooks**: Interactive examples and visualizations

## 🐛 Troubleshooting

### Quick Check
```bash
python tests/test_setup.py
```

### Common Issues

**CuPy cache errors**: Check `/t/users/altp/.cupy` is writable, or set `preprocessing.resampling.use_gpu=false`

**GPU out of memory**: Reduce `model.batch_size` (try 8 or 16)

**Resolution not found**: Verify CSV `filename` column matches `<sample_code>.tif`

See [PIPELINE_README.md](PIPELINE_README.md) for more troubleshooting tips.

## 📦 Dependencies

- Python 3.8+
- PyTorch (with CUDA for GPU)
- CuPy (optional, for GPU acceleration)
- timm (UNI model)
- HuggingFace Hub
- hydra-core (configuration)
- h5py (output format)
- tifffile (input format)

Full list in `requirements.txt`

## 🏗️ Architecture

```
Input Data → Load & Validate → Normalize → Resample → Extract Patches
                                                            ↓
HDF5 Output ← Compute Embeddings ← RGB Conversion ← Filter by Tissue
```

## 🤝 Contributing

1. Run tests: `pytest tests/`
2. Check code style: `pre-commit run --all-files`
3. Update documentation as needed

## 📝 License

[Add your license here]

## 👥 Authors

[Add authors here]

## 🙏 Acknowledgments

- UNI foundation model by MahmoodLab
- timm library for model implementations

---

**Ready to start?** See [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) for a complete walkthrough!
