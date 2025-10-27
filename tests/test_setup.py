"""
Test script to validate pipeline setup before full run.

This script checks:
1. All dependencies are installed
2. Data files exist and are accessible
3. Resolution CSV is properly formatted
4. GPU/CuPy is available (if configured)
5. Model can be loaded
6. Small test run on one sample

Usage:
    pytest tests/test_setup.py
    python tests/test_setup.py
    python tests/test_setup.py --config-name=preprocess_config
"""

import os
import sys
from pathlib import Path
import logging

import pytest
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def cfg():
    """Pytest fixture to load Hydra configuration."""
    config_dir = Path(__file__).parent.parent / "config"
    config_dir = config_dir.resolve()
    
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="preprocess_config")
        return cfg


def test_imports():
    """Test that all required packages are installed."""
    logger.info("Testing imports...")
    
    required_packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("h5py", "h5py"),
        ("tifffile", "tifffile"),
        ("torch", "PyTorch"),
        ("timm", "timm"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("PIL", "Pillow"),
    ]
    
    optional_packages = [
        ("cupy", "CuPy (GPU acceleration)"),
    ]
    
    all_ok = True
    
    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {package_name}")
        except ImportError as e:
            logger.error(f"  ✗ {package_name} - MISSING: {e}")
            all_ok = False
    
    for module_name, package_name in optional_packages:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {package_name}")
        except ImportError:
            logger.warning(f"  ⚠ {package_name} - not installed (optional)")
    
    if not all_ok:
        logger.error("\n❌ Some required packages are missing. Install with:")
        logger.error("   pip install -r requirements.txt")
        pytest.fail("Some required packages are missing")
    
    logger.info("✅ All required packages installed\n")
    assert all_ok


def test_data_files(cfg: DictConfig):
    """Test that data directory and files exist."""
    logger.info("Testing data files...")
    
    data_dir = Path(cfg.paths.data_dir)
    
    if not data_dir.exists():
        logger.error(f"  ✗ Data directory not found: {data_dir}")
        pytest.fail("Test failed")
    
    logger.info(f"  ✓ Data directory exists: {data_dir}")
    
    # Check for volume files
    volume_files = list(data_dir.glob("*.tif"))
    volume_files = [f for f in volume_files if "_tissue" not in f.name and "_mask" not in f.name]
    
    if len(volume_files) == 0:
        logger.error("  ✗ No volume files found (.tif files without '_tissue' or '_mask')")
        pytest.fail("Test failed")
    
    logger.info(f"  ✓ Found {len(volume_files)} volume files")
    
    # Check first sample for all three files
    sample_file = volume_files[0]
    sample_code = sample_file.stem
    
    volume_path = data_dir / cfg.data.volume_pattern.replace("{sample_code}", sample_code)
    tissue_path = data_dir / cfg.data.tissue_mask_pattern.replace("{sample_code}", sample_code)
    invasion_path = data_dir / cfg.data.invasion_mask_pattern.replace("{sample_code}", sample_code)
    
    logger.info(f"\n  Checking sample: {sample_code}")
    
    files_ok = True
    
    if volume_path.exists():
        logger.info(f"    ✓ Volume: {volume_path.name}")
    else:
        logger.error(f"    ✗ Volume not found: {volume_path}")
        files_ok = False
    
    if tissue_path.exists():
        logger.info(f"    ✓ Tissue mask: {tissue_path.name}")
    else:
        logger.error(f"    ✗ Tissue mask not found: {tissue_path}")
        files_ok = False
    
    if invasion_path.exists():
        logger.info(f"    ✓ Invasion mask: {invasion_path.name}")
    else:
        logger.warning(f"    ⚠ Invasion mask not found (will assume no invasion): {invasion_path.name}")
        # Not a failure - samples without invasion masks are allowed
    
    if not files_ok:
        logger.error("\n❌ File naming mismatch. Check patterns in config:")
        logger.error(f"   volume_pattern: {cfg.data.volume_pattern}")
        logger.error(f"   tissue_mask_pattern: {cfg.data.tissue_mask_pattern}")
        logger.error(f"   invasion_mask_pattern: {cfg.data.invasion_mask_pattern}")
        pytest.fail("Test failed")
    
    logger.info("✅ Data files accessible\n")
    # Test passed


def test_resolution_csv(cfg: DictConfig):
    """Test resolution CSV format."""
    logger.info("Testing resolution CSV...")
    
    import pandas as pd
    
    csv_path = Path(cfg.paths.resolution_csv)
    
    if not csv_path.exists():
        logger.error(f"  ✗ Resolution CSV not found: {csv_path}")
        pytest.fail("Test failed")
    
    logger.info(f"  ✓ CSV file exists: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"  ✓ CSV loaded with {len(df)} entries")
        
        # Check required columns
        required_cols = [
            cfg.data.csv_columns.filename,
            cfg.data.csv_columns.x_resolution,
            cfg.data.csv_columns.y_resolution,
            cfg.data.csv_columns.z_resolution,
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"  ✗ Missing columns: {missing_cols}")
            logger.error(f"    Available columns: {list(df.columns)}")
            pytest.fail("Test failed")
        
        logger.info(f"  ✓ All required columns present: {required_cols}")
        
        # Show sample
        logger.info("\n  Sample entries:")
        logger.info(df.head(3).to_string(index=False))
        
    except Exception as e:
        logger.error(f"  ✗ Error reading CSV: {e}")
        pytest.fail("Test failed")
    
    logger.info("\n✅ Resolution CSV properly formatted\n")
    # Test passed


def test_gpu_cupy(cfg: DictConfig):
    """Test GPU and CuPy availability."""
    logger.info("Testing GPU/CuPy...")
    
    import torch
    
    # Test PyTorch CUDA
    if torch.cuda.is_available():
        logger.info(f"  ✓ PyTorch CUDA available")
        logger.info(f"    Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("  ⚠ PyTorch CUDA not available - will use CPU")
    
    # Test CuPy
    if cfg.preprocessing.resampling.use_gpu:
        try:
            import cupy as cp
            logger.info(f"  ✓ CuPy available")
            
            # Test cache directory
            cache_dir = cfg.paths.get("cupy_cache")
            if cache_dir:
                cache_path = Path(cache_dir)
                cache_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"  ✓ CuPy cache directory: {cache_dir}")
            
            # Small test
            x = cp.array([1, 2, 3])
            y = x * 2
            logger.info(f"  ✓ CuPy test passed: {cp.asnumpy(y)}")
            
        except ImportError:
            logger.error("  ✗ CuPy not installed but use_gpu=true")
            logger.error("    Install with: pip install cupy-cuda11x")
            logger.error("    Or set: preprocessing.resampling.use_gpu=false")
            pytest.fail("Test failed")
        except Exception as e:
            logger.error(f"  ✗ CuPy error: {e}")
            pytest.fail("Test failed")
    else:
        logger.info("  ℹ GPU resampling disabled (use_gpu=false)")
    
    logger.info("✅ GPU/CuPy check complete\n")
    # Test passed


def test_model_loading(cfg: DictConfig):
    """Test UNI model loading."""
    logger.info("Testing UNI model loading...")
    
    import torch
    import timm
    from timm.data import resolve_data_config, create_transform
    
    # Set cache directories
    if cfg.paths.get("hf_cache"):
        os.environ["HF_HOME"] = cfg.paths.hf_cache
    
    try:
        # Try to login if token provided
        if cfg.model.hf_token_file and Path(cfg.model.hf_token_file).exists():
            from huggingface_hub import login
            with open(cfg.model.hf_token_file, "r") as f:
                token = f.read().strip()
            login(token=token)
            logger.info("  ✓ Logged in to HuggingFace")
        
        logger.info(f"  Loading model: {cfg.model.name}...")
        
        device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
        
        # UNI2-h specific kwargs
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        
        model = timm.create_model(
            cfg.model.name,
            pretrained=cfg.model.pretrained,
            **timm_kwargs
        )
        model = model.to(device)
        model.eval()
        
        logger.info(f"  ✓ Model loaded on {device}")
        logger.info(f"  ✓ Embedding dimension: {model.num_features}")
        
        # Test preprocessing transform
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        logger.info("  ✓ Transform created")
        
        # Small forward pass test
        logger.info("  Testing forward pass...")
        import numpy as np
        from PIL import Image
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_img)
        dummy_tensor = transform(dummy_pil).unsqueeze(0).to(device)
        
        logger.info(f"  Input tensor shape: {dummy_tensor.shape}")
        
        with torch.no_grad():
            output = model(dummy_tensor)
        
        logger.info(f"  ✓ Forward pass successful, output shape: {output.shape}")
        logger.info("✅ Model loading test complete\n")
        
    except Exception as e:
        logger.error(f"  ✗ Model loading failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("\nPossible issues:")
        logger.error("  - HuggingFace token required (set model.hf_token_file)")
        logger.error("  - Network connection issue")
        logger.error("  - Model name incorrect")
        logger.error("  - Model architecture mismatch")
        pytest.fail("Test failed")
    
    logger.info("✅ Model loading successful\n")
    # Test passed


def test_small_run(cfg: DictConfig):
    """Run a small test on one sample."""
    logger.info("Running small test on one sample...")
    
    try:
        import numpy as np
        import tifffile
        from preprocessing import (
            normalize_volume_zscore,
            resample_volume,
            extract_patches,
            grayscale_to_rgb,
        )
        
        # Find first sample
        data_dir = Path(cfg.paths.data_dir)
        volume_files = [f for f in data_dir.glob("*.tif") if "_tissue" not in f.name and "_mask" not in f.name]
        
        if len(volume_files) == 0:
            logger.error("  ✗ No samples found")
            pytest.fail("Test failed")
        
        sample_code = volume_files[0].stem
        logger.info(f"  Test sample: {sample_code}")
        
        # Load volume
        volume_path = data_dir / cfg.data.volume_pattern.replace("{sample_code}", sample_code)
        volume = tifffile.imread(str(volume_path)).astype(np.float32)
        logger.info(f"  ✓ Loaded volume: {volume.shape}")
        
        # Normalize
        volume = normalize_volume_zscore(volume, dtype=np.float32)
        logger.info(f"  ✓ Normalized (mean={volume.mean():.2f}, std={volume.std():.2f})")
        
        # Resample (small target for test)
        original_spacing = (1.0, 1.0, 1.0)  # Dummy spacing for test
        target_spacing = (2.0, 2.0, 2.0)  # Downsample for speed
        
        volume_small = resample_volume(
            volume[:50, :256, :256],  # Small crop for test
            original_spacing=original_spacing,
            target_spacing=target_spacing,
            order=1,
        )
        logger.info(f"  ✓ Resampled: {volume.shape[:3]} -> {volume_small.shape}")
        
        # Extract patches
        patches, positions = extract_patches(
            volume_small,
            patch_size=tuple(cfg.preprocessing.patching.patch_size),
            stride=tuple(cfg.preprocessing.patching.stride),
        )
        logger.info(f"  ✓ Extracted {patches.shape[0]} patches")
        
        # Convert to RGB
        patches_2d = patches[:, 0, :, :]
        rgb_patches = grayscale_to_rgb(patches_2d)
        logger.info(f"  ✓ Converted to RGB: {rgb_patches.shape}")
        
        logger.info("✅ Small test run successful\n")
        # Test passed
        
    except Exception as e:
        logger.error(f"  ✗ Test run failed: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail("Test failed")


@hydra.main(version_base=None, config_path="../config", config_name="preprocess_config")
def main(cfg: DictConfig):
    """Run all validation tests."""
    
    logger.info("=" * 80)
    logger.info("Pipeline Setup Validation")
    logger.info("=" * 80)
    logger.info("")
    
    # Show configuration
    logger.info("Configuration:")
    logger.info("-" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("-" * 80)
    logger.info("")
    
    # Run tests
    tests = [
        ("Dependencies", lambda: test_imports()),
        ("Data Files", lambda: test_data_files(cfg)),
        ("Resolution CSV", lambda: test_resolution_csv(cfg)),
        ("GPU/CuPy", lambda: test_gpu_cupy(cfg)),
        ("Model Loading", lambda: test_model_loading(cfg)),
        ("Small Test Run", lambda: test_small_run(cfg)),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Test: {test_name}")
        logger.info(f"{'=' * 80}\n")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"  {status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ All tests passed! Ready to run full pipeline.")
        logger.info("\nRun with:")
        logger.info("  python src/preprocess_pipeline.py")
    else:
        logger.error("❌ Some tests failed. Fix issues before running pipeline.")
    logger.info("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
