"""
Preprocessing and embedding extraction pipeline for CT volumes.

This script processes 3D CT volumes, extracts 2D patches, and computes embeddings
using the UNI foundation model. Results are saved in HDF5 format with metadata.

Usage:
    python src/preprocess_pipeline.py
    python src/preprocess_pipeline.py preprocessing.resampling.use_gpu=false
    python src/preprocess_pipeline.py paths.output_dir=/custom/path
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import h5py
import tifffile
import torch
import timm
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    normalize_volume_zscore,
    normalize_volume_zscore_inplace,
    resample_volume,
    resample_volume_gpu,
    extract_patches,
    grayscale_to_rgb,
)


# Configure logging
logger = logging.getLogger(__name__)


def setup_environment(cfg: DictConfig) -> None:
    """
    Set up environment variables for cache directories and other settings.
    
    Args:
        cfg: Hydra configuration object
    """
    # Set HuggingFace cache directory
    if cfg.paths.get("hf_cache"):
        os.environ["HF_HOME"] = cfg.paths.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = str(Path(cfg.paths.hf_cache) / "transformers")
        logger.info(f"Set HF_HOME to: {cfg.paths.hf_cache}")
    
    # Set CuPy cache directory
    if cfg.paths.get("cupy_cache"):
        os.environ["CUPY_CACHE_DIR"] = cfg.paths.cupy_cache
        # Also set the kernel cache directory
        os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "0"  # Don't save CUDA source
        logger.info(f"Set CUPY_CACHE_DIR to: {cfg.paths.cupy_cache}")
        
        # Create cache directory if it doesn't exist
        Path(cfg.paths.cupy_cache).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    if cfg.runtime.deterministic:
        np.random.seed(cfg.runtime.seed)
        torch.manual_seed(cfg.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.runtime.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set random seed to {cfg.runtime.seed} for reproducibility")


def load_resolution_csv(csv_path: str, cfg: DictConfig) -> pd.DataFrame:
    """
    Load the resolution CSV file.
    
    Args:
        csv_path: Path to the resolution CSV file
        cfg: Configuration containing column names
        
    Returns:
        DataFrame with resolution information
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded resolution CSV with {len(df)} entries")
    
    # Validate required columns
    required_cols = [
        cfg.data.csv_columns.filename,
        cfg.data.csv_columns.x_resolution,
        cfg.data.csv_columns.y_resolution,
        cfg.data.csv_columns.z_resolution,
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    return df


def get_sample_resolution(
    sample_code: str, resolution_df: pd.DataFrame, cfg: DictConfig
) -> Tuple[float, float, float]:
    """
    Get the resolution for a specific sample from the CSV.
    
    Args:
        sample_code: Sample identifier (e.g., "002_B05.20964B")
        resolution_df: DataFrame with resolution information
        cfg: Configuration containing column names
        
    Returns:
        Tuple of (depth_spacing, height_spacing, width_spacing) in mm
    """
    # Construct the filename pattern
    filename_col = cfg.data.csv_columns.filename
    volume_pattern = cfg.data.volume_pattern.replace("{sample_code}", sample_code)
    
    # Find the matching row
    match = resolution_df[resolution_df[filename_col] == volume_pattern]
    
    if len(match) == 0:
        # Try without .tif extension
        sample_name = sample_code.replace(".tif", "")
        match = resolution_df[resolution_df[filename_col].str.contains(sample_name)]
    
    if len(match) == 0:
        raise ValueError(f"Resolution not found for sample: {sample_code}")
    
    if len(match) > 1:
        logger.warning(f"Multiple resolution entries found for {sample_code}, using first")
    
    row = match.iloc[0]
    
    # Extract spacing values (CSV has x, y, z which map to width, height, depth)
    z_spacing = float(row[cfg.data.csv_columns.z_resolution])  # Depth
    y_spacing = float(row[cfg.data.csv_columns.y_resolution])  # Height
    x_spacing = float(row[cfg.data.csv_columns.x_resolution])  # Width
    
    return (z_spacing, y_spacing, x_spacing)


def load_sample_data(
    sample_code: str, data_dir: Path, cfg: DictConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load volume, tissue mask, and invasion mask for a sample.
    
    Args:
        sample_code: Sample identifier
        data_dir: Directory containing the data
        cfg: Configuration containing file patterns
        
    Returns:
        Tuple of (volume, tissue_mask, invasion_mask)
    """
    # Construct file paths
    volume_path = data_dir / cfg.data.volume_pattern.replace("{sample_code}", sample_code)
    tissue_path = data_dir / cfg.data.tissue_mask_pattern.replace("{sample_code}", sample_code)
    invasion_path = data_dir / cfg.data.invasion_mask_pattern.replace("{sample_code}", sample_code)
    
    # Load data
    logger.info(f"Loading {sample_code}...")
    
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume not found: {volume_path}")
    volume = tifffile.imread(str(volume_path)).astype(np.float32)
    
    if not tissue_path.exists():
        raise FileNotFoundError(f"Tissue mask not found: {tissue_path}")
    tissue_mask = tifffile.imread(str(tissue_path)).astype(np.uint8)
    
    # Handle missing invasion mask (no invasion = all zeros)
    if not invasion_path.exists():
        logger.info(f"  No invasion mask found - assuming no invasion (all zeros)")
        invasion_mask = np.zeros_like(volume, dtype=np.uint8)
    else:
        invasion_mask = tifffile.imread(str(invasion_path)).astype(np.uint8)
    
    logger.info(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
    logger.info(f"  Tissue mask shape: {tissue_mask.shape}, dtype: {tissue_mask.dtype}")
    logger.info(f"  Invasion mask shape: {invasion_mask.shape}, dtype: {invasion_mask.dtype}")
    
    # Validate shapes match
    if volume.shape != tissue_mask.shape or volume.shape != invasion_mask.shape:
        raise ValueError(
            f"Shape mismatch for {sample_code}: "
            f"volume={volume.shape}, tissue={tissue_mask.shape}, invasion={invasion_mask.shape}"
        )
    
    return volume, tissue_mask, invasion_mask


def preprocess_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    cfg: DictConfig,
) -> np.ndarray:
    """
    Normalize and resample a volume.
    
    Args:
        volume: Input 3D volume
        original_spacing: Original voxel spacing (depth, height, width)
        cfg: Configuration for preprocessing
        
    Returns:
        Preprocessed volume
    """
    # Normalize
    norm_cfg = cfg.preprocessing.normalization
    logger.info(f"  Normalizing with {norm_cfg.method}...")
    
    if norm_cfg.method == "zscore":
        volume = normalize_volume_zscore(
            volume, epsilon=norm_cfg.epsilon, dtype=getattr(np, norm_cfg.dtype)
        )
    elif norm_cfg.method == "zscore_inplace":
        volume = normalize_volume_zscore_inplace(volume, epsilon=norm_cfg.epsilon)
    else:
        raise ValueError(f"Unknown normalization method: {norm_cfg.method}")
    
    # Resample
    resamp_cfg = cfg.preprocessing.resampling
    target_spacing = tuple(resamp_cfg.target_spacing)
    
    logger.info(f"  Resampling from {original_spacing} to {target_spacing} mm...")
    
    # Check volume size to avoid GPU timeout on large volumes
    volume_size_mb = volume.nbytes / (1024 * 1024)
    max_gpu_volume_mb = 1000  # Max 1000MB for GPU to avoid Windows TDR timeout
    
    use_gpu = resamp_cfg.use_gpu
    if use_gpu and volume_size_mb > max_gpu_volume_mb:
        logger.warning(
            f"  Volume too large for GPU resampling ({volume_size_mb:.1f} MB > {max_gpu_volume_mb} MB), "
            f"using CPU to avoid timeout"
        )
        use_gpu = False
    
    if use_gpu:
        try:
            volume = resample_volume_gpu(
                volume,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                order=resamp_cfg.interpolation_order,
                mode=resamp_cfg.mode,
                cval=resamp_cfg.cval,
                preserve_range=resamp_cfg.preserve_range,
            )
            logger.info(" Used GPU resampling")
        except (ImportError, Exception) as e:
            logger.warning(f"  GPU resampling failed ({e}), falling back to CPU")
            volume = resample_volume(
                volume,
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                order=resamp_cfg.interpolation_order,
                mode=resamp_cfg.mode,
                cval=resamp_cfg.cval,
                preserve_range=resamp_cfg.preserve_range,
            )
    else:
        volume = resample_volume(
            volume,
            original_spacing=original_spacing,
            target_spacing=target_spacing,
            order=resamp_cfg.interpolation_order,
            mode=resamp_cfg.mode,
            cval=resamp_cfg.cval,
            preserve_range=resamp_cfg.preserve_range,
        )
    
    logger.info(f"  Resampled shape: {volume.shape}")
    
    return volume


def resample_mask(
    mask: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    use_gpu: bool = False,
) -> np.ndarray:
    """
    Resample a binary/integer mask using nearest-neighbor interpolation.
    
    Args:
        mask: Input mask
        original_spacing: Original voxel spacing
        target_spacing: Target voxel spacing
        use_gpu: Whether to use GPU resampling
        
    Returns:
        Resampled mask
    """
    if use_gpu:
        try:
            return resample_volume_gpu(
                mask.astype(np.float32),
                original_spacing=original_spacing,
                target_spacing=target_spacing,
                order=0,  # Nearest-neighbor for masks
                mode="constant",
                cval=0.0,
                preserve_range=True,
            ).astype(mask.dtype)
        except ImportError:
            pass
    
    return resample_volume(
        mask.astype(np.float32),
        original_spacing=original_spacing,
        target_spacing=target_spacing,
        order=0,  # Nearest-neighbor for masks
        mode="constant",
        cval=0.0,
        preserve_range=True,
    ).astype(mask.dtype)


def extract_and_filter_patches(
    volume: np.ndarray,
    tissue_mask: np.ndarray,
    invasion_mask: np.ndarray,
    cfg: DictConfig,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Extract patches and compute metadata, filtering by tissue content.
    
    Args:
        volume: Preprocessed 3D volume
        tissue_mask: Binary tissue mask
        invasion_mask: Integer invasion mask (0=no invasion, 1-4=invasion types)
        cfg: Configuration for patching
        
    Returns:
        Tuple of (filtered_patches, filtered_rgb_patches, metadata_list)
    """
    patch_cfg = cfg.preprocessing.patching
    
    # Extract patches from all three volumes
    logger.info("  Extracting patches...")
    patches, positions = extract_patches(
        volume,
        patch_size=tuple(patch_cfg.patch_size),
        stride=tuple(patch_cfg.stride),
        padding_mode=patch_cfg.padding_mode,
        padding_value=patch_cfg.padding_value,
        drop_incomplete=patch_cfg.drop_incomplete,
    )
    
    tissue_patches, _ = extract_patches(
        tissue_mask.astype(np.float32),
        patch_size=tuple(patch_cfg.patch_size),
        stride=tuple(patch_cfg.stride),
        padding_mode=patch_cfg.padding_mode,
        padding_value=0.0,
        drop_incomplete=patch_cfg.drop_incomplete,
    )
    
    invasion_patches, _ = extract_patches(
        invasion_mask.astype(np.float32),
        patch_size=tuple(patch_cfg.patch_size),
        stride=tuple(patch_cfg.stride),
        padding_mode=patch_cfg.padding_mode,
        padding_value=0.0,
        drop_incomplete=patch_cfg.drop_incomplete,
    )
    
    logger.info(f"  Extracted {patches.shape[0]} patches")
    
    # Filter patches and compute metadata
    min_tissue_pct = cfg.preprocessing.tissue_filter.min_tissue_percentage
    
    filtered_patches = []
    metadata_list = []
    
    vol_d, vol_h, vol_w = volume.shape
    
    for i in range(patches.shape[0]):
        # Calculate tissue percentage
        tissue_patch = tissue_patches[i, 0]  # Remove depth dimension
        tissue_pct = (tissue_patch > 0).sum() / tissue_patch.size * 100.0
        
        # Filter by tissue content
        if tissue_pct < min_tissue_pct:
            continue
        
        # Calculate invasion metadata
        invasion_patch = invasion_patches[i, 0].astype(np.uint8)
        has_invasion = bool(invasion_patch.max() > 0)
        invasion_type = int(invasion_patch.max())  # 0 or 1-4
        invasion_pct = (invasion_patch > 0).sum() / invasion_patch.size * 100.0
        
        # Calculate normalized coordinates
        pos_d, pos_h, pos_w = positions[i]
        coord_d_norm = pos_d / vol_d if vol_d > 1 else 0.0
        coord_h_norm = pos_h / vol_h if vol_h > 1 else 0.0
        coord_w_norm = pos_w / vol_w if vol_w > 1 else 0.0
        
        # Store patch and metadata
        filtered_patches.append(patches[i, 0])  # Remove depth dimension (1, H, W) -> (H, W)
        
        metadata_list.append({
            "patch_index": len(filtered_patches) - 1,
            "position_d": int(pos_d),
            "position_h": int(pos_h),
            "position_w": int(pos_w),
            "coord_d_normalized": float(coord_d_norm),
            "coord_h_normalized": float(coord_h_norm),
            "coord_w_normalized": float(coord_w_norm),
            "tissue_percentage": float(tissue_pct),
            "has_invasion": has_invasion,
            "invasion_type": invasion_type,
            "invasion_percentage": float(invasion_pct),
        })
    
    if len(filtered_patches) == 0:
        logger.warning("  No patches passed tissue filter!")
        return np.array([]), np.array([]), []
    
    # Stack filtered patches
    filtered_patches = np.stack(filtered_patches, axis=0)
    logger.info(f"  Kept {len(filtered_patches)} patches after tissue filtering")
    
    # Convert to RGB
    logger.info("  Converting to RGB...")
    rgb_patches = grayscale_to_rgb(filtered_patches)
    
    return filtered_patches, rgb_patches, metadata_list


def load_uni_model(cfg: DictConfig) -> Tuple[torch.nn.Module, callable]:
    """
    Load the UNI foundation model and its preprocessing transform.
    
    Args:
        cfg: Configuration for model loading
        
    Returns:
        Tuple of (model, transform)
    """
    # Login to HuggingFace if token provided
    if cfg.model.hf_token_file and Path(cfg.model.hf_token_file).exists():
        from huggingface_hub import login
        with open(cfg.model.hf_token_file, "r") as f:
            token = f.read().strip()
        login(token=token)
        logger.info("Logged in to HuggingFace")
    
    # Load model
    logger.info(f"Loading UNI model: {cfg.model.name}")
    
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
    
    # Get preprocessing transform
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    logger.info(f"Model loaded on {device}")
    logger.info(f"Embedding dimension: {model.num_features}")
    
    return model, transform


def compute_embeddings(
    rgb_patches: np.ndarray,
    model: torch.nn.Module,
    transform: callable,
    cfg: DictConfig,
) -> np.ndarray:
    """
    Compute embeddings for patches using the UNI model.
    
    Args:
        rgb_patches: RGB patches with shape (N, 3, H, W)
        model: UNI model
        transform: Preprocessing transform
        cfg: Configuration for inference
        
    Returns:
        Embeddings array with shape (N, embedding_dim)
    """
    device = next(model.parameters()).device
    batch_size = cfg.model.batch_size
    num_patches = rgb_patches.shape[0]
    
    embeddings_list = []
    
    logger.info(f"  Computing embeddings for {num_patches} patches...")
    
    with torch.no_grad():
        for i in tqdm(range(0, num_patches, batch_size), desc="  Batches"):
            batch_end = min(i + batch_size, num_patches)
            batch_patches = rgb_patches[i:batch_end]
            
            # Convert to PIL images and apply transform
            # Transform expects PIL images, so we need to convert from numpy
            batch_tensors = []
            for patch in batch_patches:
                # patch is (3, H, W), convert to (H, W, 3) for PIL
                patch_hwc = np.transpose(patch, (1, 2, 0))
                
                # Normalize to [0, 1] if needed
                if patch_hwc.min() < 0 or patch_hwc.max() > 1:
                    patch_hwc = (patch_hwc - patch_hwc.min()) / (patch_hwc.max() - patch_hwc.min() + 1e-8)
                
                # Convert to uint8 for PIL
                patch_uint8 = (patch_hwc * 255).astype(np.uint8)
                
                # Convert to PIL and apply transform
                from PIL import Image
                pil_img = Image.fromarray(patch_uint8)
                tensor = transform(pil_img)
                batch_tensors.append(tensor)
            
            # Stack batch
            batch_tensor = torch.stack(batch_tensors).to(device)
            
            # Forward pass
            batch_embeddings = model(batch_tensor)
            
            # Move to CPU and convert to numpy
            embeddings_list.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all batches
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    logger.info(f"  Computed embeddings with shape: {embeddings.shape}")
    
    return embeddings


def save_to_hdf5(
    output_path: Path,
    sample_code: str,
    embeddings: np.ndarray,
    metadata_list: List[Dict],
    cfg: DictConfig,
    append: bool = True,
) -> None:
    """
    Save embeddings and metadata to HDF5 file.
    
    Args:
        output_path: Path to output HDF5 file
        sample_code: Sample identifier
        embeddings: Embeddings array (N, embedding_dim)
        metadata_list: List of metadata dictionaries
        cfg: Configuration for output settings
        append: Whether to append to existing file or create new
    """
    mode = "a" if append and output_path.exists() else "w"
    
    with h5py.File(output_path, mode) as f:
        # Create dataset for embeddings (append all samples to single dataset)
        if "embeddings" not in f:
            # Create dataset with unlimited size
            compression_args = {
                "compression": cfg.output.compression,
                "chunks": True,
                "maxshape": (None, embeddings.shape[1])
            }
            if cfg.output.compression == "gzip":
                compression_args["compression_opts"] = cfg.output.compression_level
            f.create_dataset(
                "embeddings",
                data=embeddings,
                **compression_args
            )

            # Store embedding dimension as attribute
            f["embeddings"].attrs["embedding_dim"] = embeddings.shape[1]
            f["embeddings"].attrs["model_name"] = cfg.model.name

            # Create metadata datasets
            num_patches = len(metadata_list)

            # Sample codes as variable-length strings
            dt = h5py.string_dtype(encoding='utf-8')
            sample_codes_data = np.array([sample_code] * num_patches, dtype=object)
            compression_args = {
                "compression": cfg.output.compression,
                "chunks": True,
                "maxshape": (None,),
                "dtype": dt
            }
            if cfg.output.compression == "gzip":
                compression_args["compression_opts"] = cfg.output.compression_level
            f.create_dataset(
                "sample_codes",
                data=sample_codes_data,
                **compression_args
            )

            # Create datasets for each metadata field
            for field in cfg.output.metadata_fields:
                if field == "sample_code":
                    continue  # Already handled

                # Determine dtype
                first_value = metadata_list[0][field]
                if isinstance(first_value, bool):
                    dtype = np.bool_
                elif isinstance(first_value, int):
                    dtype = np.int32
                elif isinstance(first_value, float):
                    dtype = np.float32
                else:
                    dtype = np.float32

                data = np.array([m[field] for m in metadata_list], dtype=dtype)
                compression_args = {
                    "compression": cfg.output.compression,
                    "chunks": True,
                    "maxshape": (None,)
                }
                if cfg.output.compression == "gzip":
                    compression_args["compression_opts"] = cfg.output.compression_level
                f.create_dataset(
                    f"metadata/{field}",
                    data=data,
                    **compression_args
                )
        else:
            # Append to existing datasets
            curr_size = f["embeddings"].shape[0]
            new_size = curr_size + embeddings.shape[0]

            # Resize and append embeddings
            f["embeddings"].resize((new_size, embeddings.shape[1]))
            f["embeddings"][curr_size:new_size] = embeddings
            
            # Resize and append sample codes
            dt = h5py.string_dtype(encoding='utf-8')
            sample_codes_data = np.array([sample_code] * len(metadata_list), dtype=object)
            f["sample_codes"].resize((new_size,))
            f["sample_codes"][curr_size:new_size] = sample_codes_data
            
            # Resize and append metadata
            for field in cfg.output.metadata_fields:
                if field == "sample_code":
                    continue
                
                data = np.array([m[field] for m in metadata_list])
                f[f"metadata/{field}"].resize((new_size,))
                f[f"metadata/{field}"][curr_size:new_size] = data
    
    logger.info(f"  Saved {embeddings.shape[0]} embeddings to {output_path}")


@hydra.main(version_base=None, config_path="../config", config_name="preprocess_config")
def main(cfg: DictConfig) -> None:
    """
    Main preprocessing pipeline.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 80)
    logger.info("Starting Preprocessing and Embedding Extraction Pipeline")
    logger.info("=" * 80)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Setup environment (cache dirs, etc.)
    setup_environment(cfg)
    
    # Load resolution CSV
    resolution_df = load_resolution_csv(cfg.paths.resolution_csv, cfg)
    
    # Find all samples in data directory
    data_dir = Path(cfg.paths.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all volume files
    volume_files = sorted(data_dir.glob("*.tif"))
    
    # Filter to only volume files (not masks)
    sample_codes = []
    for f in volume_files:
        if "_tissue" not in f.name and "_mask" not in f.name:
            sample_codes.append(f.stem)
    
    logger.info(f"\nFound {len(sample_codes)} samples to process")
    
    if len(sample_codes) == 0:
        logger.error("No samples found! Check data directory and file patterns.")
        return
    
    # Load UNI model
    model, transform = load_uni_model(cfg)
    
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / cfg.paths.output_filename
    
    # Process each sample
    for idx, sample_code in enumerate(sample_codes):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing sample {idx + 1}/{len(sample_codes)}: {sample_code}")
        logger.info(f"{'=' * 80}")
        
        try:
            # Get resolution for this sample
            original_spacing = get_sample_resolution(sample_code, resolution_df, cfg)
            logger.info(f"  Original spacing: {original_spacing} mm")
            
            # Load data
            volume, tissue_mask, invasion_mask = load_sample_data(
                sample_code, data_dir, cfg
            )
            
            # Preprocess volume
            volume = preprocess_volume(volume, original_spacing, cfg)
            
            # Resample masks to match preprocessed volume
            target_spacing = tuple(cfg.preprocessing.resampling.target_spacing)
            logger.info("  Resampling masks...")
            
            tissue_mask = resample_mask(
                tissue_mask,
                original_spacing,
                target_spacing,
                use_gpu=cfg.preprocessing.resampling.use_gpu,
            )
            
            invasion_mask = resample_mask(
                invasion_mask,
                original_spacing,
                target_spacing,
                use_gpu=cfg.preprocessing.resampling.use_gpu,
            )
            
            # Extract and filter patches
            _, rgb_patches, metadata_list = extract_and_filter_patches(
                volume, tissue_mask, invasion_mask, cfg
            )
            
            if len(rgb_patches) == 0:
                logger.warning(f"  Skipping {sample_code} - no valid patches")
                continue
            
            # Compute embeddings
            embeddings = compute_embeddings(rgb_patches, model, transform, cfg)
            
            # Add sample_code to metadata
            for meta in metadata_list:
                meta["sample_code"] = sample_code
            
            # Save to HDF5
            save_to_hdf5(
                output_path,
                sample_code,
                embeddings,
                metadata_list,
                cfg,
                append=(idx > 0),  # Append for all samples after the first
            )
            
            logger.info(f"  Completed {sample_code}")
            
        except Exception as e:
            logger.error(f"  X Error processing {sample_code}: {e}", exc_info=True)
            continue
    
    logger.info(f"\n{'=' * 80}")
    logger.info("Pipeline completed!")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
