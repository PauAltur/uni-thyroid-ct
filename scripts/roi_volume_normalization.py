"""
CT Volume Normalization with ROI-based Calibration

This script loads a CT reconstruction from 2D TIFF slices and performs
intensity normalization based on user-defined ROIs for air and wax.

The GUI allows users to:
1. Navigate through the volume depth
2. Draw rectangular ROIs for air and wax regions
3. Perform normalization based on target intensities
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, Slider, RectangleSelector
from matplotlib.patches import Rectangle
import argparse


class CTVolumeNormalizer:
    """
    Interactive CT volume normalization tool with ROI-based calibration.
    """
    
    def __init__(self, volume: np.ndarray, target_air: float = 0.0, target_wax: float = 1.0):
        """
        Initialize the normalizer.
        
        Args:
            volume: 3D numpy array (depth, height, width)
            target_air: Target intensity value for air regions
            target_wax: Target intensity value for wax regions
        """
        self.volume = volume
        self.original_dtype = volume.dtype
        self.target_air = target_air
        self.target_wax = target_wax
        self.normalized_volume = None
        
        # ROI storage: {roi_name: {'slice': int, 'coords': (x1, y1, x2, y2)}}
        self.air_roi = None
        self.wax_roi = None
        
        # For GUI
        self.current_slice = volume.shape[0] // 2
        self.fig = None
        self.ax = None
        self.image_display = None
        self.rect_selector = None
        self.current_roi_type = 'air'  # 'air' or 'wax'
        
    def get_roi_statistics(self, roi_info: Dict) -> Dict[str, float]:
        """
        Calculate statistics for a given ROI.
        
        Args:
            roi_info: Dictionary containing 'slice' and 'coords' keys
            
        Returns:
            Dictionary with mean, std, min, max values
        """
        slice_idx = roi_info['slice']
        x1, y1, x2, y2 = roi_info['coords']
        
        # Ensure coordinates are integers and in correct order
        x1, x2 = int(min(x1, x2)), int(max(x1, x2))
        y1, y2 = int(min(y1, y2)), int(max(y1, y2))
        
        roi_data = self.volume[slice_idx, y1:y2, x1:x2]
        
        return {
            'mean': np.mean(roi_data),
            'std': np.std(roi_data),
            'min': np.min(roi_data),
            'max': np.max(roi_data)
        }
    
    def normalize_volume(self) -> np.ndarray:
        """
        Perform linear normalization based on air and wax ROI means.
        Handles 16-bit data appropriately.
        
        Returns:
            Normalized volume
        """
        if self.air_roi is None or self.wax_roi is None:
            raise ValueError("Both air and wax ROIs must be defined before normalization")
        
        # Get mean intensities from ROIs
        air_stats = self.get_roi_statistics(self.air_roi)
        wax_stats = self.get_roi_statistics(self.wax_roi)
        
        air_mean = air_stats['mean']
        wax_mean = wax_stats['mean']
        
        print(f"\nNormalization parameters:")
        print(f"  Original dtype: {self.original_dtype}")
        print(f"  Original range: [{self.volume.min()}, {self.volume.max()}]")
        print(f"  Air ROI - Mean: {air_mean:.2f}, Std: {air_stats['std']:.2f}")
        print(f"  Wax ROI - Mean: {wax_mean:.2f}, Std: {wax_stats['std']:.2f}")
        print(f"  Target Air: {self.target_air}")
        print(f"  Target Wax: {self.target_wax}")
        
        # Linear normalization: I_norm = (I - air_mean) * scale + target_air
        # where scale = (target_wax - target_air) / (wax_mean - air_mean)
        
        if abs(wax_mean - air_mean) < 1e-6:
            raise ValueError("Air and wax mean values are too similar for normalization")
        
        scale = (self.target_wax - self.target_air) / (wax_mean - air_mean)
        
        # Perform normalization in float64 for precision
        print(f"  Scale factor: {scale:.6f}")
        print(f"  Computing normalized volume...")
        
        volume_float = self.volume.astype(np.float64)
        normalized_float = (volume_float - air_mean) * scale + self.target_air
        
        # Determine output dtype based on target range
        if self._should_keep_16bit():
            # Clip to valid range for the original dtype
            if np.issubdtype(self.original_dtype, np.integer):
                dtype_info = np.iinfo(self.original_dtype)
                normalized_float = np.clip(normalized_float, dtype_info.min, dtype_info.max)
                self.normalized_volume = normalized_float.astype(self.original_dtype)
                print(f"  Output dtype: {self.original_dtype} (preserving original)")
            else:
                self.normalized_volume = normalized_float.astype(self.original_dtype)
                print(f"  Output dtype: {self.original_dtype} (preserving original)")
        else:
            # Keep as float32 for flexibility
            self.normalized_volume = normalized_float.astype(np.float32)
            print(f"  Output dtype: float32 (converted for target range)")
        
        print(f"  Normalized range: [{self.normalized_volume.min()}, {self.normalized_volume.max()}]")
        print(f"  Normalization complete!")
        
        return self.normalized_volume
    
    def _should_keep_16bit(self) -> bool:
        """
        Determine if we should preserve the original 16-bit dtype.
        
        Returns True if target values fit within the original data type range.
        """
        if not np.issubdtype(self.original_dtype, np.integer):
            return True  # Keep original float dtype
        
        dtype_info = np.iinfo(self.original_dtype)
        target_range = abs(self.target_wax - self.target_air)
        
        # Check if target range is reasonable for the dtype
        # (allow some headroom for values outside air-wax range)
        max_expected = max(abs(self.target_air), abs(self.target_wax)) * 2
        
        return max_expected <= dtype_info.max
    
    def launch_gui(self):
        """
        Launch interactive GUI for ROI selection and normalization.
        """
        self.fig = plt.figure(figsize=(14, 8))
        
        # Main image axes
        self.ax = plt.subplot(121)
        self.image_display = self.ax.imshow(
            self.volume[self.current_slice], 
            cmap='gray',
            vmin=np.percentile(self.volume, 1),
            vmax=np.percentile(self.volume, 99)
        )
        self.ax.set_title(f'Slice {self.current_slice}/{self.volume.shape[0]-1}')
        plt.colorbar(self.image_display, ax=self.ax, fraction=0.046)
        
        # Info panel axes
        ax_info = plt.subplot(122)
        ax_info.axis('off')
        self.info_text = ax_info.text(
            0.05, 0.95, self._get_info_text(),
            transform=ax_info.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=9
        )
        
        # Slider for slice navigation
        ax_slider = plt.axes([0.15, 0.02, 0.35, 0.03])
        self.slider = Slider(
            ax_slider, 'Slice', 
            0, self.volume.shape[0] - 1,
            valinit=self.current_slice,
            valstep=1
        )
        self.slider.on_changed(self._update_slice)
        
        # Buttons
        btn_width, btn_height = 0.12, 0.04
        btn_y_start = 0.92
        btn_spacing = 0.06
        
        # Air ROI button
        ax_btn_air = plt.axes([0.55, btn_y_start, btn_width, btn_height])
        self.btn_air = Button(ax_btn_air, 'Select Air ROI')
        self.btn_air.on_clicked(lambda event: self._set_roi_mode('air'))
        
        # Wax ROI button
        ax_btn_wax = plt.axes([0.55, btn_y_start - btn_spacing, btn_width, btn_height])
        self.btn_wax = Button(ax_btn_wax, 'Select Wax ROI')
        self.btn_wax.on_clicked(lambda event: self._set_roi_mode('wax'))
        
        # Clear ROI button
        ax_btn_clear = plt.axes([0.55, btn_y_start - 2*btn_spacing, btn_width, btn_height])
        self.btn_clear = Button(ax_btn_clear, 'Clear ROIs')
        self.btn_clear.on_clicked(self._clear_rois)
        
        # Normalize button
        ax_btn_norm = plt.axes([0.55, btn_y_start - 3*btn_spacing, btn_width, btn_height])
        self.btn_normalize = Button(ax_btn_norm, 'Normalize')
        self.btn_normalize.on_clicked(self._perform_normalization)
        
        # Save button
        ax_btn_save = plt.axes([0.55, btn_y_start - 4*btn_spacing, btn_width, btn_height])
        self.btn_save = Button(ax_btn_save, 'Save Volume')
        self.btn_save.on_clicked(self._save_volume)
        
        # Target value text boxes
        ax_text_air = plt.axes([0.70, btn_y_start, 0.08, btn_height])
        self.text_air = TextBox(ax_text_air, 'Target Air: ', initial=str(self.target_air))
        self.text_air.on_submit(self._update_target_air)
        
        ax_text_wax = plt.axes([0.70, btn_y_start - btn_spacing, 0.08, btn_height])
        self.text_wax = TextBox(ax_text_wax, 'Target Wax: ', initial=str(self.target_wax))
        self.text_wax.on_submit(self._update_target_wax)
        
        # Rectangle selector (initially inactive)
        self.rect_selector = RectangleSelector(
            self.ax, self._on_roi_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False
        )
        self.rect_selector.set_active(False)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08, top=0.98)
        plt.show()
    
    def _get_info_text(self) -> str:
        """Generate information text for the info panel."""
        lines = [
            "=== CT Volume Normalization ===",
            f"Volume shape: {self.volume.shape}",
            f"Data type: {self.original_dtype}",
            f"Current slice: {self.current_slice}/{self.volume.shape[0]-1}",
            f"Data range: [{self.volume.min():.2f}, {self.volume.max()}]",
            "",
            "=== Instructions ===",
            "1. Navigate through slices using slider",
            "2. Click 'Select Air ROI' button",
            "3. Draw rectangle on air region",
            "4. Click 'Select Wax ROI' button",
            "5. Draw rectangle on wax region",
            "6. Click 'Normalize' to process",
            "7. Click 'Save Volume' to export",
            "",
            "=== ROI Information ===",
        ]
        
        if self.air_roi:
            air_stats = self.get_roi_statistics(self.air_roi)
            lines.extend([
                f"Air ROI (Slice {self.air_roi['slice']}):",
                f"  Mean: {air_stats['mean']:.2f}",
                f"  Std:  {air_stats['std']:.2f}",
                f"  Range: [{air_stats['min']:.2f}, {air_stats['max']:.2f}]",
            ])
        else:
            lines.append("Air ROI: Not defined")
        
        lines.append("")
        
        if self.wax_roi:
            wax_stats = self.get_roi_statistics(self.wax_roi)
            lines.extend([
                f"Wax ROI (Slice {self.wax_roi['slice']}):",
                f"  Mean: {wax_stats['mean']:.2f}",
                f"  Std:  {wax_stats['std']:.2f}",
                f"  Range: [{wax_stats['min']:.2f}, {wax_stats['max']:.2f}]",
            ])
        else:
            lines.append("Wax ROI: Not defined")
        
        lines.extend([
            "",
            "=== Target Values ===",
            f"Target Air: {self.target_air}",
            f"Target Wax: {self.target_wax}",
            "",
        ])
        
        if self.normalized_volume is not None:
            lines.extend([
                "=== Normalization Status ===",
                "✓ Volume normalized successfully!",
                f"Output dtype: {self.normalized_volume.dtype}",
                f"Normalized range: [{self.normalized_volume.min():.2f}, {self.normalized_volume.max():.2f}]",
            ])
        
        return "\n".join(lines)
    
    def _update_slice(self, val):
        """Update displayed slice when slider changes."""
        self.current_slice = int(val)
        self.image_display.set_data(self.volume[self.current_slice])
        self.ax.set_title(f'Slice {self.current_slice}/{self.volume.shape[0]-1}')
        self._redraw_rois()
        self.fig.canvas.draw_idle()
    
    def _set_roi_mode(self, roi_type: str):
        """Activate ROI selection mode."""
        self.current_roi_type = roi_type
        self.rect_selector.set_active(True)
        print(f"\nDraw a rectangle for {roi_type.upper()} ROI on the current slice...")
    
    def _on_roi_select(self, eclick, erelease):
        """Callback when ROI rectangle is drawn."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        roi_info = {
            'slice': self.current_slice,
            'coords': (x1, y1, x2, y2)
        }
        
        if self.current_roi_type == 'air':
            self.air_roi = roi_info
            print(f"Air ROI defined on slice {self.current_slice}: ({x1}, {y1}) to ({x2}, {y2})")
        else:
            self.wax_roi = roi_info
            print(f"Wax ROI defined on slice {self.current_slice}: ({x1}, {y1}) to ({x2}, {y2})")
        
        self.rect_selector.set_active(False)
        self._redraw_rois()
        self._update_info_text()
        self.fig.canvas.draw_idle()
    
    def _redraw_rois(self):
        """Redraw ROI rectangles on the current image."""
        # Remove existing ROI patches
        for patch in [p for p in self.ax.patches]:
            patch.remove()
        
        # Draw air ROI if on correct slice
        if self.air_roi and self.air_roi['slice'] == self.current_slice:
            x1, y1, x2, y2 = self.air_roi['coords']
            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height, 
                           linewidth=2, edgecolor='cyan', facecolor='none',
                           label='Air ROI')
            self.ax.add_patch(rect)
            self.ax.text(x1, y1-5, 'AIR', color='cyan', fontweight='bold')
        
        # Draw wax ROI if on correct slice
        if self.wax_roi and self.wax_roi['slice'] == self.current_slice:
            x1, y1, x2, y2 = self.wax_roi['coords']
            width = x2 - x1
            height = y2 - y1
            rect = Rectangle((x1, y1), width, height,
                           linewidth=2, edgecolor='yellow', facecolor='none',
                           label='Wax ROI')
            self.ax.add_patch(rect)
            self.ax.text(x1, y1-5, 'WAX', color='yellow', fontweight='bold')
    
    def _update_info_text(self):
        """Update the information panel."""
        self.info_text.set_text(self._get_info_text())
    
    def _clear_rois(self, event):
        """Clear all defined ROIs."""
        self.air_roi = None
        self.wax_roi = None
        self._redraw_rois()
        self._update_info_text()
        self.fig.canvas.draw_idle()
        print("\nAll ROIs cleared.")
    
    def _update_target_air(self, text):
        """Update target air value."""
        try:
            self.target_air = float(text)
            self._update_info_text()
            print(f"Target air value updated to: {self.target_air}")
        except ValueError:
            print(f"Invalid value for target air: {text}")
    
    def _update_target_wax(self, text):
        """Update target wax value."""
        try:
            self.target_wax = float(text)
            self._update_info_text()
            print(f"Target wax value updated to: {self.target_wax}")
        except ValueError:
            print(f"Invalid value for target wax: {text}")
    
    def _perform_normalization(self, event):
        """Perform volume normalization."""
        try:
            self.normalize_volume()
            self._update_info_text()
            self.fig.canvas.draw_idle()
            
            # Show before/after comparison
            self._show_comparison()
        except Exception as e:
            print(f"\nError during normalization: {str(e)}")
    
    def _show_comparison(self):
        """Show before/after normalization comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        slice_indices = [
            self.volume.shape[0] // 4,
            self.volume.shape[0] // 2,
            3 * self.volume.shape[0] // 4
        ]
        
        for idx, slice_idx in enumerate(slice_indices):
            # Original
            axes[0, idx].imshow(self.volume[slice_idx], cmap='gray')
            axes[0, idx].set_title(f'Original - Slice {slice_idx}')
            axes[0, idx].axis('off')
            
            # Normalized
            axes[1, idx].imshow(self.normalized_volume[slice_idx], cmap='gray')
            axes[1, idx].set_title(f'Normalized - Slice {slice_idx}')
            axes[1, idx].axis('off')
        
        plt.suptitle('Normalization Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _save_volume(self, event):
        """Save the normalized volume."""
        if self.normalized_volume is None:
            print("\nNo normalized volume to save. Please normalize first.")
            return
        
        # Ask for output path
        print("\n" + "="*60)
        print("Enter output path for normalized volume:")
        print("(Press Enter in terminal to provide path)")
        print("="*60)
        
        # This will print to console; user needs to provide path via another mechanism
        # For a full GUI solution, you'd use tkinter.filedialog
        output_path = input("Output path (or 'cancel' to abort): ").strip()
        
        if output_path.lower() == 'cancel':
            print("Save cancelled.")
            return
        
        try:
            save_volume_as_tiff(self.normalized_volume, output_path)
            print(f"\nNormalized volume saved successfully to: {output_path}")
        except Exception as e:
            print(f"\nError saving volume: {str(e)}")


def load_volume_from_tiff_folder(folder_path: str, file_pattern: str = "*.tif*") -> np.ndarray:
    """
    Load a 3D volume from a folder containing 2D TIFF slices.
    Properly handles 16-bit TIFF data.
    
    Args:
        folder_path: Path to folder containing TIFF files
        file_pattern: Glob pattern for TIFF files (default: "*.tif*")
        
    Returns:
        3D numpy array (depth, height, width) with preserved dtype
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all TIFF files matching pattern
    tiff_files = sorted(folder.glob(file_pattern))
    
    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {folder_path} matching pattern {file_pattern}")
    
    print(f"Found {len(tiff_files)} TIFF files in {folder_path}")
    
    # Read first image to get dimensions and dtype
    first_slice = tifffile.imread(str(tiff_files[0]))
    
    # Handle multi-channel images (convert to grayscale if needed)
    if first_slice.ndim == 3:
        print("Warning: Multi-channel image detected. Converting to grayscale using mean.")
        first_slice = first_slice.mean(axis=-1)
    
    height, width = first_slice.shape
    original_dtype = first_slice.dtype
    
    print(f"Image properties:")
    print(f"  Dimensions: {height} x {width}")
    print(f"  Data type: {original_dtype}")
    print(f"  Bit depth: {original_dtype.itemsize * 8} bits")
    
    # Pre-allocate volume with original dtype
    volume = np.zeros((len(tiff_files), height, width), dtype=original_dtype)
    volume[0] = first_slice
    
    # Load remaining slices
    print("Loading slices...")
    for idx, tiff_file in enumerate(tiff_files[1:], start=1):
        slice_data = tifffile.imread(str(tiff_file))
        
        # Convert to grayscale if needed
        if slice_data.ndim == 3:
            slice_data = slice_data.mean(axis=-1).astype(original_dtype)
        
        # Ensure dtype consistency
        if slice_data.dtype != original_dtype:
            print(f"  Warning: Slice {idx} has dtype {slice_data.dtype}, converting to {original_dtype}")
            slice_data = slice_data.astype(original_dtype)
        
        volume[idx] = slice_data
        
        if (idx + 1) % 100 == 0:
            print(f"  Loaded {idx + 1}/{len(tiff_files)} slices...")
    
    print(f"\nVolume loaded successfully:")
    print(f"  Shape: {volume.shape}")
    print(f"  Dtype: {volume.dtype}")
    print(f"  Value range: [{volume.min()}, {volume.max()}]")
    
    if np.issubdtype(volume.dtype, np.integer):
        dtype_info = np.iinfo(volume.dtype)
        print(f"  Dtype range: [{dtype_info.min}, {dtype_info.max}]")
        utilization = (volume.max() - volume.min()) / (dtype_info.max - dtype_info.min) * 100
        print(f"  Dynamic range utilization: {utilization:.1f}%")
    
    return volume


def save_volume_as_tiff(volume: np.ndarray, output_path: str):
    """
    Save a 3D volume as TIFF slices.
    Preserves 16-bit data type if present.
    
    Args:
        volume: 3D numpy array (depth, height, width)
        output_path: Output folder path or single TIFF file path
    """
    output = Path(output_path)
    
    print(f"\nSaving volume:")
    print(f"  Shape: {volume.shape}")
    print(f"  Dtype: {volume.dtype}")
    print(f"  Value range: [{volume.min()}, {volume.max()}]")
    
    # If output is a directory, save as individual slices
    if output.suffix == '':
        output.mkdir(parents=True, exist_ok=True)
        print(f"Saving {volume.shape[0]} slices to {output_path}...")
        
        for idx in range(volume.shape[0]):
            slice_path = output / f"slice_{idx:04d}.tif"
            # tifffile automatically handles 16-bit data
            tifffile.imwrite(str(slice_path), volume[idx], photometric='minisblack')
            
            if (idx + 1) % 100 == 0:
                print(f"  Saved {idx + 1}/{volume.shape[0]} slices...")
        
        print(f"All slices saved to {output_path}")
    
    # Otherwise, save as multi-page TIFF
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving volume as multi-page TIFF to {output_path}...")
        # For 16-bit data, tifffile handles compression and metadata automatically
        tifffile.imwrite(str(output), volume, photometric='minisblack', compression='lzw')
        print(f"Volume saved to {output_path}")
        
        # Verify file size
        file_size_mb = output.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='CT Volume Normalization with ROI-based Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load TIFF slices and launch GUI
  python roi_volume_normalization.py -i /path/to/tiff/folder
  
  # Specify custom target values
  python roi_volume_normalization.py -i /path/to/tiff/folder --target-air 0 --target-wax 100
  
  # Use specific file pattern
  python roi_volume_normalization.py -i /path/to/tiff/folder --pattern "*.tiff"
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to folder containing TIFF slices'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.tif*',
        help='File pattern for TIFF files (default: *.tif*)'
    )
    
    parser.add_argument(
        '--target-air',
        type=float,
        default=0.0,
        help='Target intensity value for air (default: 0.0)'
    )
    
    parser.add_argument(
        '--target-wax',
        type=float,
        default=1.0,
        help='Target intensity value for wax (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Load volume
    try:
        volume = load_volume_from_tiff_folder(args.input, args.pattern)
    except Exception as e:
        print(f"Error loading volume: {str(e)}")
        sys.exit(1)
    
    # Create normalizer and launch GUI
    normalizer = CTVolumeNormalizer(volume, args.target_air, args.target_wax)
    normalizer.launch_gui()


if __name__ == '__main__':
    main()
