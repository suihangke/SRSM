import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from skimage import filters, morphology, exposure
from skimage.measure import label, regionprops, find_contours
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from math import pi, sqrt
import matplotlib.pyplot as plt


def segmentation_pipeline(srs_image_2855: np.ndarray, dc_image_2855: np.ndarray, pixel_size_um: Tuple[float, float], spheroid_id: str, visualize_output_path: str = None, show_plot: bool = False):
    mask_3d_srs, slice_metrics_srs, processed_stack_srs = segment_3d_stack(-srs_image_2855, method='otsu', connect_components=True, pixel_size_um=pixel_size_um)
    mask_3d_dc, slice_metrics_dc, processed_stack_dc = segment_3d_stack(-dc_image_2855, method='otsu', connect_components=True, pixel_size_um=pixel_size_um)

    # intesection of the two masks
    mask_3d_intersection = np.logical_and(mask_3d_srs, mask_3d_dc)
    slice_metrics = calculate_mask_metrics(mask_3d_intersection, pixel_size_um)

    visualize_segmentation_overlay(srs_image_2855, mask_3d_intersection,slice_metrics, spheroid_id=spheroid_id, channel_info="SRS 2855cm-1", output_path=visualize_output_path, show_plot=show_plot)

    return mask_3d_intersection, slice_metrics, processed_stack_srs, processed_stack_dc
# =============================================================================
# Preprocessing Functions (Only for Segmentation)
# =============================================================================
def preprocess_for_segmentation(image: np.ndarray, 
                               denoise: bool = True,
                               enhance_contrast: bool = True,
                               normalize: bool = True) -> np.ndarray:
    """
    Preprocess image for segmentation only. Original data remains unchanged.
    
    Args:
        image: Input 2D image
        denoise: Apply denoising
        enhance_contrast: Enhance contrast using percentile stretching  
        normalize: Normalize to 0-1 range
        
    Returns:
        Preprocessed image for segmentation
    """
    # Work on copy to preserve original
    processed = image.astype(np.float64)
    
    # Remove invalid values
    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1. Denoising
    if denoise:
        processed = filters.gaussian(processed, sigma=1, preserve_range=True)
    
    # 2. Contrast enhancement
    if enhance_contrast:
        p1, p99 = np.percentile(processed, [1, 99])
        if p99 > p1:
            processed = exposure.rescale_intensity(processed, in_range=(p1, p99))
    
    # 3. Normalization
    if normalize:
        min_val, max_val = np.min(processed), np.max(processed)
        if max_val > min_val:
            processed = (processed - min_val) / (max_val - min_val)
        else:
            processed = np.zeros_like(processed)
    
    return processed

# =============================================================================
# Core Segmentation Functions
# =============================================================================

def segment_spheroid_otsu(processed_image: np.ndarray) -> np.ndarray:
    """Basic Otsu thresholding segmentation."""
    threshold = filters.threshold_otsu(processed_image)
    binary = processed_image > threshold
    return binary

def segment_spheroid_multi_otsu(processed_image: np.ndarray) -> np.ndarray:
    """Multi-level Otsu thresholding."""
    try:
        thresholds = filters.threshold_multiotsu(processed_image, classes=3)
        binary = processed_image > thresholds[-1]  # Use highest threshold
        return binary
    except:
        # Fallback to regular Otsu
        return segment_spheroid_otsu(processed_image)

def segment_spheroid_adaptive(processed_image: np.ndarray) -> np.ndarray:
    """Adaptive thresholding using Sauvola method."""
    try:
        threshold = filters.threshold_sauvola(processed_image, window_size=31)
        binary = processed_image > threshold
        return binary
    except:
        # Fallback to Otsu
        return segment_spheroid_otsu(processed_image)

def segment_single_slice(image: np.ndarray, 
                        method: str = 'otsu',
                        preprocess: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a single 2D slice.
    
    Args:
        image: Input 2D image (original data)
        method: Segmentation method ('otsu', 'multi_otsu', 'adaptive')
        preprocess: Whether to apply preprocessing
        
    Returns:
        mask: Binary segmentation mask
        processed: Preprocessed image used for segmentation (for visualization)
    """
    # Preprocess for segmentation
    if preprocess:
        processed = preprocess_for_segmentation(image)
    else:
        processed = image.astype(np.float64)
    
    # Apply segmentation method
    if method == 'otsu':
        mask = segment_spheroid_otsu(processed)
    elif method == 'multi_otsu':
        mask = segment_spheroid_multi_otsu(processed)
    elif method == 'adaptive':
        mask = segment_spheroid_adaptive(processed)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    return mask, processed

# =============================================================================
# Post-processing Functions
# =============================================================================

def postprocess_mask(binary_mask: np.ndarray, 
                    min_object_size: int = 500,
                    hole_size_threshold: int = 500,
                    morphology_disk_size: int = 3) -> np.ndarray:
    """
    Post-process binary mask to clean up segmentation.
    
    Args:
        binary_mask: Input binary mask
        min_object_size: Minimum size for objects to keep
        hole_size_threshold: Maximum hole size to fill
        morphology_disk_size: Size of morphological operations
        
    Returns:
        Cleaned binary mask
    """
    mask = binary_mask.copy()
    
    # Remove small objects
    mask = morphology.remove_small_objects(mask, min_size=min_object_size)
    
    # Fill small holes
    mask = morphology.remove_small_holes(mask, area_threshold=hole_size_threshold)
    mask = ndi.binary_fill_holes(mask)
    
    # Get largest connected component
    mask = get_largest_component(mask)
    
    # Morphological operations for smoothing
    disk = morphology.disk(morphology_disk_size)
    mask = morphology.binary_closing(mask, disk)
    mask = morphology.binary_opening(mask, morphology.disk(morphology_disk_size-1))
    
    # Final hole filling
    mask = ndi.binary_fill_holes(mask)
    
    return mask

def get_largest_component(binary_mask: np.ndarray) -> np.ndarray:
    """Get the largest connected component from binary mask."""
    labeled = label(binary_mask)
    if labeled.max() == 0:
        return np.zeros_like(binary_mask, dtype=bool)
    
    # Find largest region
    regions = regionprops(labeled)
    if not regions:
        return np.zeros_like(binary_mask, dtype=bool)
    
    largest_region = max(regions, key=lambda r: r.area)
    return labeled == largest_region.label

def connect_regions_with_convex_hull(binary_mask: np.ndarray) -> np.ndarray:
    """
    Connect multiple regions using convex hull if multiple components exist.
    
    Args:
        binary_mask: Input binary mask
        
    Returns:
        Connected mask using convex hull
    """
    labeled = label(binary_mask)
    if labeled.max() <= 1:
        return binary_mask
    
    # Get coordinates of all foreground pixels
    coords = np.column_stack(np.nonzero(binary_mask))
    
    if len(coords) < 3:
        return binary_mask
    
    try:
        # Create convex hull
        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices]
        
        # Draw filled polygon
        rr, cc = polygon(hull_coords[:, 0], hull_coords[:, 1], binary_mask.shape)
        connected_mask = np.zeros_like(binary_mask, dtype=bool)
        connected_mask[rr, cc] = True
        
        # Smooth the result
        connected_mask = morphology.binary_closing(connected_mask, morphology.disk(3))
        connected_mask = ndi.binary_fill_holes(connected_mask)
        
        return connected_mask
    except:
        # Fallback to original mask
        return binary_mask

# =============================================================================
# 3D Segmentation Pipeline
# =============================================================================

def segment_3d_stack(image_stack: np.ndarray,
                     method: str = 'otsu',
                     connect_components: bool = True,
                     pixel_size_um: Tuple[float, float] = (1.0, 1.0)) -> Tuple[np.ndarray, List[Dict], List[np.ndarray]]:
    """
    Segment 3D image stack slice by slice.
    
    Args:
        image_stack: 3D array [z, h, w]
        method: Segmentation method
        connect_components: Whether to connect multiple components with convex hull
        pixel_size_um: (pixel_x_um, pixel_y_um)
        
    Returns:
        mask_3d: 3D binary mask [z, h, w]
        slice_metrics: List of metrics per slice
        processed_stack: List of preprocessed images for visualization
    """
    z_slices, height, width = image_stack.shape
    mask_3d = np.zeros_like(image_stack, dtype=bool)
    slice_metrics = []
    processed_stack = []
    
    pix_x_um, pix_y_um = pixel_size_um
    
    for z in range(z_slices):
        print(f"Processing slice Z={z}")
        
        # Segment single slice
        mask, processed = segment_single_slice(image_stack[z], method=method)
        
        # Post-process mask
        mask = postprocess_mask(mask)
        
        # Connect components if requested
        if connect_components:
            mask = connect_regions_with_convex_hull(mask)
        
        mask_3d[z] = mask
        processed_stack.append(processed)
        
        # Calculate metrics
        area_px = int(mask.sum())
        area_um2 = float(area_px * pix_x_um * pix_y_um)
        # Equivalent circular diameter
        diameter_px = float(2.0 * sqrt(area_px / pi)) if area_px > 0 else 0.0
        # Use geometric mean pixel size to approximate isotropic pixel scaling for diameter in µm
        px_um_geo = sqrt(pix_x_um * pix_y_um)
        diameter_um = float(diameter_px * px_um_geo)
        
        metrics = {
            "z": z,
            "area_px": area_px,
            "area_um2": area_um2,
            "diameter_px": diameter_px,
            "diameter_um": diameter_um,
        }
        
        print(f"  Z{z}: area={area_px} px ({area_um2:.2f} µm²), diameter≈{diameter_px:.1f} px ({diameter_um:.2f} µm)")
        slice_metrics.append(metrics)
    
    return mask_3d, slice_metrics, processed_stack

def calculate_mask_metrics(mask_3d: np.ndarray, pixel_size_um: Tuple[float, float]) -> Dict:
    """
    Calculate metrics for a 3D mask.
    """
    z_slices, height, width = mask_3d.shape
    pix_x_um, pix_y_um = pixel_size_um
    slice_metrics = []
    for z in range(z_slices):
        area_px = int(mask_3d[z].sum())
        area_um2 = float(area_px * pix_x_um * pix_y_um)
        diameter_px = float(2.0 * sqrt(area_px / pi)) if area_px > 0 else 0.0
        px_um_geo = sqrt(pix_x_um * pix_y_um)
        diameter_um = float(diameter_px * px_um_geo)
        metrics = {
            "z": z,
            "area_px": area_px,
            "area_um2": area_um2,
            "diameter_px": diameter_px,
            "diameter_um": diameter_um,
        }
        slice_metrics.append(metrics)
    return slice_metrics
# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_preprocessed_images(processed_stack: List[np.ndarray],
                                 title_prefix: str = "Preprocessed",
                                 spheroid_id: str = "",
                                 output_path: Optional[str] = None,
                                 show_plot: bool = False) -> None:
    """
    Visualize preprocessed images before segmentation.
    
    Args:
        processed_stack: List of preprocessed 2D images
        title_prefix: Prefix for plot title
        spheroid_id: Spheroid identifier
        output_path: Path to save figure
        show_plot: Whether to display plot
    """
    n_slices = len(processed_stack)
    fig, axes = plt.subplots(1, n_slices, figsize=(4*n_slices, 4))
    
    if n_slices == 1:
        axes = [axes]
    
    for z, processed_img in enumerate(processed_stack):
        ax = axes[z]
        
        # Display with robust contrast
        p1, p99 = np.percentile(processed_img, [1, 99])
        if p99 > p1:
            display_img = np.clip((processed_img - p1) / (p99 - p1), 0, 1)
        else:
            display_img = processed_img
            
        ax.imshow(display_img, cmap='gray', interpolation='nearest')
        ax.set_title(f"{title_prefix} Z{z}", fontsize=10)
        ax.axis('off')
    
    # Overall title
    full_title = f"{title_prefix} Images"
    if spheroid_id:
        full_title += f" - {spheroid_id}"
    fig.suptitle(full_title, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved preprocessed visualization: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def visualize_segmentation_overlay(original_stack: np.ndarray,
                                 mask_3d: np.ndarray,
                                 slice_metrics: Optional[List[Dict]] = None,
                                 spheroid_id: str = "",
                                 channel_info: str = "",
                                 output_path: Optional[str] = None,
                                 show_plot: bool = False) -> None:
    """
    Visualize segmentation overlay with original images and mask contours.
    
    Args:
        original_stack: Original 3D image stack [z, h, w]
        mask_3d: 3D binary mask [z, h, w]
        slice_metrics: Optional metrics per slice
        spheroid_id: Spheroid identifier
        channel_info: Channel information string
        output_path: Path to save figure
        show_plot: Whether to display plot
    """
    z_slices = original_stack.shape[0]
    fig, axes = plt.subplots(1, z_slices, figsize=(5*z_slices, 5))
    
    if z_slices == 1:
        axes = [axes]
    
    for z in range(z_slices):
        ax = axes[z]
        
        # Display original image with robust contrast
        original_slice = original_stack[z]
        p1, p99 = np.percentile(original_slice, [1, 99])
        if p99 > p1:
            display_img = np.clip((original_slice - p1) / (p99 - p1), 0, 1)
        else:
            display_img = original_slice
            
        ax.imshow(display_img, cmap='viridis')
        
        # Overlay mask contours
        if np.any(mask_3d[z]):
            contours = find_contours(mask_3d[z].astype(float), 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=2)

        # Title with metrics
        title = f"Z{z}"
        if channel_info:
            title += f" | {channel_info}"
        if slice_metrics and len(slice_metrics) > z:
            metrics = slice_metrics[z]
            title += f"\nArea: {metrics['area_um2']:.0f} µm², Diameter≈{metrics['diameter_um']:.1f} µm"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Overall title
    full_title = "Segmentation Overlay"
    if spheroid_id:
        full_title += f" - {spheroid_id}"
    fig.suptitle(full_title, fontsize=14)
    
    plt.tight_layout()
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved segmentation overlay: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)