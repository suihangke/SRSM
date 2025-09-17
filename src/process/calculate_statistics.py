import numpy as np
import glob
import os

def calculate_spheroid_statistics(srs_image: np.ndarray, mask_3d: np.ndarray):
    """
    Calculate statistics for masked SRS spheroid data.
    
    Args:
        srs_image: SRS data array [n_wavenumbers, n_z_slices, height, width]
        mask_3d: 3D binary mask [n_z_slices, height, width]
        
    Returns:
        Dictionary containing 3D and 2D statistics for each wavenumber
    """
    n_wavenumbers, n_z_slices, height, width = srs_image.shape
    
    # Validate mask shape
    mask_3d = mask_3d.astype(bool)
    assert mask_3d.shape == (n_z_slices, height, width), f"mask_3d shape {mask_3d.shape} mismatch with expected {(n_z_slices, height, width)}"
    
    # Expand mask to match srs_image dimensions [n_wavenumbers, n_z_slices, height, width]
    mask_4d = np.broadcast_to(mask_3d[None, :, :, :], srs_image.shape)

    masked_data_3d = np.ma.masked_array(srs_image, mask=~mask_4d)
    
    # Calculate 3D stats using masked array operations
    mean_3d = np.ma.mean(masked_data_3d, axis=(1, 2, 3))
    std_3d = np.ma.std(masked_data_3d, axis=(1, 2, 3))
    count_3d = np.ma.count(masked_data_3d, axis=(1, 2, 3))
    
    # Calculate 2D statistics (per z-slice)
    mean_2d = np.ma.mean(masked_data_3d, axis=(2, 3))
    std_2d = np.ma.std(masked_data_3d, axis=(2, 3))
    count_2d = np.ma.count(masked_data_3d, axis=(2, 3))

    # Calculate lipid/protein ratio mean
    ratio_mean_3d = (masked_data_3d[2] / masked_data_3d[6]).mean(axis=(0, 1, 2))
    ratio_mean_2d = (masked_data_3d[2] / masked_data_3d[6]).mean(axis=(1,2))

    stats = {
            'mean': {
                '3d': np.ma.filled(mean_3d, np.nan), # [n_wavenumbers]
                '2d': np.ma.filled(mean_2d, np.nan), # [n_wavenumbers, n_z_slices]
            },
            'std': {
                '3d': np.ma.filled(std_3d, np.nan), # [n_wavenumbers]
                '2d': np.ma.filled(std_2d, np.nan), # [n_wavenumbers, n_z_slices]
            },
            'pixel_count': {
                '3d': count_3d.astype(int), # [n_wavenumbers]
                '2d': count_2d.astype(int), # [n_wavenumbers, n_z_slices]
            },
            'lipid_protein_ratio_mean': {
                '3d': np.ma.filled(ratio_mean_3d, np.nan), # an integer
                '2d': np.ma.filled(ratio_mean_2d, np.nan), # [n_z_slices]
            }
        }

    return stats


def print_single_spheroid_statistics_summary(stats: dict, wavenumbers: np.ndarray = None) -> None:
    """
    Print a summary of calculated statistics.
    
    Args:
        stats: Statistics dictionary from calculate_spheroid_statistics
        wavenumbers: Optional wavenumber array for labeling
    """
    n_wavenumbers = len(stats['mean']['3d'])
    n_z_slices = stats['mean']['2d'].shape[1]
    
    print(f"Spheroid Statistics Summary:")
    print(f"  Wavenumbers: {n_wavenumbers}, Z-slices: {n_z_slices}")
    print(f"  Total 3D pixels: {stats['pixel_count']['3d'].sum()}")
    print()
    
    # 3D statistics
    print("3D Statistics (across all z-slices):")
    for w in range(n_wavenumbers):
        wavenumber_label = f" ({wavenumbers[w]:.0f} cm⁻¹)" if wavenumbers is not None and len(wavenumbers) > w else ""
        mean_val = stats['mean']['3d'][w]
        std_val = stats['std']['3d'][w]
        pixel_count = stats['pixel_count']['3d'][w]
        
        if not np.isnan(mean_val):
            print(f"  W{w}{wavenumber_label}: mean={mean_val:.2f}, std={std_val:.2f}, pixels={pixel_count}")
        else:
            print(f"  W{w}{wavenumber_label}: No valid pixels")
    
    print()
    
    # 2D statistics summary
    print("2D Statistics (per z-slice):")
    for z in range(n_z_slices):
        print(f"  Z{z}:")
        for w in range(n_wavenumbers):
            wavenumber_label = f" ({wavenumbers[w]:.0f} cm⁻¹)" if wavenumbers is not None and len(wavenumbers) > w else ""
            mean_val = stats['mean']['2d'][w, z]
            std_val = stats['std']['2d'][w, z]
            pixel_count = stats['pixel_count']['2d'][w, z]
            
            if not np.isnan(mean_val):
                print(f"    W{w}{wavenumber_label}: mean={mean_val:.2f}, std={std_val:.2f}, pixels={pixel_count}")
            else:
                print(f"    W{w}{wavenumber_label}: No valid pixels")
