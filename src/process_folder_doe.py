import glob
import os
import numpy as np
import argparse

from collections import defaultdict
from utils.load import extract_spheroid_id_and_day_idx, load_srs_data, load_dc_data
from process.denoising import pca_denoise_3d
from process.segmentation import segmentation_pipeline
from process.calculate_statistics import calculate_spheroid_statistics

def segment_single_spheroid(file_path, visualize_output_path=None):
    """
    Process a single spheroid and extract statistics.
    
    Parameters:
    file_path: str, path to h5 file
    """
    # Extract spheroid info
    spheroid_id, _ = extract_spheroid_id_and_day_idx(file_path)
    
    # Load data
    srs_data = load_srs_data(file_path)
    dc_data = load_dc_data(file_path)
    
    srs_image = srs_data['data']
    dc_image = dc_data['data']
    pixel_size_um = srs_data['attributes']['Pix_size (x,y)']
    
    # Denoise using channel index 2 (2855 cm-1)
    denoised_srs = pca_denoise_3d(srs_image[2], n_components_ratio=0.8, standardize=True)
    denoised_dc = pca_denoise_3d(dc_image[2], n_components_ratio=0.8, standardize=True)
    
    # Segmentation pipeline
    mask_3d, slice_metrics, _, _ = segmentation_pipeline(
        denoised_srs, denoised_dc, pixel_size_um, spheroid_id,
        visualize_output_path=visualize_output_path, show_plot=False
    )

    return srs_image, mask_3d, slice_metrics, pixel_size_um
    

def process_folder_batch(folder_path, doe_id=11, output_path=None, save_segmentation_overlay=False):
    """
    Process all h5 files in folder and extract statistics by day.
    
    Returns:
    results_dict: dict with day as key, containing statistics for each file
    """
    # Save results if output path provided
    if output_path:
        import pickle
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if save_segmentation_overlay:
            os.makedirs(os.path.join(output_path, "segmentation_overlay"), exist_ok=True)
            visualize_output_folder = os.path.join(output_path, "segmentation_overlay")
        else:
            visualize_output_folder = None
    
    # Find all h5 files
    h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
    print(f"Found {len(h5_files)} h5 files")
    
    if not h5_files:
        print("No h5 files found in the specified folder")
        return {}
    
    # Group files by day
    files_by_day = defaultdict(list)
    for file_path in h5_files:
        try:
            spheroid_id, day_idx = extract_spheroid_id_and_day_idx(file_path, doe_id)
            if day_idx:
                files_by_day[day_idx].append(file_path)
        except Exception as e:
            print(f"Warning: Could not extract day from {os.path.basename(file_path)}: {e}")
            continue
    
    print(f"Files by day: {dict((k, len(v)) for k, v in files_by_day.items())}")
    
    # Process files for each day
    results = {}
    
    for day in sorted(files_by_day.keys()):  # Process in sorted order
        print(f"\nProcessing {day} ({len(files_by_day[day])} files)...")
        
        day_stats = []
        successful_files = []
        failed_files = []
        
        for file_path in files_by_day[day]:
            filename = os.path.basename(file_path)
            try:
                if visualize_output_folder:
                    visualize_output_path = os.path.join(visualize_output_folder, filename)
                else:
                    visualize_output_path = None

                srs_image, mask_3d, slice_metrics, pixel_size_um = segment_single_spheroid(file_path, visualize_output_path)
                
                # skip if no spheroid detected
                if not np.any(mask_3d):
                    print(f"  Skipping {filename}: No spheroid detected")
                    failed_files.append((filename, "No spheroid detected"))
                    continue
                
                # Calculate statistics for all wavenumbers
                stats = calculate_spheroid_statistics(srs_image, mask_3d)
                # Store results with metadata
                file_result = {
                    'filename': filename,
                    'spheroid_id': spheroid_id,
                    'stats': stats,
                    'slice_metrics': slice_metrics,
                    'pixel_size_um': pixel_size_um,
                    'total_volume_voxels': np.sum(mask_3d)
                }
                
                day_stats.append(file_result)
                successful_files.append(filename)
                
                print(f"  ✓ Processed {filename}")
                
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {str(e)}")
                failed_files.append((filename, str(e)))
                continue
        
        # Store day results
        results[day] = {
            'stats': day_stats,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'n_successful': len(successful_files),
            'n_failed': len(failed_files)
        }
        
        print(f"  Successfully processed {len(successful_files)}/{len(files_by_day[day])} files for {day}")
        if failed_files:
            print(f"  Failed files: {[f[0] for f in failed_files]}")
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {output_path}")
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    total_processed = sum(r['n_successful'] for r in results.values())
    total_failed = sum(r['n_failed'] for r in results.values())
    print(f"Total files processed: {total_processed}")
    print(f"Total files failed: {total_failed}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a folder of h5 files')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing h5 files')
    parser.add_argument('--doe_id', type=int, required=True, help='DOE ID number')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    args = parser.parse_args()
    
    process_folder_batch(args.folder_path, args.doe_id, args.output_path)
    print(f"Results saved to {args.output_path}")
