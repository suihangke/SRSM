import glob
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
import ast
import re
from typing import Any
import matplotlib.pyplot as plt

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

def process_folder_and_collect_data(folder_path, output_path, doe_id = 11):

    # if exists df, load it
    if os.path.exists(os.path.join(output_path, "stats_df.csv")):
        stats_df = pd.read_csv(os.path.join(output_path, "stats_df.csv"))
        print(f"Stats dataframe loaded from {output_path}")
        return stats_df

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(os.path.join(output_path, "segmentation_overlay")):
        os.makedirs(os.path.join(output_path, "segmentation_overlay"))

    visualize_output_folder = os.path.join(output_path, "segmentation_overlay")

    # Find all h5 files
    h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
    print(f"Found {len(h5_files)} h5 files")

    if not h5_files:
        print("No h5 files found in the specified folder")

    stats_df = pd.DataFrame(columns=['spheroid_id', 'day_idx','mean_3d','mean_2d', 'std_3d','std_2d', 'pixel_count_3d','pixel_count_2d', 'ratio_mean_3d','ratio_mean_2d'])

    # Group files by day
    files_by_day = defaultdict(list)
    failed_files = []
    for file_path in h5_files:
        try:
            spheroid_id, day_idx = extract_spheroid_id_and_day_idx(file_path, doe_id)
            # Save overlay image inside visualize_output_folder using only the filename
            visualize_output_path = os.path.join(visualize_output_folder, os.path.basename(file_path).replace('.h5', '.png'))
            srs_image, mask_3d, slice_metrics, pixel_size_um = segment_single_spheroid(file_path, visualize_output_path)
                
            # skip if no spheroid detected
            if not np.any(mask_3d):
                print(f"  Skipping {file_path}: No spheroid detected")
                failed_files.append((file_path, "No spheroid detected"))
                continue
            
            # Calculate statistics for all wavenumbers
            stats = calculate_spheroid_statistics(srs_image, mask_3d)
            # Append to DataFrame
            new_row = pd.DataFrame({
                'spheroid_id': [spheroid_id],
                'day_idx': [day_idx],
                'mean_3d': [stats['mean']['3d']],
                'mean_2d': [stats['mean']['2d']],
                'std_3d': [stats['std']['3d']],
                'std_2d': [stats['std']['2d']],
                'pixel_count_3d': [stats['pixel_count']['3d']],
                'pixel_count_2d': [stats['pixel_count']['2d']],
                'ratio_3d': [stats['lipid_protein_ratio_mean']['3d']],
                'ratio_2d': [stats['lipid_protein_ratio_mean']['2d']]
            })
            
            stats_df = pd.concat([stats_df, new_row], ignore_index=True)
            print(f"✅Processed {file_path}")
            
        except Exception as e:
            print(f"❌Error processing {os.path.basename(file_path)}: {e}")
            continue
    if output_path:
        stats_df.to_csv(os.path.join(output_path, "stats_df.csv"), index=False)
        print(f"Stats dataframe saved to {output_path}")

    print(f"Files by day: {dict((k, len(v)) for k, v in files_by_day.items())}")
    print(f"Total files: {len(stats_df)}")
    print(f"Failed files: {len(failed_files)}")

    return stats_df


def prepare_d1_to_d2_prediction_data(stats_df):
    """
    Prepare data for D1->D2 prediction: use D1 mean_3d to predict D2 mean_3d.
    
    Parameters:
    stats_df: pd.DataFrame, output from process_folder_and_collect_data
    
    Returns:
    dict: organized data by spheroid with features (D1 mean_3d) and targets (D2 mean_3d)
    """
    # Group by spheroid_id
    spheroid_groups = stats_df.groupby('spheroid_id')
    
    prediction_data = {}
    
    for spheroid_id, group in spheroid_groups:
        # Check if we have both D1 and D2
        try:
            days_available = sorted(int(d) for d in group['day_idx'].unique())
        except Exception:
            days_available = sorted(group['day_idx'].unique())
        
        if 1 not in days_available or 2 not in days_available:
            print(f"Skipping spheroid {spheroid_id}: missing D1 or D2 (days: {days_available})")
            continue
        
        # Extract D1 and D2 data
        d1_data = group[group['day_idx'] == 1].iloc[0]
        d2_data = group[group['day_idx'] == 2].iloc[0]
        
        # Parse mean_3d arrays
        try:
            d1_mean = np.fromstring(d1_data['mean_3d'].strip().strip("[]"), sep=" ")
            d2_mean = np.fromstring(d2_data['mean_3d'].strip().strip("[]"), sep=" ")
        except Exception as e:
            print(f"Skipping spheroid {spheroid_id}: failed to parse mean_3d arrays - {e}")
            continue
        
        # Check for valid data
        if len(d1_mean) == 0 or len(d2_mean) == 0 or np.all(np.isnan(d1_mean)) or np.all(np.isnan(d2_mean)):
            print(f"Skipping spheroid {spheroid_id}: empty or all-NaN arrays")
            continue
        
        # Fill any remaining NaNs with mean
        d1_mean = np.where(np.isnan(d1_mean), np.nanmean(d1_mean), d1_mean)
        d2_mean = np.where(np.isnan(d2_mean), np.nanmean(d2_mean), d2_mean)
        
        prediction_data[spheroid_id] = {
            'features': d1_mean,  # D1 mean_3d as features
            'target': d2_mean,    # D2 mean_3d as target
            'd1_data': d1_data,
            'd2_data': d2_data
        }
    
    print(f"Prepared D1->D2 prediction data for {len(prediction_data)} spheroids")
    return prediction_data

def train_d1_to_d2_model(prediction_data, test_size=0.3, random_state=42):
    """
    Train D1->D2 prediction model using 70/30 train/test split.
    
    Parameters:
    prediction_data: dict, output from prepare_d1_to_d2_prediction_data
    test_size: float, proportion of test set
    random_state: int, random seed for reproducibility
    
    Returns:
    dict: trained models and evaluation results
    """
    # Extract spheroid IDs, features, and targets
    spheroid_ids = list(prediction_data.keys())
    X = np.array([prediction_data[sid]['features'] for sid in spheroid_ids])
    y = np.array([prediction_data[sid]['target'] for sid in spheroid_ids])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    
    # Split data
    (X_train, X_test, y_train, y_test, 
     ids_train, ids_test) = train_test_split(
        X, y, spheroid_ids, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} spheroids")
    print(f"Test set: {len(X_test)} spheroids")
    
    # Train separate models for each wavenumber
    n_wavenumbers = y.shape[1]
    models = {}
    train_scores = {}
    test_scores = {}
    
    # Hold full prediction matrices
    y_train_pred_matrix = np.zeros_like(y_train)
    y_test_pred_matrix = np.zeros_like(y_test)
    
    for wn_idx in range(n_wavenumbers):
        # Train model for this wavenumber
        model = LinearRegression()
        model.fit(X_train, y_train[:, wn_idx])
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_pred_matrix[:, wn_idx] = y_train_pred
        y_test_pred_matrix[:, wn_idx] = y_test_pred
        
        # Evaluation
        train_r2 = r2_score(y_train[:, wn_idx], y_train_pred)
        test_r2 = r2_score(y_test[:, wn_idx], y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train[:, wn_idx], y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test[:, wn_idx], y_test_pred))
        
        models[wn_idx] = model
        train_scores[wn_idx] = {'r2': train_r2, 'rmse': train_rmse}
        test_scores[wn_idx] = {'r2': test_r2, 'rmse': test_rmse}
        
        print(f"Wavenumber {wn_idx}: Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}")
    
    return {
        'models': models,
        'train_scores': train_scores,
        'test_scores': test_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred_matrix,
        'y_test_pred': y_test_pred_matrix,
        'ids_train': ids_train,
        'ids_test': ids_test
    }

def _mean_and_ci(data: np.ndarray, ci: float = 95.0):
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    # Approximate 95% CI with 1.96*std/sqrt(n)
    n = np.maximum(1, np.sum(np.isfinite(data), axis=0))
    sem = std / np.sqrt(n)
    z = 1.96 if ci == 95.0 else 1.96
    lower = mean - z * sem
    upper = mean + z * sem
    return mean, lower, upper

def plot_d1_to_d2_prediction(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             split_name: str,
                             output_path: str) -> None:
    """
    Plot mean D1->D2 prediction and ground truth across wavenumber indices with 95% CI.
    y_true/y_pred: shape (n_samples, n_wavenumbers)
    """
    n_wavenumbers = y_true.shape[1]
    # Preferred wavenumbers axis
    preferred_wavenumbers = np.array([2800.0, 2845.0, 2855.0, 2878.0, 2891.0, 2906.0, 2931.0, 2964.0, 3007.0, 3027.0])
    if len(preferred_wavenumbers) == n_wavenumbers:
        x = preferred_wavenumbers
    else:
        x = np.arange(n_wavenumbers)

    true_mean, true_lo, true_hi = _mean_and_ci(y_true)
    pred_mean, pred_lo, pred_hi = _mean_and_ci(y_pred)

    plt.figure(figsize=(12, 6))
    plt.plot(x, true_mean, label=f"D6 Ground Truth ({split_name})", color="#1f77b4", linewidth=2, marker='o', markersize=3, markevery=max(1, n_wavenumbers//50))
    plt.fill_between(x, true_lo, true_hi, color="#1f77b4", alpha=0.2)

    plt.plot(x, pred_mean, label=f"D6 Prediction from D1 ({split_name})", color="#d62728", linewidth=2, marker='s', markersize=3, markevery=max(1, n_wavenumbers//50))
    plt.fill_between(x, pred_lo, pred_hi, color="#d62728", alpha=0.2)

    plt.xlabel("Wavenumber (cm^-1)")
    plt.ylabel("Mean Intensity")
    plt.title(f"D4→D6 Prediction vs Ground Truth - {split_name} Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_d1_to_d2_predictions(results: dict, output_dir: str) -> None:
    """
    Generate train and test plots for D1->D2 prediction.
    Saves PNGs to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Train
    plot_d1_to_d2_prediction(
        results['y_train'], results['y_train_pred'], 'Train',
        os.path.join(output_dir, 'd1_to_d2_prediction_train.png')
    )
    # Test
    plot_d1_to_d2_prediction(
        results['y_test'], results['y_test_pred'], 'Test',
        os.path.join(output_dir, 'd1_to_d2_prediction_test.png')
    )

def export_d1_to_d2_predictions(results: dict, output_dir: str) -> None:
    """
    Save per-spheroid D1->D2 predictions as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Train predictions
    df_train = pd.DataFrame({
        'spheroid_id': results['ids_train'],
        'd1_mean_3d': [results['X_train'][i].tolist() for i in range(len(results['X_train']))],
        'd2_true_mean_3d': [results['y_train'][i].tolist() for i in range(len(results['y_train']))],
        'd2_pred_mean_3d': [results['y_train_pred'][i].tolist() for i in range(len(results['y_train_pred']))]
    })
    df_train.to_csv(os.path.join(output_dir, 'd1_to_d2_predictions_train.csv'), index=False)
    
    # Test predictions
    df_test = pd.DataFrame({
        'spheroid_id': results['ids_test'],
        'd1_mean_3d': [results['X_test'][i].tolist() for i in range(len(results['X_test']))],
        'd2_true_mean_3d': [results['y_test'][i].tolist() for i in range(len(results['y_test']))],
        'd2_pred_mean_3d': [results['y_test_pred'][i].tolist() for i in range(len(results['y_test_pred']))]
    })
    df_test.to_csv(os.path.join(output_dir, 'd1_to_d2_predictions_test.csv'), index=False)

def _violin_plot(data_lists, labels, title, ylabel, output_path):
    num_samples = len(data_lists)
    if num_samples == 0:
        return
    plt.figure(figsize=(max(8, num_samples * 0.4), 6))
    parts = plt.violinplot(data_lists, showmeans=True, showextrema=False, widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.5)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('#d62728')
        parts['cmeans'].set_linewidth(1.2)
    positions = np.arange(1, num_samples + 1)
    # Overlay median points for clarity
    medians = [np.nanmedian(arr) for arr in data_lists]
    plt.scatter(positions, medians, color='#d62728', s=12, zorder=3, label='Median')
    plt.xticks(positions, labels, rotation=45, ha='right')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def violin_plot_per_sample(results: dict, output_dir: str) -> None:
    """
    Violin plots per wavenumber across samples (x-axis = wavenumbers):
      - Blue: ground truth distribution across samples at each wavenumber
      - Red: prediction distribution across samples at each wavenumber
    Generates plots for both train and test splits.
    """
    os.makedirs(output_dir, exist_ok=True)
    preferred_wavenumbers = np.array([2800.0, 2845.0, 2855.0, 2878.0, 2891.0, 2906.0, 2931.0, 2964.0, 3007.0, 3027.0])
    for split in ['train', 'test']:
        y_true = results[f'y_{split}']
        y_pred = results[f'y_{split}_pred']
        n_samples, n_wavenumbers = y_true.shape
        if len(preferred_wavenumbers) == n_wavenumbers:
            x_labels = preferred_wavenumbers
        else:
            x_labels = np.arange(n_wavenumbers)

        # Collect distributions per wavenumber across samples
        true_per_wn = [y_true[:, i] for i in range(n_wavenumbers)]
        pred_per_wn = [y_pred[:, i] for i in range(n_wavenumbers)]

        # Build side-by-side violins
        plt.figure(figsize=(max(10, n_wavenumbers * 1.0), 6))
        positions = np.arange(n_wavenumbers)

        # Ground truth (blue) shifted left
        parts_true = plt.violinplot(true_per_wn, positions=positions - 0.15, widths=0.25, showmeans=True, showextrema=False)
        for pc in parts_true['bodies']:
            pc.set_facecolor('#1f77b4')
            pc.set_alpha(0.5)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        if 'cmeans' in parts_true:
            parts_true['cmeans'].set_color('#1f77b4')
            parts_true['cmeans'].set_linewidth(1.2)

        # Prediction (red) shifted right
        parts_pred = plt.violinplot(pred_per_wn, positions=positions + 0.15, widths=0.25, showmeans=True, showextrema=False)
        for pc in parts_pred['bodies']:
            pc.set_facecolor('#d62728')
            pc.set_alpha(0.5)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.5)
        if 'cmeans' in parts_pred:
            parts_pred['cmeans'].set_color('#d62728')
            parts_pred['cmeans'].set_linewidth(1.2)

        plt.xticks(positions, [str(x) for x in x_labels], rotation=0)
        plt.xlabel("Wavenumber (cm^-1)")
        plt.ylabel("Intensity")
        plt.title(f"Distributions across Samples by Wavenumber ({split.title()})")
        plt.legend([parts_true['bodies'][0], parts_pred['bodies'][0]], ['Ground Truth', 'Prediction'])
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'violin_per_wavenumber_{split}.png'), dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    folder_path = "/home/hangkes2/MASCOT/srs/DOE_RUN_compiled/filtered/300_D4-6-8"
    doe_id = 11
    output_path = "/home/hangkes2/MASCOT/srs/results/DOE_RUN_compiled/filtered/300_D4-6-8"
    stats_df = process_folder_and_collect_data(folder_path, output_path, doe_id)
    
    # Prepare D1->D2 prediction data
    prediction_data = prepare_d1_to_d2_prediction_data(stats_df)
    
    # Train D1->D2 prediction models
    if len(prediction_data) > 0:
        results = train_d1_to_d2_model(prediction_data, test_size=0.2, random_state=42)
        
        # Visualize results
        visualize_d1_to_d2_predictions(results, "/home/hangkes2/MASCOT/srs/DOE_RUN_compiled/filtered/300_D4-6-8/d1_to_d2_results")
        violin_plot_per_sample(results, "/home/hangkes2/MASCOT/srs/DOE_RUN_compiled/filtered/300_D4-6-8/d1_to_d2_results")
        export_d1_to_d2_predictions(results, "/home/hangkes2/MASCOT/srs/DOE_RUN_compiled/filtered/300_D4-6-8/d1_to_d2_results")

        
        print(f"Results saved to d1_to_d2_results/")
    else:
        print("No valid D1->D2 prediction data available")

    