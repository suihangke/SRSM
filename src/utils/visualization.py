import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_montage(data: np.ndarray, wavenumbers: list = None, 
                      spheroid_id: str = "", day_label: str = "",
                      output_path: str = None, show_plot: bool = True):
    """
    Visualize images in a 3*10 grid layout (3 z-slices as rows, 10 channels as columns).
    
    Args:
        data (np.ndarray): Images with shape [10, 3, 512, 512]
        wavenumbers (list): List of wavenumbers for channel labels
        spheroid_id (str): Spheroid ID for title
        day_label (str): Day label for title
        output_path (str): Path to save the plot (optional)
        show_plot (bool): Whether to display the plot
    """
    if data.shape != (10, 3, 512, 512):
        raise ValueError(f"Expected data shape (10, 3, 512, 512), got {data.shape}")
    
    # Create figure with 3×10 subplots (rows=z, cols=channels)
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(20, 7), constrained_layout=True)
    
    # Define robust contrast function
    def robust_contrast(img2d):
        p1, p99 = np.percentile(img2d, [1, 99])
        if p99 <= p1:
            return img2d
        return np.clip((img2d - p1) / (p99 - p1), 0, 1)
    
    # Plot each z-slice (rows) and channel (columns)
    for z in range(3):
        for ch in range(10):
            ax = axes[z, ch]
            
            # Get the image data
            img = data[ch, z].astype(np.float32)
            img_disp = robust_contrast(img)
            
            # Display image
            ax.imshow(img_disp, cmap='gray', interpolation='nearest')
            ax.axis('off')
            
            # Column titles: channel info on first row only
            if z == 0:
                if wavenumbers and len(wavenumbers) == 10:
                    ax.set_title(f"Ch{ch} ({wavenumbers[ch]:.0f} cm⁻¹)", fontsize=9, fontweight='bold')
                else:
                    ax.set_title(f"Ch{ch}", fontsize=9, fontweight='bold')
            
            # Row labels: z-slice label on first column only
            if ch == 0:
                ax.set_ylabel(f"Z-slice {z}", fontsize=10, fontweight='bold')
    
    # Add overall title
    title = f"SRS Data Visualization"
    if spheroid_id:
        title += f" - Spheroid {spheroid_id}"
    if day_label:
        title += f" - Day {day_label}"
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Save plot if output path is provided
    if output_path:
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()