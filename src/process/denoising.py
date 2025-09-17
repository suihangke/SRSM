import numpy as np
from sklearn.decomposition import PCA
from skimage import restoration

def pca_denoise_3d(image_stack, n_components_ratio=0.95, standardize=True):
    """
    Apply PCA denoising to a 3D image stack.
    
    Parameters:
    -----------
    image_stack : ndarray
        3D array of shape (n_wavelengths, height, width) or 4D array (n_wavelengths, n_z, height, width)
    n_components_ratio : float
        Ratio of variance to preserve (0-1) or number of components if > 1
    standardize : bool
        Whether to standardize the data before PCA
    
    Returns:
    --------
    denoised_stack : ndarray
        Denoised image stack with same shape as input
    """
    original_shape = image_stack.shape
    
    # Handle both 3D and 4D arrays
    if len(original_shape) == 4:
        # Reshape 4D to 2D for PCA: (n_wavelengths * n_z, height * width)
        n_wavelengths, n_z, height, width = original_shape
        reshaped = image_stack.reshape(n_wavelengths * n_z, height * width)
    else:
        # Reshape 3D to 2D for PCA: (n_wavelengths, height * width)
        n_wavelengths, height, width = original_shape
        reshaped = image_stack.reshape(n_wavelengths, height * width)
    
    # Transpose to have pixels as samples and spectral bands as features
    data_matrix = reshaped.T
    
    # Standardize if requested
    if standardize:
        mean = np.mean(data_matrix, axis=0)
        std = np.std(data_matrix, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        data_matrix = (data_matrix - mean) / std
    
    # Apply PCA
    pca = PCA(n_components=n_components_ratio)
    transformed = pca.fit_transform(data_matrix)
    
    # Reconstruct the data
    reconstructed = pca.inverse_transform(transformed)
    
    # Reverse standardization if applied
    if standardize:
        reconstructed = reconstructed * std + mean
    
    # Reshape back to original shape
    denoised = reconstructed.T.reshape(original_shape)
    
    print(f"PCA denoising: {pca.n_components_} components used, "
          f"explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return denoised

def wavelet_denoise_3d(image_stack, wavelet='db4', level=None, sigma=None, mode='soft'):
    """
    Apply wavelet denoising to a 3D image stack.
    
    Parameters:
    -----------
    image_stack : ndarray
        3D array of shape (n_wavelengths, height, width) or 4D array
    wavelet : str
        Wavelet to use for decomposition
    level : int or None
        Decomposition level. If None, uses maximum level
    sigma : float or None
        Noise standard deviation. If None, estimates from data
    mode : str
        Thresholding mode: 'soft' or 'hard'
    
    Returns:
    --------
    denoised_stack : ndarray
        Denoised image stack
    """
    denoised = np.zeros_like(image_stack)
    
    # Process each 2D slice independently
    if len(image_stack.shape) == 4:
        for i in range(image_stack.shape[0]):
            for j in range(image_stack.shape[1]):
                # Estimate sigma if not provided
                if sigma is None:
                    sigma_est = restoration.estimate_sigma(image_stack[i, j])
                else:
                    sigma_est = sigma
                
                # Apply wavelet denoising
                denoised[i, j] = restoration.denoise_wavelet(
                    image_stack[i, j],
                    wavelet=wavelet,
                    sigma=sigma_est,
                    mode=mode,
                    wavelet_levels=level,
                    rescale_sigma=True
                )
    else:
        for i in range(image_stack.shape[0]):
            # Estimate sigma if not provided
            if sigma is None:
                sigma_est = restoration.estimate_sigma(image_stack[i])
            else:
                sigma_est = sigma
            
            # Apply wavelet denoising
            denoised[i] = restoration.denoise_wavelet(
                image_stack[i],
                wavelet=wavelet,
                sigma=sigma_est,
                mode=mode,
                wavelet_levels=level,
                rescale_sigma=True
            )
    return denoised