import h5py
import numpy as np
import os
import re
from pathlib import Path

def load_h5_file(h5_path: str):
    """
    Load H5 file and return the file object.
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    return h5py.File(h5_path, 'r')

def load_srs_data(h5_path: str):
    """
    Load SRS data from H5 file.
    Returns:
        dict: Dictionary containing SRS data and metadata
            - 'data': numpy array with shape [10, 3, 512, 512]
            - 'wavenumbers': list of wavenumbers
            - 'attributes': dict of all SRS dataset attributes
    """
    with load_h5_file(h5_path) as f:
        if 'SRS' not in f:
            raise KeyError("SRS dataset not found in H5 file")
        
        srs_dataset = f['SRS']
        data = srs_dataset[()]

        wavenumbers = None
        if 'Wavenumbers (cm-1)' in srs_dataset.attrs:
            wavenumbers = srs_dataset.attrs['Wavenumbers (cm-1)'].tolist()

        attributes = dict(srs_dataset.attrs)
        
        return {
            'data': data,
            'wavenumbers': wavenumbers,
            'attributes': attributes,
            'shape': data.shape,
            'dtype': str(data.dtype)
        }

def load_dc_data(h5_path: str):
    """
    Load DC data from H5 file.
    Returns:
        dict: Dictionary containing DC data and metadata
            - 'data': numpy array with shape [10, 3, 512, 512]
            - 'attributes': dict of all DC dataset attributes
    """
    with load_h5_file(h5_path) as f:
        if 'DC' not in f:
            raise KeyError("DC dataset not found in H5 file")
        
        dc_dataset = f['DC']
        data = dc_dataset[()]
        
        attributes = dict(dc_dataset.attrs)
        
        return {
            'data': data,
            'attributes': attributes,
            'shape': data.shape,
            'dtype': str(data.dtype)
        }

def extract_spheroid_id_and_day_idx(filename: str, doe_id: int = 11) -> tuple[str, int]:
    """
    Args:
        filename (str): H5 filename (e.g., 'Plate_80_A1_D1.h5')
        doe_id (int): DOE ID number (default: 11)
    Returns:
        str: Spheroid ID in format '11_80_A01'
        int: Day index
    Example:
        extract_spheroid_id_and_day_idx('Plate_80_A1_D1.h5', 11) -> '11_80_A01', 1
    """
    # Remove .h5 extension
    name = filename.replace('.h5', '')
    
    # Extract plate number
    plate_match = re.search(r'Plate_(\d+)', name)
    plate = plate_match.group(1) if plate_match else ''
    
    # Extract well (e.g., A01, B10, D06, etc.)
    well_match = re.search(r'Plate_\d+_([A-Z]\d+)', name)
    well = well_match.group(1) if well_match else ''
    if plate and well:
        spheroid_id = f"{doe_id}_{plate}_{well.zfill(2)}"
    else:
        raise ValueError(f"Could not extract plate and well from filename: {filename}")

    # Extract day index
    pattern = r'_D(\d+)\.h5$'
    day_idx_match = re.search(pattern, filename)
    if day_idx_match:
        day_idx = int(day_idx_match.group(1))
    else:
        raise ValueError(f"Could not extract day from filename: {filename}")
    
    return spheroid_id, day_idx

