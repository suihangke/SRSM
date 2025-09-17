import h5py
import numpy as np
import os

def explore_h5_structure(file_path: str, max_depth: int = 10):
    """
    Explore and print the complete structure of an HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file
        max_depth (int): Maximum depth to explore (default: 10)
    """
    def explore_group(group, path="", depth=0):
        """Recursively explore HDF5 groups and datasets."""
        if depth > max_depth:
            print("  " * depth + "... (max depth reached)")
            return
            
        for name, item in group.items():
            current_path = f"{path}/{name}" if path else name
            
            if isinstance(item, h5py.Group):
                print("  " * depth + f"📁 {name}/ (Group)")
                # Print group attributes if any
                if item.attrs:
                    print("  " * (depth + 1) + f"  Group Attributes:")
                    for attr_name, attr_value in item.attrs.items():
                        attr_str = str(attr_value)
                        if len(attr_str) > 100:
                            attr_str = attr_str[:97] + "..."
                        print("  " * (depth + 2) + f"    @{attr_name}: {attr_str}")
                explore_group(item, current_path, depth + 1)
                
            elif isinstance(item, h5py.Dataset):
                print("  " * depth + f"📊 {name} (Dataset)")
                print("  " * (depth + 1) + f"  Shape: {item.shape}")
                print("  " * (depth + 1) + f"  Dtype: {item.dtype}")
                print("  " * (depth + 1) + f"  Size: {item.size:,} elements")
                print("  " * (depth + 1) + f"  Path: {current_path}")
                
                # Print dataset attributes if any
                if item.attrs:
                    print("  " * (depth + 1) + f"  Dataset Attributes:")
                    for attr_name, attr_value in item.attrs.items():
                        attr_str = str(attr_value)
                        if len(attr_str) > 100:
                            attr_str = attr_str[:97] + "..."
                        print("  " * (depth + 2) + f"    @{attr_name}: {attr_str}")
                
                # Show data statistics for reasonable sized datasets
                if item.size <= 1000:  # Only show data for small datasets
                    try:
                        sample_data = item[()]
                        if isinstance(sample_data, np.ndarray):
                            print("  " * (depth + 1) + f"  Data: {sample_data}")
                        else:
                            print("  " * (depth + 1) + f"  Value: {sample_data}")
                    except:
                        print("  " * (depth + 1) + "  Data: [Unable to read]")
                else:
                    # Show statistics for large datasets
                    try:
                        if len(item.shape) == 1:
                            print("  " * (depth + 1) + f"  Range: {item[0]} ... {item[-1]}")
                        elif len(item.shape) == 2:
                            print("  " * (depth + 1) + f"  Matrix: {item.shape[0]} x {item.shape[1]}")
                            print("  " * (depth + 1) + f"  Sample: {item[0, 0]} ... {item[-1, -1]}")
                        elif len(item.shape) == 3:
                            print("  " * (depth + 1) + f"  3D Array: {item.shape[0]} x {item.shape[1]} x {item.shape[2]}")
                            print("  " * (depth + 1) + f"  Sample: {item[0, 0, 0]} ... {item[-1, -1, -1]}")
                        elif len(item.shape) == 4:
                            print("  " * (depth + 1) + f"  4D Array: {item.shape[0]} x {item.shape[1]} x {item.shape[2]} x {item.shape[3]}")
                            print("  " * (depth + 1) + f"  Sample: {item[0, 0, 0, 0]} ... {item[-1, -1, -1, -1]}")
                        else:
                            print("  " * (depth + 1) + f"  {len(item.shape)}D Array")
                    except:
                        print("  " * (depth + 1) + "  Statistics: [Unable to compute]")

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"🔍 HDF5 File Structure: {file_path}")
            print("=" * 80)
            
            # File-level information
            print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print(f"Total groups: {len(f.keys())}")
            
            # Count total datasets
            total_datasets = 0
            def count_datasets(group):
                nonlocal total_datasets
                for name, item in group.items():
                    if isinstance(item, h5py.Dataset):
                        total_datasets += 1
                    elif isinstance(item, h5py.Group):
                        count_datasets(item)
            
            count_datasets(f)
            print(f"Total datasets: {total_datasets}")
            print("=" * 80)
            
            # Explore structure
            explore_group(f)
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {str(e)}")

def get_h5_summary(file_path: str):
    """
    Get a summary of the H5 file including all datasets and their shapes.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"📋 H5 File Summary: {file_path}")
            print("=" * 60)
            
            datasets_info = []
            
            def collect_datasets(group, path=""):
                for name, item in group.items():
                    current_path = f"{path}/{name}" if path else name
                    
                    if isinstance(item, h5py.Dataset):
                        datasets_info.append({
                            'path': current_path,
                            'shape': item.shape,
                            'dtype': str(item.dtype),
                            'size': item.size,
                            'attrs': dict(item.attrs)
                        })
                    elif isinstance(item, h5py.Group):
                        collect_datasets(item, current_path)
            
            collect_datasets(f)
            
            # Print summary
            print(f"Total datasets found: {len(datasets_info)}")
            print("\nDataset Details:")
            print("-" * 60)
            
            for i, ds_info in enumerate(datasets_info, 1):
                print(f"{i:2d}. {ds_info['path']}")
                print(f"    Shape: {ds_info['shape']}")
                print(f"    Dtype: {ds_info['dtype']}")
                print(f"    Size: {ds_info['size']:,} elements")
                
                if ds_info['attrs']:
                    print(f"    Attributes: {len(ds_info['attrs'])} found")
                    for attr_name, attr_value in ds_info['attrs'].items():
                        attr_str = str(attr_value)
                        if len(attr_str) > 50:
                            attr_str = attr_str[:47] + "..."
                        print(f"      @{attr_name}: {attr_str}")
                print()
            
            return datasets_info
            
    except Exception as e:
        print(f"❌ Error reading HDF5 file: {str(e)}")
        return None

if __name__ == "__main__":
    # File path to explore
    file_path = "/home/hangkes2/MASCOT/srs/DOE_RUN_compiled/filtered/300_D4-6-8/Plate_80_A1_D1.h5"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Please check the file path.")
    else:
        print(f"✅ File found: {file_path}")
        print()
        
        # Get summary first
        datasets_info = get_h5_summary(file_path)
        
        print("\n" + "=" * 80)
        print("DETAILED STRUCTURE EXPLORATION")
        print("=" * 80)
        
        # Explore detailed structure
        explore_h5_structure(file_path)