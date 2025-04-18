#!/usr/bin/env python
"""
Check the contents of a .mat file.
"""

import sys
from scipy.io import loadmat

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_mat_file.py <mat_file>")
        sys.exit(1)
    
    mat_file = sys.argv[1]
    print(f"Checking {mat_file}...")
    
    data = loadmat(mat_file)
    
    print("Keys in the file:", list(data.keys()))
    
    for key in data.keys():
        if key not in ['__header__', '__version__', '__globals__']:
            print(f"{key} shape: {data[key].shape}")
            
            # Print a sample of the data if it's an array
            if hasattr(data[key], 'shape') and len(data[key].shape) > 0:
                if len(data[key].shape) == 2 and data[key].shape[0] == 1 and data[key].shape[1] == 1:
                    print(f"{key} value: {data[key][0, 0]}")
                else:
                    print(f"{key} min: {data[key].min()}, max: {data[key].max()}")

if __name__ == "__main__":
    main()
