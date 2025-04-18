#!/usr/bin/env python
"""
Simple test script for Farneback Numba implementation
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.data import camera
from Farneback_Numba import Farneback_Numba

def main():
    # Create a simple test case with a small image
    print("Creating test images...")
    img1 = camera()  # Use a standard test image
    img1 = img1.astype(np.float32)
    
    # Create a slightly shifted version of the image
    shift_x = 2
    shift_y = 3
    img2 = np.zeros_like(img1)
    img2[shift_y:, shift_x:] = img1[:-shift_y, :-shift_x]
    
    # Resize to make it even smaller and faster
    small_shape = (100, 100)
    from skimage.transform import resize
    img1_small = resize(img1, small_shape, preserve_range=True).astype(np.float32)
    img2_small = resize(img2, small_shape, preserve_range=True).astype(np.float32)
    
    print(f"Test images created with shape {img1_small.shape}")
    print(f"Ground truth shift: ({shift_x}, {shift_y})")
    
    # Initialize flow fields
    U = np.zeros_like(img1_small)
    V = np.zeros_like(img1_small)
    
    # Create Farneback algorithm with small parameters for quick testing
    print("\nInitializing Farneback Numba algorithm...")
    fb = Farneback_Numba(
        windowSize=7,
        Niters=2,
        polyN=5,
        polySigma=1.1,
        pyramidalLevels=1
    )
    
    # Run the algorithm
    print("\nRunning Farneback optical flow computation...")
    U, V, error = fb.compute(img1_small, img2_small, U, V)
    
    # Print results
    print("\nResults:")
    print(f"Mean flow U: {np.mean(U):.4f} (expected negative value around -{shift_x})")
    print(f"Mean flow V: {np.mean(V):.4f} (expected negative value around -{shift_y})")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
