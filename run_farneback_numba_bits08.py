#!/usr/bin/env python
"""
Run Farneback_Numba algorithm on the Bits08 test case.
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Import algorithms
from Farneback_Numba import Farneback_Numba
from GenericPyramidalOpticalFlowWrapper import GenericPyramidalOpticalFlowWrapper

def extract_parabolic_profile(U, V, axis='horizontal', position=None):
    """
    Extract a parabolic profile from the flow field.
    
    Args:
        U, V: Horizontal and vertical velocity components
        axis: 'horizontal' or 'vertical'
        position: Position along the perpendicular axis (default: middle)
        
    Returns:
        x: Position along the axis
        v: Velocity profile
    """
    if axis == 'horizontal':
        if position is None:
            position = U.shape[0] // 2
        x = np.arange(U.shape[1])
        v = V[position, :]
    else:  # vertical
        if position is None:
            position = U.shape[1] // 2
        x = np.arange(U.shape[0])
        v = V[:, position]
    
    return x, v

def main():
    print("Starting Farneback_Numba benchmark on Bits08 test case...")
    sys.stdout.flush()
    
    # Create output directory
    output_dir = 'farneback_numba_bits08_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Bits08 test images
    print("Loading test images...")
    sys.stdout.flush()
    
    img1_path = "examples/testImages/Bits08/Ni06/parabolic01_0.tif"
    img2_path = "examples/testImages/Bits08/Ni06/parabolic01_1.tif"
    
    img1 = imread(img1_path).astype(np.float32)
    print("First image loaded.")
    sys.stdout.flush()
    
    img2 = imread(img2_path).astype(np.float32)
    print("Second image loaded.")
    sys.stdout.flush()
    
    # Print image information
    print(f"Image shape: {img1.shape}")
    print(f"Image dtype: {img1.dtype}")
    print(f"Image min/max values: {img1.min()}/{img1.max()}")
    sys.stdout.flush()
    
    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)
    
    # Run Farneback algorithm
    print("\nRunning Farneback_Numba algorithm...")
    sys.stdout.flush()
    
    # Create Farneback adapter
    print("Setting up Farneback adapter...")
    sys.stdout.flush()
    fb_adapter = Farneback_Numba(
        windowSize=33,
        Niters=5,
        polyN=7,
        polySigma=1.5,
        pyramidalLevels=3
    )
    
    # Create pyramidal optical flow wrapper
    print("Creating pyramid wrapper...")
    sys.stdout.flush()
    pyr_of = GenericPyramidalOpticalFlowWrapper(
        fb_adapter,
        filter_sigma=0.0,
        pyr_levels=3
    )
    
    # Run the algorithm
    print("Computing optical flow...")
    sys.stdout.flush()
    start_time = time.time()
    U, V = pyr_of.calculateFlow(img1, img2)
    elapsed_time = time.time() - start_time
    
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")
    sys.stdout.flush()
    
    # Extract profiles
    print("Extracting parabolic profiles...")
    sys.stdout.flush()
    
    # Extract horizontal profile at the middle of the image
    x_h, v_h = extract_parabolic_profile(U, V, axis='horizontal')
    
    # Extract vertical profile at the middle of the image
    x_v, v_v = extract_parabolic_profile(U, V, axis='vertical')
    
    # Plot flow field
    print("Plotting flow field...")
    sys.stdout.flush()
    
    plt.figure(figsize=(15, 5))
    
    # Plot the first image
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    # Plot the second image
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    # Plot the flow
    plt.subplot(1, 3, 3)
    
    # Create a grid of points
    y, x = np.mgrid[0:U.shape[0]:5, 0:U.shape[1]:5]
    
    # Subsample the flow for visualization
    u = U[::5, ::5]
    v = V[::5, ::5]
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(u**2 + v**2)
    
    # Plot quiver with colors based on magnitude
    plt.quiver(x, y, u, v, magnitude, 
               scale=50, scale_units='inches',
               cmap='jet')
    
    plt.colorbar(label='Magnitude (pixels/frame)')
    plt.title('Optical Flow')
    plt.xlim(0, U.shape[1])
    plt.ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    
    plt.suptitle("Farneback (Numba)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "farneback_numba_flow.png"), dpi=150)
    plt.close()
    
    # Plot parabolic profiles
    print("Plotting parabolic profiles...")
    sys.stdout.flush()
    
    plt.figure(figsize=(12, 5))
    
    # Plot horizontal profile
    plt.subplot(1, 2, 1)
    plt.plot(x_h, v_h)
    plt.title('Horizontal Parabolic Profile')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    
    # Plot vertical profile
    plt.subplot(1, 2, 2)
    plt.plot(x_v, v_v)
    plt.title('Vertical Parabolic Profile')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "farneback_numba_profiles.png"), dpi=150)
    plt.close()
    
    print(f"Benchmark completed. Results saved to {output_dir} directory.")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
