#!/usr/bin/env python
"""
Debug Farneback algorithm with a single step, no pyramid.

This script runs a single step of the Farneback algorithm without pyramidal processing
to identify potential bugs.
"""

import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.io import savemat
import time

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the Farneback implementation
from Farneback_Numba_optimized import Farneback_Numba

def load_parabolic_images():
    """Load the parabolic test images from the Bits08 dataset."""
    base_path = os.path.join('testImages', 'Bits08', 'Ni06')
    fn1 = os.path.join(base_path, 'parabolic01_0.tif')
    fn2 = os.path.join(base_path, 'parabolic01_1.tif')
    
    print(f"Loading images from {fn1} and {fn2}")
    
    # Load images
    im1 = imread(fn1).astype(np.float32)
    im2 = imread(fn2).astype(np.float32)
    
    return im1, im2

def visualize_flow(U, V, title, output_file=None):
    """Visualize flow field as quiver plot."""
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create quiver plot with color-coded arrows based on magnitude
    quiver_skip = 20
    y, x = np.mgrid[0:U.shape[0]:quiver_skip, 0:U.shape[1]:quiver_skip]
    u_skip = U[::quiver_skip, ::quiver_skip]
    v_skip = V[::quiver_skip, ::quiver_skip]
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(u_skip**2 + v_skip**2)
    
    # Plot quiver with colors based on magnitude
    quiv = plt.quiver(x, y, u_skip, v_skip, magnitude, 
                     scale=50, scale_units='inches',
                     cmap='jet', clim=[0, np.percentile(magnitude, 95)])
    plt.colorbar(quiv, label='Magnitude (pixels/frame)')
    
    plt.title(title)
    plt.xlim(0, U.shape[1])
    plt.ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=200)
        print(f"Flow visualization saved to {output_file}")
    
    plt.close()

def visualize_horizontal_profile(U, V, title, output_file=None):
    """Visualize horizontal profile of flow field."""
    # Extract middle row
    position = U.shape[0] // 2
    
    x = np.arange(U.shape[1])
    u = U[position, :]
    v = V[position, :]
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, u, 'b-', linewidth=2)
    plt.title(f'{title} - Horizontal Velocity Profile')
    plt.xlabel('X (pixels)')
    plt.ylabel('Horizontal Velocity (pixels/frame)')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(x, v, 'r-', linewidth=2)
    plt.title(f'{title} - Vertical Velocity Profile')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=200)
        print(f"Profile visualization saved to {output_file}")
    
    plt.close()

def debug_farneback_single_step(im1, im2, window_size, iterations=1, poly_n=5, poly_sigma=1.2, output_dir="farneback_single_step"):
    """
    Debug a single step of the Farneback algorithm.
    
    Args:
        im1, im2: Input images
        window_size: Window size for Farneback
        iterations: Number of iterations (set to 1 for single step)
        poly_n: Polynomial degree
        poly_sigma: Polynomial sigma
        output_dir: Output directory for debug visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Create Farneback adapter with no pyramid levels
    fb = Farneback_Numba(
        windowSize=window_size,
        Niters=iterations,
        polyN=poly_n,
        polySigma=poly_sigma,
        pyramidalLevels=1  # No pyramid, just one level
    )
    
    # Print parameters
    print(f"\nRunning single step Farneback with parameters:")
    print(f"  Window size: {window_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Polynomial degree: {poly_n}")
    print(f"  Polynomial sigma: {poly_sigma}")
    
    # Run the algorithm
    try:
        start_time = time.time()
        
        # Initialize matrices
        height, width = im1.shape
        M = np.zeros((5*height, width), np.float32)
        bufM = np.zeros((5*height, width), np.float32)
        RA = np.zeros((5*height, width), np.float32)
        RB = np.zeros((5*height, width), np.float32)
        
        # Step 1: Polynomial expansion for both images
        print("Step 1: Polynomial expansion for image A...")
        RA = fb.polynomialExpansion(im1, RA)
        print("Step 1: Polynomial expansion for image B...")
        RB = fb.polynomialExpansion(im2, RB)
        
        # Step 2: Update matrices based on initial flow (zeros)
        print("Step 2: Update matrices based on initial flow...")
        M = fb.updateMatrices(U, V, RA, RB, M)
        
        # Step 3: Apply Gaussian blur to matrices
        print("Step 3: Apply Gaussian blur to matrices...")
        bufM = fb.gaussianBlur5(M, fb.windowSize//2, bufM)
        
        # Swap M and bufM
        M, bufM = bufM, M
        
        # Step 4: Update flow based on matrices
        print("Step 4: Update flow based on matrices...")
        U, V = fb.updateFlow(M, U, V)
        
        elapsed_time = time.time() - start_time
        print(f"Single step completed in {elapsed_time:.2f} seconds")
        
        # Print flow statistics
        print(f"Flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")
        
        # Visualize flow
        visualize_flow(U, V, f"Farneback Single Step (window={window_size})", 
                     os.path.join(output_dir, "flow.png"))
        visualize_horizontal_profile(U, V, f"Farneback Single Step (window={window_size})", 
                                   os.path.join(output_dir, "profile.png"))
        
        # Save flow field
        results = {
            'U': U,
            'V': V,
            'window_size': np.array([[window_size]]),
            'iterations': np.array([[iterations]]),
            'poly_n': np.array([[poly_n]]),
            'poly_sigma': np.array([[poly_sigma]])
        }
        
        output_file = os.path.join(output_dir, f'farneback_window{window_size}.mat')
        savemat(output_file, results)
        print(f"Flow saved to {output_file}")
        
        return U, V
    
    except Exception as e:
        print(f"Error in Farneback single step: {e}")
        return None, None

def main():
    # Load test images
    im1, im2 = load_parabolic_images()
    
    # Debug Farneback algorithm with different window sizes
    window_sizes = [65, 33, 17, 15]  # Must be odd numbers
    
    for window_size in window_sizes:
        output_dir = f'farneback_single_step_window{window_size}'
        
        U, V = debug_farneback_single_step(
            im1, im2,
            window_size=window_size,
            iterations=1,
            poly_n=5,
            poly_sigma=1.2,
            output_dir=output_dir
        )
        
        print(f"Completed window size {window_size}\n")
    
    print("\nDebug complete.")

if __name__ == "__main__":
    main()
