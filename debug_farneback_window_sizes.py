#!/usr/bin/env python
"""
Debug Farneback algorithm with different window sizes.

This script tests the Farneback algorithm with different window sizes,
starting from 64x64 and working down to 16x16, to identify potential bugs.
"""

import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from tqdm import tqdm

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

def save_flow(U, V, window_size, filename):
    """Save flow field to a .mat file."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the flow field
    results = {
        'U': U,
        'V': V,
        'window_size': np.array([[window_size]]),
        'iterations': np.array([[5]]),
        'poly_n': np.array([[5]]),
        'poly_sigma': np.array([[1.2]])
    }
    
    savemat(filename, results)
    print(f"Flow saved to {filename}")

def run_farneback_direct(im1, im2, window_size, iterations=5, poly_n=5, poly_sigma=1.2, pyr_levels=1):
    """
    Run Farneback algorithm directly without pyramidal wrapper.
    
    Args:
        im1, im2: Input images
        window_size: Window size for Farneback
        iterations: Number of iterations
        poly_n: Polynomial degree
        poly_sigma: Polynomial sigma
        pyr_levels: Number of pyramid levels
        
    Returns:
        U, V: Flow components
    """
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Create Farneback adapter
    fb_adapter = Farneback_Numba(
        windowSize=window_size,
        Niters=iterations,
        polyN=poly_n,
        polySigma=poly_sigma,
        pyramidalLevels=pyr_levels
    )
    
    # Run the algorithm
    print(f"\nRunning Farneback with window size {window_size}:")
    print(f"  Iterations: {iterations}")
    print(f"  Polynomial degree: {poly_n}")
    print(f"  Polynomial sigma: {poly_sigma}")
    print(f"  Pyramid levels: {pyr_levels}")
    
    start_time = time.time()
    
    try:
        U, V, error = fb_adapter.compute(im1, im2, U, V)
        print(f"Computation completed in {time.time() - start_time:.2f} seconds")
        print(f"Flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")
        return U, V
    except Exception as e:
        print(f"Error computing optical flow: {e}")
        return None, None

def plot_flow_field(U, V, window_size, output_dir):
    """Plot optical flow as colormesh and quiver plots."""
    if U is None or V is None:
        print(f"Cannot plot flow field for window size {window_size} - computation failed")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Auto-calculate color limits for horizontal velocity
    u_abs_max = max(abs(np.percentile(U, 1)), abs(np.percentile(U, 99)))
    vmin = -u_abs_max
    vmax = u_abs_max
    
    # Plot horizontal velocity component as colormesh with high contrast colormap
    im1 = ax1.imshow(U, cmap='jet', vmin=vmin, vmax=vmax)
    ax1.set_title(f'Farneback (window={window_size}) - Horizontal Velocity (u)')
    plt.colorbar(im1, ax=ax1, label='Pixels/frame')
    
    # Add a grid for better reference
    ax1.grid(False)
    
    # Create quiver plot with color-coded arrows based on magnitude
    quiver_skip = 20
    y, x = np.mgrid[0:U.shape[0]:quiver_skip, 0:U.shape[1]:quiver_skip]
    u_skip = U[::quiver_skip, ::quiver_skip]
    v_skip = V[::quiver_skip, ::quiver_skip]
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(u_skip**2 + v_skip**2)
    
    # Plot quiver with colors based on magnitude
    quiv = ax2.quiver(x, y, u_skip, v_skip, magnitude, 
                     scale=50, scale_units='inches',
                     cmap='jet', clim=[0, np.percentile(magnitude, 95)])
    plt.colorbar(quiv, ax=ax2, label='Magnitude (pixels/frame)')
    
    ax2.set_title(f'Farneback (window={window_size}) - Vector Field')
    ax2.set_xlim(0, U.shape[1])
    ax2.set_ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'farneback_window{window_size}_flow.png')
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"Flow plot saved to {output_file}")

def extract_horizontal_profile(U, V):
    """
    Extract a horizontal profile from the flow field.
    
    Args:
        U, V: Horizontal and vertical velocity components
        
    Returns:
        x: Position along the horizontal axis
        u: Horizontal velocity profile
        v: Vertical velocity profile
    """
    # Extract middle row
    position = U.shape[0] // 2
    
    x = np.arange(U.shape[1])
    u = U[position, :]
    v = V[position, :]
    
    return x, u, v

def plot_horizontal_profile(U, V, window_size, output_dir):
    """Plot horizontal profile of the flow field."""
    if U is None or V is None:
        print(f"Cannot plot horizontal profile for window size {window_size} - computation failed")
        return
    
    # Extract horizontal profile
    x, u, v = extract_horizontal_profile(U, V)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot horizontal velocity profile
    ax1.plot(x, u, 'b-', linewidth=2)
    ax1.set_title(f'Farneback (window={window_size}) - Horizontal Velocity Profile')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Horizontal Velocity (pixels/frame)')
    ax1.grid(True)
    
    # Plot vertical velocity profile
    ax2.plot(x, v, 'r-', linewidth=2)
    ax2.set_title(f'Farneback (window={window_size}) - Vertical Velocity Profile')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Vertical Velocity (pixels/frame)')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'farneback_window{window_size}_profile.png')
    plt.savefig(output_file, dpi=200)
    plt.close()
    print(f"Profile plot saved to {output_file}")

def main():
    # Load test images
    im1, im2 = load_parabolic_images()
    
    # Create output directory
    output_dir = 'farneback_debug'
    os.makedirs(output_dir, exist_ok=True)
    
    # Window sizes to test
    window_sizes = [65, 33, 25, 17, 15]  # Must be odd numbers
    
    # Fixed parameters
    iterations = 5
    poly_n = 5
    poly_sigma = 1.2
    pyr_levels = 1  # Start with a single level for simplicity
    
    # Run Farneback with different window sizes
    for window_size in window_sizes:
        # Run Farneback algorithm
        U, V = run_farneback_direct(
            im1, im2,
            window_size=window_size,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            pyr_levels=pyr_levels
        )
        
        if U is not None and V is not None:
            # Save flow field
            save_flow(
                U, V, window_size,
                os.path.join(output_dir, f'farneback_window{window_size}.mat')
            )
            
            # Plot flow field
            plot_flow_field(U, V, window_size, output_dir)
            
            # Plot horizontal profile
            plot_horizontal_profile(U, V, window_size, output_dir)
    
    print("\nDebug complete. Results saved to", output_dir)

if __name__ == "__main__":
    main()
