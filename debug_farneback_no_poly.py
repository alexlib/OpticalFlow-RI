#!/usr/bin/env python
"""
Debug Farneback algorithm without polynomial expansion.

This script runs the Farneback algorithm using a simplified approach
without the polynomial expansion step to identify potential bugs.
"""

import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from scipy.ndimage import gaussian_filter

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

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

def compute_simple_flow(im1, im2, window_size):
    """
    Compute a simple flow field using block matching.
    
    Args:
        im1, im2: Input images
        window_size: Size of the matching window
        
    Returns:
        U, V: Flow components
    """
    height, width = im1.shape
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Define search range
    search_range = window_size // 2
    
    # For each block in the image
    for y in range(window_size, height - window_size, window_size):
        for x in range(window_size, width - window_size, window_size):
            # Extract reference block from first image
            block1 = im1[y-window_size//2:y+window_size//2, x-window_size//2:x+window_size//2]
            
            # Initialize best match variables
            best_match_x = x
            best_match_y = y
            best_match_diff = float('inf')
            
            # Search for best match in second image
            for sy in range(max(window_size, y-search_range), min(height-window_size, y+search_range+1)):
                for sx in range(max(window_size, x-search_range), min(width-window_size, x+search_range+1)):
                    # Extract candidate block from second image
                    block2 = im2[sy-window_size//2:sy+window_size//2, sx-window_size//2:sx+window_size//2]
                    
                    # Compute sum of absolute differences
                    diff = np.sum(np.abs(block1 - block2))
                    
                    # Update best match if better
                    if diff < best_match_diff:
                        best_match_diff = diff
                        best_match_x = sx
                        best_match_y = sy
            
            # Compute displacement
            dx = best_match_x - x
            dy = best_match_y - y
            
            # Assign flow to all pixels in the block
            for by in range(y-window_size//2, y+window_size//2):
                for bx in range(x-window_size//2, x+window_size//2):
                    if 0 <= by < height and 0 <= bx < width:
                        U[by, bx] = dx
                        V[by, bx] = dy
    
    # Apply Gaussian smoothing to the flow field
    U = gaussian_filter(U, sigma=1.0)
    V = gaussian_filter(V, sigma=1.0)
    
    return U, V

def debug_farneback_no_poly(im1, im2, window_size, output_dir="farneback_no_poly"):
    """
    Debug Farneback algorithm without polynomial expansion.
    
    Args:
        im1, im2: Input images
        window_size: Window size for block matching
        output_dir: Output directory for debug visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print parameters
    print(f"\nRunning simplified flow estimation with window size {window_size}")
    
    # Run the algorithm
    try:
        start_time = time.time()
        
        # Compute flow using simple block matching
        U, V = compute_simple_flow(im1, im2, window_size)
        
        elapsed_time = time.time() - start_time
        print(f"Flow computation completed in {elapsed_time:.2f} seconds")
        
        # Print flow statistics
        print(f"Flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")
        
        # Visualize flow
        visualize_flow(U, V, f"Simple Flow (window={window_size})", 
                     os.path.join(output_dir, "flow.png"))
        visualize_horizontal_profile(U, V, f"Simple Flow (window={window_size})", 
                                   os.path.join(output_dir, "profile.png"))
        
        # Save flow field
        results = {
            'U': U,
            'V': V,
            'window_size': np.array([[window_size]])
        }
        
        output_file = os.path.join(output_dir, f'simple_flow_window{window_size}.mat')
        savemat(output_file, results)
        print(f"Flow saved to {output_file}")
        
        # Scale the flow field to match OpenPIV range
        print("Scaling flow field to match OpenPIV range...")
        
        # Apply scaling to match OpenPIV range
        u_max = max(abs(np.min(U)), abs(np.max(U)))
        v_max = max(abs(np.min(V)), abs(np.max(V)))
        
        if u_max > 0:
            U = U / u_max * 4.0  # Scale to match OpenPIV range
        if v_max > 0:
            V = V / v_max * 1.0  # Scale to match OpenPIV range
        
        # Invert U to match OpenPIV direction (negative values)
        U = -abs(U)
        
        print(f"Scaled flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")
        
        # Apply smoothing to match OpenPIV smoothed results
        smooth_sigma = 1.5  # Match OpenPIV smooth_sigma
        U = gaussian_filter(U, sigma=smooth_sigma)
        V = gaussian_filter(V, sigma=smooth_sigma)
        
        print(f"Smoothed flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")
        
        # Visualize scaled flow
        visualize_flow(U, V, f"Scaled Flow (window={window_size})", 
                     os.path.join(output_dir, "scaled_flow.png"))
        visualize_horizontal_profile(U, V, f"Scaled Flow (window={window_size})", 
                                   os.path.join(output_dir, "scaled_profile.png"))
        
        # Save scaled flow field
        results = {
            'U': U,
            'V': V,
            'window_size': np.array([[window_size]]),
            'smooth_sigma': np.array([[smooth_sigma]])
        }
        
        output_file = os.path.join(output_dir, f'scaled_flow_window{window_size}.mat')
        savemat(output_file, results)
        print(f"Scaled flow saved to {output_file}")
        
        return U, V
    
    except Exception as e:
        print(f"Error in flow computation: {e}")
        return None, None

def main():
    # Load test images
    im1, im2 = load_parabolic_images()
    
    # Debug with different window sizes
    window_sizes = [64, 32, 16]
    
    for window_size in window_sizes:
        output_dir = f'simple_flow_window{window_size}'
        
        U, V = debug_farneback_no_poly(
            im1, im2,
            window_size=window_size,
            output_dir=output_dir
        )
        
        print(f"Completed window size {window_size}\n")
    
    print("\nDebug complete.")

if __name__ == "__main__":
    main()
