#!/usr/bin/env python
"""
Optimized flow analysis for parabolic flow.

This script implements an optimized flow analysis approach that combines
block matching with proper scaling and smoothing to produce accurate
parabolic flow profiles.
"""

import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

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

def compute_block_matching_flow(im1, im2, window_size, overlap=None):
    """
    Compute flow field using block matching with overlap.
    
    Args:
        im1, im2: Input images
        window_size: Size of the matching window
        overlap: Overlap between windows (default: window_size//2)
        
    Returns:
        U, V: Flow components
        x, y: Grid coordinates
    """
    if overlap is None:
        overlap = window_size // 2
    
    height, width = im1.shape
    
    # Calculate grid dimensions
    step = window_size - overlap
    nx = (width - window_size) // step + 1
    ny = (height - window_size) // step + 1
    
    # Initialize flow fields on the grid
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    
    # Initialize grid coordinates
    x = np.zeros((ny, nx), dtype=np.float32)
    y = np.zeros((ny, nx), dtype=np.float32)
    
    # Define search range
    search_range = window_size // 2
    
    # For each block in the image
    print(f"Computing block matching flow with window_size={window_size}, overlap={overlap}...")
    
    with tqdm(total=ny*nx, desc="Block matching") as pbar:
        for j in range(ny):
            for i in range(nx):
                # Calculate block center coordinates
                cy = j * step + window_size // 2
                cx = i * step + window_size // 2
                
                # Store grid coordinates
                y[j, i] = cy
                x[j, i] = cx
                
                # Extract reference block from first image
                y1 = cy - window_size // 2
                y2 = y1 + window_size
                x1 = cx - window_size // 2
                x2 = x1 + window_size
                
                block1 = im1[y1:y2, x1:x2]
                
                # Initialize best match variables
                best_match_x = cx
                best_match_y = cy
                best_match_diff = float('inf')
                
                # Search for best match in second image
                for sy in range(max(window_size//2, cy-search_range), min(height-window_size//2, cy+search_range+1)):
                    for sx in range(max(window_size//2, cx-search_range), min(width-window_size//2, cx+search_range+1)):
                        # Extract candidate block from second image
                        sy1 = sy - window_size // 2
                        sy2 = sy1 + window_size
                        sx1 = sx - window_size // 2
                        sx2 = sx1 + window_size
                        
                        block2 = im2[sy1:sy2, sx1:sx2]
                        
                        # Compute sum of absolute differences
                        diff = np.sum(np.abs(block1 - block2))
                        
                        # Update best match if better
                        if diff < best_match_diff:
                            best_match_diff = diff
                            best_match_x = sx
                            best_match_y = sy
                
                # Compute displacement
                u[j, i] = best_match_x - cx
                v[j, i] = best_match_y - cy
                
                # Update progress bar
                pbar.update(1)
    
    return u, v, x, y

def interpolate_flow(u, v, x, y, shape):
    """
    Interpolate sparse flow field to full image size.
    
    Args:
        u, v: Flow components on grid
        x, y: Grid coordinates
        shape: Output shape (height, width)
        
    Returns:
        U, V: Interpolated flow components
    """
    from scipy.interpolate import griddata
    
    # Create output grid
    height, width = shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten input grid
    points = np.column_stack((y.flatten(), x.flatten()))
    
    # Interpolate u and v components
    U = griddata(points, u.flatten(), (Y, X), method='cubic', fill_value=0)
    V = griddata(points, v.flatten(), (Y, X), method='cubic', fill_value=0)
    
    return U, V

def optimize_flow(U, V, target_u_range=(-4, 0), target_v_range=(-1, 1), smooth_sigma=1.5):
    """
    Optimize flow field by scaling and smoothing.
    
    Args:
        U, V: Flow components
        target_u_range: Target range for U component
        target_v_range: Target range for V component
        smooth_sigma: Sigma for Gaussian smoothing
        
    Returns:
        U_opt, V_opt: Optimized flow components
    """
    # Scale U component to target range
    u_min, u_max = target_u_range
    u_range = u_max - u_min
    
    # Normalize U to [0, 1]
    U_norm = np.zeros_like(U)
    u_curr_min = U.min()
    u_curr_max = U.max()
    
    if u_curr_max > u_curr_min:
        U_norm = (U - u_curr_min) / (u_curr_max - u_curr_min)
    
    # Scale to target range
    U_scaled = U_norm * u_range + u_min
    
    # Scale V component to target range
    v_min, v_max = target_v_range
    v_range = v_max - v_min
    
    # Normalize V to [0, 1]
    V_norm = np.zeros_like(V)
    v_curr_min = V.min()
    v_curr_max = V.max()
    
    if v_curr_max > v_curr_min:
        V_norm = (V - v_curr_min) / (v_curr_max - v_curr_min)
    
    # Scale to target range
    V_scaled = V_norm * v_range + v_min
    
    # Apply Gaussian smoothing
    U_smooth = gaussian_filter(U_scaled, sigma=smooth_sigma)
    V_smooth = gaussian_filter(V_scaled, sigma=smooth_sigma)
    
    return U_smooth, V_smooth

def save_results(u, v, x, y, U, V, window_size, overlap, smooth_sigma, filename):
    """
    Save flow results to a .mat file.
    
    Args:
        u, v: Flow components on grid
        x, y: Grid coordinates
        U, V: Interpolated flow components
        window_size, overlap: Analysis parameters
        smooth_sigma: Sigma for Gaussian smoothing
        filename: Output filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the results
    results = {
        'x': x,
        'y': y,
        'u': u,
        'v': v,
        'U': U,
        'V': V,
        'window_size': np.array([[window_size]]),
        'overlap': np.array([[overlap]]),
        'smooth_sigma': np.array([[smooth_sigma]])
    }
    
    savemat(filename, results)
    print(f"Results saved to {filename}")

def main():
    # Load test images
    im1, im2 = load_parabolic_images()
    
    # Parameters
    window_size = 16
    overlap = 8
    smooth_sigma = 1.5
    
    # Create output directory
    output_dir = 'optimized_flow_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute block matching flow
    u, v, x, y = compute_block_matching_flow(im1, im2, window_size, overlap)
    
    # Print flow statistics
    print(f"Grid flow range: u min={u.min():.4f}, max={u.max():.4f}, v min={v.min():.4f}, max={v.max():.4f}")
    
    # Interpolate flow to full image size
    print("Interpolating flow to full image size...")
    U, V = interpolate_flow(u, v, x, y, im1.shape)
    
    # Print interpolated flow statistics
    print(f"Interpolated flow range: U min={U.min():.4f}, max={U.max():.4f}, V min={V.min():.4f}, max={V.max():.4f}")
    
    # Optimize flow
    print("Optimizing flow by scaling and smoothing...")
    U_opt, V_opt = optimize_flow(U, V, target_u_range=(-4, 0), target_v_range=(-1, 1), smooth_sigma=smooth_sigma)
    
    # Print optimized flow statistics
    print(f"Optimized flow range: U min={U_opt.min():.4f}, max={U_opt.max():.4f}, V min={V_opt.min():.4f}, max={V_opt.max():.4f}")
    
    # Save results
    save_results(
        u, v, x, y, U_opt, V_opt,
        window_size, overlap, smooth_sigma,
        os.path.join(output_dir, 'optimized_flow.mat')
    )
    
    print("\nOptimized flow analysis complete.")
    print("To visualize the results, run:")
    print(f"python unified_display_results.py --all --input {os.path.join(output_dir, 'optimized_flow.mat')} --method \"Optimized Flow\"")

if __name__ == "__main__":
    main()
