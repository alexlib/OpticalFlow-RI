#!/usr/bin/env python
"""
PIV analysis with smoothing between windows.
"""

import os
import numpy as np
from skimage.io import imread
from scipy.io import savemat
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Import OpenPIV
from openpiv import pyprocess, filters

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

def analyze_with_piv(frame_a, frame_b, window_size=16, overlap=8, dt=1.0, smooth_sigma=1.0):
    """
    Analyze images using OpenPIV with smoothing between windows.
    
    Args:
        frame_a, frame_b: Input frames
        window_size: Size of the interrogation window
        overlap: Overlap between windows
        dt: Time step between frames
        smooth_sigma: Sigma for Gaussian smoothing
        
    Returns:
        x, y: Coordinates of the vectors
        u, v: Velocity components
    """
    print(f"Analyzing with PIV (window_size={window_size}, overlap={overlap}, smooth_sigma={smooth_sigma})...")
    
    # Make sure input images are valid
    frame_a = np.asarray(frame_a).astype(np.int32)
    frame_b = np.asarray(frame_b).astype(np.int32)
    
    # Process the images to get the vector field
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=window_size,
        sig2noise_method='peak2peak'
    )
    
    # Create a mask for invalid vectors (all False for now)
    mask = np.zeros_like(u, dtype=bool)
    
    # Replace outliers
    u, v = filters.replace_outliers(u, v, mask, method='localmean', max_iter=3, kernel_size=3)
    
    # Apply Gaussian smoothing to the velocity components
    print(f"Applying Gaussian smoothing with sigma={smooth_sigma}...")
    u_smooth = gaussian_filter(u, sigma=smooth_sigma)
    v_smooth = gaussian_filter(v, sigma=smooth_sigma)
    
    # Check for NaN or Inf values in the results
    if np.any(np.isnan(u_smooth)) or np.any(np.isinf(u_smooth)) or np.any(np.isnan(v_smooth)) or np.any(np.isinf(v_smooth)):
        print("Warning: PIV analysis produced NaN or Inf values. Replacing with zeros.")
        u_smooth = np.nan_to_num(u_smooth, nan=0, posinf=0, neginf=0)
        v_smooth = np.nan_to_num(v_smooth, nan=0, posinf=0, neginf=0)
    
    # Calculate the coordinates
    n_rows, n_cols = u.shape
    
    # Calculate the coordinates
    x = np.arange(window_size//2, frame_a.shape[1]-window_size//2+1, window_size-overlap)
    y = np.arange(window_size//2, frame_a.shape[0]-window_size//2+1, window_size-overlap)
    
    # Make sure the coordinate arrays match the velocity field dimensions
    if len(x) > n_cols:
        x = x[:n_cols]
    if len(y) > n_rows:
        y = y[:n_rows]
    
    # Create meshgrid
    x, y = np.meshgrid(x, y)
    
    return x, y, u_smooth, v_smooth

def interpolate_to_full_grid(x, y, u, v, shape, method='cubic'):
    """
    Interpolate the vector field to a full grid.
    
    Args:
        x, y: Coordinates of the vectors
        u, v: Velocity components
        shape: Shape of the output grid
        method: Interpolation method ('linear', 'cubic', or 'nearest')
        
    Returns:
        U, V: Interpolated velocity components on a full grid
    """
    print(f"Interpolating to full grid using {method} method...")
    
    # Check for NaN or Inf values in the input data
    u_flat = u.flatten()
    v_flat = v.flatten()
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Find valid indices (no NaN or Inf)
    valid_indices = ~(np.isnan(u_flat) | np.isinf(u_flat) | 
                     np.isnan(v_flat) | np.isinf(v_flat) | 
                     np.isnan(x_flat) | np.isinf(x_flat) | 
                     np.isnan(y_flat) | np.isinf(y_flat))
    
    if not np.all(valid_indices):
        print(f"Warning: Found {np.sum(~valid_indices)} invalid values in vector field. Removing them before interpolation.")
        x_flat = x_flat[valid_indices]
        y_flat = y_flat[valid_indices]
        u_flat = u_flat[valid_indices]
        v_flat = v_flat[valid_indices]
    
    # Create a meshgrid for the full image
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # Flatten the coordinates and velocity components
    points = np.column_stack((y_flat, x_flat))
    
    try:
        # Interpolate u and v components with the specified method
        U = griddata(points, u_flat, (Y, X), method=method, fill_value=0)
        V = griddata(points, v_flat, (Y, X), method=method, fill_value=0)
        
        # Check for NaN or Inf in the result
        if np.any(np.isnan(U)) or np.any(np.isinf(U)) or np.any(np.isnan(V)) or np.any(np.isinf(V)):
            print("Warning: Interpolation produced invalid values. Using nearest neighbor interpolation instead.")
            U = griddata(points, u_flat, (Y, X), method='nearest', fill_value=0)
            V = griddata(points, v_flat, (Y, X), method='nearest', fill_value=0)
    
    except Exception as e:
        print(f"Error during interpolation: {e}. Using nearest neighbor interpolation instead.")
        try:
            # Fall back to nearest neighbor interpolation
            U = griddata(points, u_flat, (Y, X), method='nearest', fill_value=0)
            V = griddata(points, v_flat, (Y, X), method='nearest', fill_value=0)
        except Exception as e2:
            print(f"Error during nearest neighbor interpolation: {e2}. Returning zeros.")
            U = np.zeros(shape)
            V = np.zeros(shape)
    
    return U, V

def save_results_to_mat(x, y, u, v, U, V, window_size, overlap, smooth_sigma, filename):
    """
    Save PIV results to a .mat file.
    
    Args:
        x, y: Coordinates of the vectors
        u, v: Velocity components (sparse grid)
        U, V: Interpolated velocity components (full grid)
        window_size, overlap: PIV parameters
        smooth_sigma: Sigma for Gaussian smoothing
        filename: Output filename
    """
    # Create a dictionary with all the results
    results = {
        'x': x,
        'y': y,
        'u': u,
        'v': v,
        'U': U,
        'V': V,
        'window_size': window_size,
        'overlap': overlap,
        'smooth_sigma': smooth_sigma
    }
    
    # Save to .mat file
    savemat(filename, results)
    print(f"Results saved to {filename}")

def main():
    # Load test images
    frame_a, frame_b = load_parabolic_images()
    
    # Create output directory
    output_dir = 'piv_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameters
    window_size = 16  # Larger window for better correlation
    overlap = 8       # 50% overlap
    smooth_sigma = 1.5  # Sigma for Gaussian smoothing
    
    # Analyze with PIV
    x, y, u, v = analyze_with_piv(
        frame_a, frame_b, 
        window_size=window_size, 
        overlap=overlap,
        smooth_sigma=smooth_sigma
    )
    
    # Interpolate to full grid using cubic interpolation
    U, V = interpolate_to_full_grid(x, y, u, v, frame_a.shape, method='cubic')
    
    # Save results to .mat file
    save_results_to_mat(
        x, y, u, v, U, V,
        window_size=window_size, 
        overlap=overlap,
        smooth_sigma=smooth_sigma,
        filename=os.path.join(output_dir, 'piv_smoothed_results.mat')
    )

if __name__ == "__main__":
    main()
