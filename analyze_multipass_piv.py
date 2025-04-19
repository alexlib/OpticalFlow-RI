#!/usr/bin/env python
"""
Analyze parabolic images using OpenPIV with multipass down to 8-pixel windows.
"""

import os
import sys
import numpy as np
from skimage.io import imread
from scipy.io import savemat
from scipy.interpolate import griddata
import time

# Import OpenPIV
from openpiv import tools, pyprocess, validation, filters

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

def analyze_with_multipass_piv(frame_a, frame_b, dt=1.0):
    """
    Analyze images using OpenPIV with multipass down to 8-pixel windows.

    Args:
        frame_a, frame_b: Input frames
        dt: Time step between frames

    Returns:
        x, y: Coordinates of the vectors
        u, v: Velocity components
    """
    print("Analyzing with multipass PIV (64px -> 32px -> 16px -> 8px)...")

    # Make sure input images are valid
    frame_a = np.asarray(frame_a).astype(np.int32)
    frame_b = np.asarray(frame_b).astype(np.int32)

    # First pass with 64x64 windows
    window_size = 64
    overlap = window_size // 2
    search_area_size = window_size

    print(f"Pass 1: window_size={window_size}, overlap={overlap}, search_area_size={search_area_size}")
    u1, v1, s2n = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        sig2noise_method='peak2peak'
    )
    flags1 = np.zeros_like(u1, dtype=bool)
    u1, v1 = filters.replace_outliers(u1, v1, flags1, method='localmean', max_iter=3, kernel_size=3)

    # Second pass with 32x32 windows
    window_size = 32
    overlap = window_size // 2
    search_area_size = window_size

    print(f"Pass 2: window_size={window_size}, overlap={overlap}, search_area_size={search_area_size}")
    u2, v2, s2n = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        sig2noise_method='peak2peak'
    )
    flags2 = np.zeros_like(u2, dtype=bool)
    u2, v2 = filters.replace_outliers(u2, v2, flags2, method='localmean', max_iter=3, kernel_size=3)

    # Third pass with 16x16 windows
    window_size = 16
    overlap = window_size // 2
    search_area_size = window_size

    print(f"Pass 3: window_size={window_size}, overlap={overlap}, search_area_size={search_area_size}")
    u3, v3, s2n = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        sig2noise_method='peak2peak'
    )
    flags3 = np.zeros_like(u3, dtype=bool)
    u3, v3 = filters.replace_outliers(u3, v3, flags3, method='localmean', max_iter=3, kernel_size=3)

    # Final pass with 8x8 windows
    window_size = 8
    overlap = window_size // 2
    search_area_size = window_size

    print(f"Pass 4: window_size={window_size}, overlap={overlap}, search_area_size={search_area_size}")
    u4, v4, s2n = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        sig2noise_method='peak2peak'
    )
    flags4 = np.zeros_like(u4, dtype=bool)
    u4, v4 = filters.replace_outliers(u4, v4, flags4, method='localmean', max_iter=3, kernel_size=3)

    # Use the final pass results
    u, v = u4, v4

    # Check for NaN or Inf values in the results
    if np.any(np.isnan(u)) or np.any(np.isinf(u)) or np.any(np.isnan(v)) or np.any(np.isinf(v)):
        print("Warning: PIV analysis produced NaN or Inf values. Replacing with zeros.")
        u = np.nan_to_num(u, nan=0, posinf=0, neginf=0)
        v = np.nan_to_num(v, nan=0, posinf=0, neginf=0)

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

    return x, y, u, v

def interpolate_to_full_grid(x, y, u, v, shape):
    """
    Interpolate the vector field to a full grid.

    Args:
        x, y: Coordinates of the vectors
        u, v: Velocity components
        shape: Shape of the output grid

    Returns:
        U, V: Interpolated velocity components on a full grid
    """
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
        # Interpolate u and v components with linear method
        U = griddata(points, u_flat, (Y, X), method='linear', fill_value=0)
        V = griddata(points, v_flat, (Y, X), method='linear', fill_value=0)

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

def save_results_to_mat(x, y, u, v, U, V, window_size, overlap, filename):
    """
    Save PIV results to a .mat file.

    Args:
        x, y: Coordinates of the vectors
        u, v: Velocity components (sparse grid)
        U, V: Interpolated velocity components (full grid)
        window_size, overlap: Final PIV parameters
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
        'multipass': True
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

    # Analyze with multipass PIV
    x, y, u, v = analyze_with_multipass_piv(frame_a, frame_b)

    # Interpolate to full grid
    U, V = interpolate_to_full_grid(x, y, u, v, frame_a.shape)

    # Save results to .mat file
    save_results_to_mat(
        x, y, u, v, U, V,
        window_size=8, overlap=4,
        filename=os.path.join(output_dir, 'piv_multipass_results.mat')
    )

if __name__ == "__main__":
    main()
