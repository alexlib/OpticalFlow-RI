#!/usr/bin/env python
"""
Utility functions for optical flow analysis.

This module contains common functions used across different optical flow scripts
to minimize code duplication and promote reuse.
"""

import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
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
                     scale=25, scale_units='inches',
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

def parabolic_function(x, a, b, c):
    """Parabolic function: f(x) = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c

def fit_parabola(x, y):
    """Fit a parabola to the data."""
    # Check for NaN or Inf values
    valid_indices = ~(np.isnan(y) | np.isinf(y))
    if not np.all(valid_indices):
        print(f"Warning: Found {np.sum(~valid_indices)} invalid values in data. Removing them before fitting.")
        x = x[valid_indices]
        y = y[valid_indices]

    if len(x) < 3:
        print("Error: Not enough valid data points to fit a parabola.")
        return [0, 0, 0], 0.0

    # Initial guess for parameters
    p0 = [-1e-3, 0, 0]

    try:
        # Fit the parabola with bounds to prevent extreme values
        popt, _ = curve_fit(parabolic_function, x, y, p0=p0,
                           bounds=([-1, -10, -10], [1, 10, 10]))

        # Calculate R-squared
        residuals = y - parabolic_function(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Check for invalid R-squared (can happen with poor fits)
        if np.isnan(r_squared) or np.isinf(r_squared):
            print("Warning: Invalid R-squared value. Setting to 0.")
            r_squared = 0.0

    except Exception as e:
        print(f"Error fitting parabola: {e}")
        popt = [0, 0, 0]
        r_squared = 0.0

    return popt, r_squared

def analyze_profile(U, V, title=None, output_file=None):
    """
    Analyze flow profile and fit parabola.

    Args:
        U, V: Flow components
        title: Title for plots
        output_file: Output file for plots

    Returns:
        popt_u, r_squared_u: Parabolic fit parameters for U
        popt_v, r_squared_v: Parabolic fit parameters for V
    """
    # Calculate average horizontal velocity profile (average over all columns)
    y = np.arange(U.shape[0])  # Vertical coordinates
    u_avg_y = np.mean(U, axis=1)  # Average u for each y
    v_avg_y = np.mean(V, axis=1)  # Average v for each y

    # Fit parabolas to the profiles
    popt_u, r_squared_u = fit_parabola(y, u_avg_y)
    popt_v, r_squared_v = fit_parabola(y, v_avg_y)

    if title or output_file:
        # Create figure with two subplots
        plt.figure(figsize=(12, 10))

        # Generate fitted y coordinates for plotting
        y_fit = np.linspace(y.min(), y.max(), 1000)

        plt.subplot(2, 1, 1)
        # Plot y on the vertical axis and u_avg_y on the horizontal axis
        plt.plot(y, u_avg_y, 'b.', label='Data')
        # For plotting, we need to calculate the fit values for the velocity
        # based on the y coordinates
        u_fit_values = parabolic_function(y_fit, *popt_u)
        plt.plot(y_fit, u_fit_values, 'r-', linewidth=2,
                 label=f'Parabolic Fit: {popt_u[0]:.6f}y² + {popt_u[1]:.6f}y + {popt_u[2]:.6f}')
        plt.title(f'Horizontal Velocity Profile (Y) with Parabolic Fit (R² = {r_squared_u:.4f})')
        plt.xlabel('Y (pixels)')
        plt.ylabel('Horizontal Velocity (pixels/frame)')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        # Plot y on the vertical axis and v_avg_y on the horizontal axis
        plt.plot(y, v_avg_y, 'g.', label='Data')
        # Calculate the fit values for the vertical velocity
        v_fit_values = parabolic_function(y_fit, *popt_v)
        plt.plot(y_fit, v_fit_values, 'm-', linewidth=2,
                 label=f'Parabolic Fit: {popt_v[0]:.6f}y² + {popt_v[1]:.6f}y + {popt_v[2]:.6f}')
        plt.title(f'Vertical Velocity Profile (Y) with Parabolic Fit (R² = {r_squared_v:.4f})')
        plt.xlabel('Y (pixels)')
        plt.ylabel('Vertical Velocity (pixels/frame)')
        plt.grid(True)
        plt.legend()

        if title:
            plt.suptitle(title)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=200)
            print(f"Profile analysis saved to {output_file}")

        plt.close()

    # Print fit parameters
    print("\nHorizontal Velocity Parabolic Fit (Y-Profile):")
    print(f"Equation: {popt_u[0]:.6f}y² + {popt_u[1]:.6f}y + {popt_u[2]:.6f}")
    print(f"R-squared: {r_squared_u:.4f}")

    print("\nVertical Velocity Profile (Y-Profile):")
    print(f"Equation: {popt_v[0]:.6f}y² + {popt_v[1]:.6f}y + {popt_v[2]:.6f}")
    print(f"R-squared: {r_squared_v:.4f}")

    # Also calculate X profiles for completeness
    x = np.arange(U.shape[1])  # Horizontal coordinates
    u_avg_x = np.mean(U, axis=0)  # Average u for each x
    v_avg_x = np.mean(V, axis=0)  # Average v for each x

    # Fit parabolas to the X profiles
    popt_u_x, r_squared_u_x = fit_parabola(x, u_avg_x)
    popt_v_x, r_squared_v_x = fit_parabola(x, v_avg_x)

    # Print X profile fit parameters
    print("\nHorizontal Velocity Parabolic Fit (X-Profile):")
    print(f"Equation: {popt_u_x[0]:.6f}x² + {popt_u_x[1]:.6f}x + {popt_u_x[2]:.6f}")
    print(f"R-squared: {r_squared_u_x:.4f}")

    print("\nVertical Velocity Profile (X-Profile):")
    print(f"Equation: {popt_v_x[0]:.6f}x² + {popt_v_x[1]:.6f}x + {popt_v_x[2]:.6f}")
    print(f"R-squared: {r_squared_v_x:.4f}")

    return (popt_u, r_squared_u), (popt_v, r_squared_v)

def compute_block_matching_flow(im1, im2, window_size, overlap=None):
    """
    Compute flow field using block matching with overlap.

    Args:
        im1, im2: Input images
        window_size: Size of the matching window
        overlap: Overlap between windows (default: window_size//2)

    Returns:
        u, v: Flow components on grid
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

def save_flow_results(u, v, x, y, U, V, params, filename):
    """
    Save flow results to a .mat file.

    Args:
        u, v: Flow components on grid
        x, y: Grid coordinates
        U, V: Interpolated flow components
        params: Dictionary of parameters
        filename: Output filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create results dictionary
    results = {
        'x': x,
        'y': y,
        'u': u,
        'v': v,
        'U': U,
        'V': V
    }

    # Add parameters
    for key, value in params.items():
        if isinstance(value, (int, float)):
            results[key] = np.array([[value]])
        else:
            results[key] = value

    # Save to .mat file
    savemat(filename, results)
    print(f"Results saved to {filename}")
