#!/usr/bin/env python
"""
Display the horizontal velocity component from the OpenPIV results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import time

# Import OpenPIV
from openpiv import tools, pyprocess

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

def analyze_with_simple_piv(frame_a, frame_b, window_size=64, overlap=32, dt=1.0, search_area_size=64):
    """
    Analyze images using OpenPIV's extended search area PIV function.
    
    Args:
        frame_a, frame_b: Input frames
        window_size: Size of the interrogation window
        overlap: Overlap between windows
        dt: Time step between frames
        search_area_size: Size of the search area
        
    Returns:
        x, y: Coordinates of the vectors
        u, v: Velocity components
    """
    print(f"Analyzing with PIV (window_size={window_size}, overlap={overlap}, search_area_size={search_area_size})...")
    
    # Make sure input images are valid
    frame_a = np.asarray(frame_a).astype(np.int32)
    frame_b = np.asarray(frame_b).astype(np.int32)
    
    # Process the images to get the vector field
    u, v, s2n = pyprocess.extended_search_area_piv(
        frame_a, frame_b,
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        sig2noise_method='peak2peak'
    )
    
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

def plot_flow_field(U, V, title, output_filename=None):
    """Plot optical flow with horizontal velocity in the colormap."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Auto-calculate color limits for horizontal velocity
    u_abs_max = max(abs(np.percentile(U, 1)), abs(np.percentile(U, 99)))
    vmin = -u_abs_max
    vmax = u_abs_max
    
    # Plot horizontal velocity component as colormesh with high contrast colormap
    im1 = ax1.imshow(U, cmap='jet', vmin=vmin, vmax=vmax)
    ax1.set_title(f'{title} - Horizontal Velocity (u)')
    plt.colorbar(im1, ax=ax1, label='Pixels/frame')
    
    # Add a grid for better reference
    ax1.grid(False)
    
    # Create quiver plot with color-coded arrows based on magnitude
    quiver_skip = 1  # Show all vectors
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
    
    ax2.set_title(f'{title} - Vector Field')
    ax2.set_xlim(0, U.shape[1])
    ax2.set_ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=200)
    
    plt.show()

def extract_horizontal_profile(U, V, position=None):
    """
    Extract a horizontal profile from the flow field.
    
    Args:
        U, V: Horizontal and vertical velocity components
        position: Position along the vertical axis (default: middle)
        
    Returns:
        x: Position along the horizontal axis
        u: Horizontal velocity profile
        v: Vertical velocity profile
    """
    if position is None:
        position = U.shape[0] // 2
    
    x = np.arange(U.shape[1])
    u = U[position, :]
    v = V[position, :]
    
    return x, u, v

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

def analyze_horizontal_profile(x, u, v, title, output_filename=None):
    """Analyze and plot the horizontal and vertical velocity profiles with parabolic fits."""
    # Fit parabolas to both horizontal and vertical velocity profiles
    popt_u, r_squared_u = fit_parabola(x, u)
    popt_v, r_squared_v = fit_parabola(x, v)
    
    # Generate fitted curves
    x_fit = np.linspace(x.min(), x.max(), 1000)
    u_fit = parabolic_function(x_fit, *popt_u)
    v_fit = parabolic_function(x_fit, *popt_v)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot horizontal velocity profile with parabolic fit
    ax1.plot(x, u, 'b.', label='Data')
    ax1.plot(x_fit, u_fit, 'r-', linewidth=2, 
             label=f'Parabolic Fit: {popt_u[0]:.6f}x² + {popt_u[1]:.6f}x + {popt_u[2]:.6f}')
    ax1.set_title(f'Horizontal Velocity Profile with Parabolic Fit (R² = {r_squared_u:.4f})')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Horizontal Velocity (pixels/frame)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot vertical velocity profile with parabolic fit
    ax2.plot(x, v, 'g.', label='Data')
    ax2.plot(x_fit, v_fit, 'm-', linewidth=2, 
             label=f'Parabolic Fit: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}')
    ax2.set_title(f'Vertical Velocity Profile with Parabolic Fit (R² = {r_squared_v:.4f})')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Vertical Velocity (pixels/frame)')
    ax2.grid(True)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_filename:
        plt.savefig(output_filename, dpi=200)
    
    plt.show()
    
    return popt_u, r_squared_u, popt_v, r_squared_v

def main():
    # Load test images
    frame_a, frame_b = load_parabolic_images()
    
    # Create output directory
    output_dir = 'horizontal_velocity_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the best parameters from previous analysis
    window_size = 64
    overlap = 32
    search_area_size = 64
    
    # Analyze with PIV
    x, y, u, v = analyze_with_simple_piv(
        frame_a, frame_b, 
        window_size=window_size, 
        overlap=overlap,
        search_area_size=search_area_size
    )
    
    # Interpolate to full grid
    U, V = interpolate_to_full_grid(x, y, u, v, frame_a.shape)
    
    # Plot the flow field with horizontal velocity in the colormap
    plot_flow_field(U, V, f"PIV Flow Field (window={window_size}, overlap={overlap}, search={search_area_size})",
                   os.path.join(output_dir, 'horizontal_velocity_flow.png'))
    
    # Extract horizontal profile
    profile_x, profile_u, profile_v = extract_horizontal_profile(U, V)
    
    # Analyze horizontal profile
    popt_u, r_squared_u, popt_v, r_squared_v = analyze_horizontal_profile(
        profile_x, profile_u, profile_v, 
        f"PIV Velocity Profiles (window={window_size}, overlap={overlap}, search={search_area_size})",
        os.path.join(output_dir, 'horizontal_velocity_profile.png')
    )
    
    print("\nHorizontal Velocity Parabolic Fit:")
    print(f"Equation: {popt_u[0]:.6f}x² + {popt_u[1]:.6f}x + {popt_u[2]:.6f}")
    print(f"R-squared: {r_squared_u:.4f}")
    
    print("\nVertical Velocity Parabolic Fit:")
    print(f"Equation: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}")
    print(f"R-squared: {r_squared_v:.4f}")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
