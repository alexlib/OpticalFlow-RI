#!/usr/bin/env python
"""
Simple PIV analysis of parabolic images using OpenPIV.
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

def analyze_with_simple_piv(frame_a, frame_b, window_size=32, overlap=16, dt=1.0, search_area_size=None):
    """
    Analyze images using OpenPIV's simple PIV function.

    Args:
        frame_a, frame_b: Input frames
        window_size: Size of the interrogation window
        overlap: Overlap between windows
        dt: Time step between frames
        search_area_size: Size of the search area (optional)

    Returns:
        x, y: Coordinates of the vectors
        u, v: Velocity components
    """
    print(f"Analyzing with Simple PIV (window_size={window_size}, overlap={overlap})...")

    # Make sure input images are valid
    frame_a = np.asarray(frame_a).astype(np.int32)
    frame_b = np.asarray(frame_b).astype(np.int32)

    # Check for NaN or Inf values in the input images
    if np.any(np.isnan(frame_a)) or np.any(np.isinf(frame_a)) or np.any(np.isnan(frame_b)) or np.any(np.isinf(frame_b)):
        print("Warning: Input images contain NaN or Inf values. Replacing with zeros.")
        frame_a = np.nan_to_num(frame_a, nan=0, posinf=0, neginf=0)
        frame_b = np.nan_to_num(frame_b, nan=0, posinf=0, neginf=0)

    try:
        # Process the images to get the vector field
        if search_area_size is not None:
            # Use extended search area PIV if search_area_size is provided
            u, v, s2n = pyprocess.extended_search_area_piv(
                frame_a, frame_b,
                window_size=window_size,
                overlap=overlap,
                dt=dt,
                search_area_size=search_area_size,
                sig2noise_method='peak2peak'
            )
        else:
            # Use simple PIV otherwise
            u, v, s2n = pyprocess.simple_piv(
                frame_a, frame_b,
                window_size=window_size,
                overlap=overlap,
                dt=dt
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

        return x, y, u, v, s2n

    except Exception as e:
        print(f"Error in PIV analysis: {e}")
        raise

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

    if len(x_flat) < 4:  # Need at least a few points for interpolation
        print("Error: Not enough valid data points for interpolation.")
        return np.zeros(shape), np.zeros(shape)

    # Create a meshgrid for the full image
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Flatten the coordinates and velocity components
    points = np.column_stack((y_flat, x_flat))

    try:
        # Interpolate u and v components with linear method first (more stable)
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

def plot_vector_field(x, y, u, v, title, output_filename, scale=1.0):
    """Plot the vector field."""
    plt.figure(figsize=(10, 8))

    # Plot the vector field
    plt.quiver(x, y, u, v, scale=scale, color='b', width=0.002)

    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.xlim(0, x.max())
    plt.ylim(y.max(), 0)  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

def plot_flow_field(U, V, title, output_filename):
    """Plot optical flow as colormesh and quiver plots."""
    # Create figure with two subplots
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)

    # Auto-calculate color limits for vertical velocity
    v_abs_max = max(abs(np.percentile(V, 1)), abs(np.percentile(V, 99)))
    vmin = -v_abs_max
    vmax = v_abs_max

    # Plot vertical velocity component as colormesh with high contrast colormap
    im1 = plt.imshow(V, cmap='jet', vmin=vmin, vmax=vmax)
    plt.title(f'{title} - Vertical Velocity (v)')
    plt.colorbar(im1, label='Pixels/frame')

    # Add a grid for better reference
    plt.grid(False)

    plt.subplot(1, 2, 2)

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

    plt.title(f'{title} - Vector Field')
    plt.xlim(0, U.shape[1])
    plt.ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

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

def analyze_horizontal_profile(x, u, v, title, output_filename):
    """Analyze and plot the horizontal profile with parabolic fit."""
    # Fit a parabola to the vertical velocity profile
    popt_v, r_squared_v = fit_parabola(x, v)

    # Generate fitted curve
    x_fit = np.linspace(x.min(), x.max(), 1000)
    v_fit = parabolic_function(x_fit, *popt_v)

    # Create figure with two subplots
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    # Plot horizontal velocity profile
    plt.plot(x, u, 'b-', linewidth=2)
    plt.title('Horizontal Velocity Profile')
    plt.xlabel('X (pixels)')
    plt.ylabel('Horizontal Velocity (pixels/frame)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    # Plot vertical velocity profile with parabolic fit
    plt.plot(x, v, 'r.', label='Data')
    plt.plot(x_fit, v_fit, 'g-', linewidth=2,
             label=f'Parabolic Fit: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}')
    plt.title(f'Vertical Velocity Profile with Parabolic Fit (R² = {r_squared_v:.4f})')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

    return popt_v, r_squared_v

def main():
    # Load test images
    frame_a, frame_b = load_parabolic_images()

    # Create output directory
    output_dir = 'simple_piv_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Try different window sizes
    window_sizes = [16, 32, 64]
    overlap_ratios = [0.5, 0.75]  # Overlap as a fraction of window size
    search_area_sizes = [None, 32, 64]  # None means use simple_piv instead of extended_search_area_piv

    best_r_squared = 0
    best_params = None
    best_results = None

    # Try different parameter combinations
    for window_size in window_sizes:
        for overlap_ratio in overlap_ratios:
            for search_area_size in search_area_sizes:
                # Skip invalid combinations
                if search_area_size is not None and search_area_size < window_size:
                    continue

                # Calculate overlap in pixels
                overlap = int(window_size * overlap_ratio)

                try:
                    # Analyze with Simple PIV
                    x, y, u, v, _ = analyze_with_simple_piv(
                        frame_a, frame_b,
                        window_size=window_size,
                        overlap=overlap,
                        search_area_size=search_area_size
                    )

                    # Interpolate to full grid
                    U, V = interpolate_to_full_grid(x, y, u, v, frame_a.shape)

                    # Extract horizontal profile
                    profile_x, profile_u, profile_v = extract_horizontal_profile(U, V)

                    # Fit parabola
                    popt, r_squared = fit_parabola(profile_x, profile_v)

                    search_area_str = f", Search area: {search_area_size}" if search_area_size else ""
                    print(f"Window size: {window_size}, Overlap: {overlap}{search_area_str}, R²: {r_squared:.4f}")

                    # Save if this is the best fit so far
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_params = (window_size, overlap, search_area_size)
                        best_results = (x, y, u, v, U, V)

                except Exception as e:
                    search_area_str = f", search_area={search_area_size}" if search_area_size else ""
                    print(f"Error with window_size={window_size}, overlap={overlap}{search_area_str}: {e}")

    # Use the best parameters
    if best_params is not None:
        window_size, overlap, search_area_size = best_params
        x, y, u, v, U, V = best_results

        search_area_str = f", search_area={search_area_size}" if search_area_size else ""
        print(f"\nBest parameters: window_size={window_size}, overlap={overlap}{search_area_str}, R²={best_r_squared:.4f}")

        # Plot the vector field
        title_suffix = f" (window={window_size}, overlap={overlap}{search_area_str})"
        plot_vector_field(x, y, u, v, f"Simple PIV Vector Field{title_suffix}",
                         os.path.join(output_dir, 'simple_piv_vectors.png'))

        # Plot the flow field
        plot_flow_field(U, V, f"Simple PIV Flow Field{title_suffix}",
                       os.path.join(output_dir, 'simple_piv_flow.png'))

        # Analyze horizontal profile
        profile_x, profile_u, profile_v = extract_horizontal_profile(U, V)
        popt, r_squared = analyze_horizontal_profile(profile_x, profile_u, profile_v,
                                                   f"Simple PIV{title_suffix}",
                                                   os.path.join(output_dir, 'simple_piv_profile.png'))

        print(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}, R²={r_squared:.4f}")

        # Save the best parameters to a file for reference
        with open(os.path.join(output_dir, 'best_parameters.txt'), 'w') as f:
            f.write(f"Window size: {window_size}\n")
            f.write(f"Overlap: {overlap}\n")
            if search_area_size:
                f.write(f"Search area size: {search_area_size}\n")
            f.write(f"R-squared: {best_r_squared:.4f}\n")
            f.write(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}\n")

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
