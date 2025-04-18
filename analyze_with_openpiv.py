#!/usr/bin/env python
"""
Analyze parabolic images using OpenPIV and use the results as a reference.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.optimize import curve_fit
import time
from tqdm import tqdm

# Import OpenPIV
from openpiv import tools, pyprocess as process, validation, filters

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

def analyze_with_openpiv(frame_a, frame_b, window_size=32, overlap=16, dt=1.0, search_area_size=64):
    """
    Analyze images using OpenPIV.

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
    print(f"Analyzing with OpenPIV (window_size={window_size}, overlap={overlap}, search_area_size={search_area_size})...")

    # Process the images to get the vector field
    u, v, sig2noise = process.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=window_size,
        overlap=overlap,
        dt=dt,
        search_area_size=search_area_size,
        sig2noise_method='peak2peak'
    )

    # Get the coordinates of the vectors
    # Calculate the number of windows in each dimension
    n_rows, n_cols = u.shape

    # Calculate the coordinates
    x = np.arange(window_size//2, frame_a.shape[1]-window_size//2+1, window_size-overlap)
    y = np.arange(window_size//2, frame_a.shape[0]-window_size//2+1, window_size-overlap)

    # Create meshgrid
    x, y = np.meshgrid(x[:n_cols], y[:n_rows])

    # Filter the vectors based on signal to noise ratio
    u, v, mask = validation.sig2noise_val(u, v, sig2noise, 1.3)

    # Replace outliers with local median
    u, v = filters.replace_outliers(
        u, v,
        method='localmean',
        max_iter=10,
        kernel_size=3
    )

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
    from scipy.interpolate import griddata

    # Create a meshgrid for the full image
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Flatten the coordinates and velocity components
    points = np.column_stack((y.flatten(), x.flatten()))

    # Interpolate u and v components
    U = griddata(points, u.flatten(), (Y, X), method='cubic', fill_value=0)
    V = griddata(points, v.flatten(), (Y, X), method='cubic', fill_value=0)

    return U, V

def plot_vector_field(x, y, u, v, title, output_filename, scale=1.0):
    """Plot the vector field."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the vector field
    ax.quiver(x, y, u, v, scale=scale, color='b', width=0.002)

    ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_xlim(0, x.max())
    ax.set_ylim(y.max(), 0)  # Invert y-axis to match image coordinates
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

def plot_flow_field(U, V, title, output_filename):
    """Plot optical flow as colormesh and quiver plots."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Auto-calculate color limits for vertical velocity
    v_abs_max = max(abs(np.percentile(V, 1)), abs(np.percentile(V, 99)))
    vmin = -v_abs_max
    vmax = v_abs_max

    # Plot vertical velocity component as colormesh with high contrast colormap
    im1 = ax1.imshow(V, cmap='jet', vmin=vmin, vmax=vmax)
    ax1.set_title(f'{title} - Vertical Velocity (v)')
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

    ax2.set_title(f'{title} - Vector Field')
    ax2.set_xlim(0, U.shape[1])
    ax2.set_ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    ax2.grid(True, alpha=0.3)

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

def parabolic_function(x, a, b, c):
    """Parabolic function: f(x) = a*x^2 + b*x + c"""
    return a * x**2 + b * x + c

def fit_parabola(x, y):
    """Fit a parabola to the data."""
    # Initial guess for parameters
    p0 = [-1e-3, 0, 0]

    # Fit the parabola
    popt, pcov = curve_fit(parabolic_function, x, y, p0=p0)

    # Calculate R-squared
    residuals = y - parabolic_function(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return popt, r_squared

def analyze_horizontal_profile(x, u, v, title, output_filename):
    """Analyze and plot the horizontal profile with parabolic fit."""
    # Fit a parabola to the vertical velocity profile
    popt_v, r_squared_v = fit_parabola(x, v)

    # Generate fitted curve
    x_fit = np.linspace(x.min(), x.max(), 1000)
    v_fit = parabolic_function(x_fit, *popt_v)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot horizontal velocity profile
    ax1.plot(x, u, 'b-', linewidth=2)
    ax1.set_title('Horizontal Velocity Profile')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Horizontal Velocity (pixels/frame)')
    ax1.grid(True)

    # Plot vertical velocity profile with parabolic fit
    ax2.plot(x, v, 'r.', label='Data')
    ax2.plot(x_fit, v_fit, 'g-', linewidth=2,
             label=f'Parabolic Fit: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}')
    ax2.set_title(f'Vertical Velocity Profile with Parabolic Fit (R² = {r_squared_v:.4f})')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Vertical Velocity (pixels/frame)')
    ax2.grid(True)
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

    return popt_v, r_squared_v

def main():
    # Load test images
    frame_a, frame_b = load_parabolic_images()

    # Create output directory
    output_dir = 'openpiv_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Try different window sizes
    window_sizes = [16, 32, 64]
    overlap_ratios = [0.5, 0.75]  # Overlap as a fraction of window size
    search_area_sizes = [32, 64, 128]

    best_r_squared = 0
    best_params = None
    best_results = None

    # Try different parameter combinations
    for window_size in window_sizes:
        for overlap_ratio in overlap_ratios:
            for search_area_size in search_area_sizes:
                # Calculate overlap in pixels
                overlap = int(window_size * overlap_ratio)

                # Skip if search area is smaller than window
                if search_area_size < window_size:
                    continue

                try:
                    # Analyze with OpenPIV
                    x, y, u, v = analyze_with_openpiv(
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

                    print(f"Window size: {window_size}, Overlap: {overlap}, Search area: {search_area_size}, R²: {r_squared:.4f}")

                    # Save if this is the best fit so far
                    if r_squared > best_r_squared:
                        best_r_squared = r_squared
                        best_params = (window_size, overlap, search_area_size)
                        best_results = (x, y, u, v, U, V)

                except Exception as e:
                    print(f"Error with window_size={window_size}, overlap={overlap}, search_area_size={search_area_size}: {e}")

    # Use the best parameters
    if best_params is not None:
        window_size, overlap, search_area_size = best_params
        x, y, u, v, U, V = best_results

        print(f"\nBest parameters: window_size={window_size}, overlap={overlap}, search_area_size={search_area_size}, R²={best_r_squared:.4f}")

        # Plot the vector field
        plot_vector_field(x, y, u, v, f"OpenPIV Vector Field (window={window_size}, overlap={overlap}, search={search_area_size})",
                         os.path.join(output_dir, 'openpiv_vectors.png'))

        # Plot the flow field
        plot_flow_field(U, V, f"OpenPIV Flow Field (window={window_size}, overlap={overlap}, search={search_area_size})",
                       os.path.join(output_dir, 'openpiv_flow.png'))

        # Analyze horizontal profile
        profile_x, profile_u, profile_v = extract_horizontal_profile(U, V)
        popt, r_squared = analyze_horizontal_profile(profile_x, profile_u, profile_v,
                                                   f"OpenPIV (window={window_size}, overlap={overlap}, search={search_area_size})",
                                                   os.path.join(output_dir, 'openpiv_profile.png'))

        print(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}, R²={r_squared:.4f}")

        # Save the best parameters to a file for reference
        with open(os.path.join(output_dir, 'best_parameters.txt'), 'w') as f:
            f.write(f"Window size: {window_size}\n")
            f.write(f"Overlap: {overlap}\n")
            f.write(f"Search area size: {search_area_size}\n")
            f.write(f"R-squared: {best_r_squared:.4f}\n")
            f.write(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}\n")

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
