#!/usr/bin/env python
"""
Optimize Farneback with Liu-Shen enhancement to match OpenPIV reference results.

This script runs the Farneback algorithm with Liu-Shen enhancement on the parabolic test images
and adjusts parameters to match the OpenPIV reference results.
"""

import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io import savemat, loadmat
import time
from tqdm import tqdm

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the optimized implementations
from Farneback_Numba_optimized import Farneback_Numba
from PhysicsBasedOpticalFlowLiuShen_optimized import LiuShenOpticalFlowAlgoAdapter
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow

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

def save_flow(U, V, filename):
    """Save flow field to a .mat file."""
    margins = {'top': 0, 'left': 0, 'bottom': 0, 'right': 0}
    results = {'u': U, 'v': V, 'iaWidth': 1, 'iaHeight': 1, 'margins': margins}
    parameters = {'overlapFactor': 1.0, 'imageHeight': np.size(U, 0), 'imageWidth': np.size(U, 1)}

    savemat(filename, mdict={'velocities': results, 'parameters': parameters})
    print(f"Flow saved to {filename}")

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

def plot_flow_field(U, V, title, output_filename):
    """Plot optical flow as colormesh and quiver plots."""
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

def load_openpiv_reference():
    """Load the OpenPIV reference results."""
    reference_file = os.path.join('piv_results', 'piv_smoothed_results.mat')

    if not os.path.exists(reference_file):
        print(f"Error: Reference file {reference_file} not found.")
        return None

    try:
        data = loadmat(reference_file)
        return data
    except Exception as e:
        print(f"Error loading reference file: {e}")
        return None

def compare_with_reference(U, V, ref_U, ref_V, title, output_filename):
    """Compare the flow field with the reference."""
    # Create figure with two rows and two columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Auto-calculate color limits for horizontal velocity
    u_abs_max = max(abs(np.percentile(U, 1)), abs(np.percentile(U, 99)),
                    abs(np.percentile(ref_U, 1)), abs(np.percentile(ref_U, 99)))
    vmin = -u_abs_max
    vmax = u_abs_max

    # Plot horizontal velocity component for the current method
    im1 = axes[0, 0].imshow(U, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'{title} - Horizontal Velocity (u)')
    plt.colorbar(im1, ax=axes[0, 0], label='Pixels/frame')
    axes[0, 0].grid(False)

    # Plot horizontal velocity component for the reference
    im2 = axes[0, 1].imshow(ref_U, cmap='jet', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title('OpenPIV Reference - Horizontal Velocity (u)')
    plt.colorbar(im2, ax=axes[0, 1], label='Pixels/frame')
    axes[0, 1].grid(False)

    # Extract horizontal profiles
    x, u, v = extract_horizontal_profile(U, V)
    x_ref, u_ref, v_ref = extract_horizontal_profile(ref_U, ref_V)

    # Plot horizontal velocity profiles
    axes[1, 0].plot(x, u, 'b-', linewidth=2, label=title)
    axes[1, 0].plot(x_ref, u_ref, 'r--', linewidth=2, label='OpenPIV Reference')
    axes[1, 0].set_title('Horizontal Velocity Profile Comparison')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Horizontal Velocity (pixels/frame)')
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    # Plot vertical velocity profiles
    axes[1, 1].plot(x, v, 'b-', linewidth=2, label=title)
    axes[1, 1].plot(x_ref, v_ref, 'r--', linewidth=2, label='OpenPIV Reference')
    axes[1, 1].set_title('Vertical Velocity Profile Comparison')
    axes[1, 1].set_xlabel('X (pixels)')
    axes[1, 1].set_ylabel('Vertical Velocity (pixels/frame)')
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.suptitle('Comparison with OpenPIV Reference')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()

def run_farneback_liushen(im1, im2, window_size, iterations, poly_n, poly_sigma, pyr_levels, liu_shen_alpha, output_dir):
    """
    Run Farneback with Liu-Shen enhancement.

    Args:
        im1, im2: Input images
        window_size: Window size for Farneback
        iterations: Number of iterations for Farneback
        poly_n: Polynomial degree for Farneback
        poly_sigma: Polynomial sigma for Farneback
        pyr_levels: Number of pyramid levels
        liu_shen_alpha: Alpha parameter for Liu-Shen
        output_dir: Output directory

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

    # Create Liu-Shen adapter
    ls_adapter = LiuShenOpticalFlowAlgoAdapter(alpha=liu_shen_alpha)

    # Common parameters
    FILTER = 2
    FILTER_OPT = 0.48
    kLevels = 1

    # Run the algorithm
    print(f"\nRunning Farneback with Liu-Shen enhancement:")
    print(f"  Window size: {window_size}")
    print(f"  Iterations: {iterations}")
    print(f"  Polynomial degree: {poly_n}")
    print(f"  Polynomial sigma: {poly_sigma}")
    print(f"  Pyramid levels: {pyr_levels}")
    print(f"  Liu-Shen alpha: {liu_shen_alpha}")

    start_time = time.time()

    # Show progress bar
    with tqdm(total=100, desc="Computing optical flow", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        # Run the algorithm
        U, V = genericPyramidalOpticalFlow(
            im1, im2, FILTER, fb_adapter,
            pyr_levels, kLevels,
            FILTER_OPT, ls_adapter,
            warping=True
        )

        # Update progress bar to 100%
        pbar.n = 100
        pbar.refresh()

    elapsed_time = time.time() - start_time
    print(f"Computation completed in {elapsed_time:.2f} seconds")

    # Generate parameter string for filenames
    param_str = f"w{window_size}_i{iterations}_n{poly_n}_s{poly_sigma}_p{pyr_levels}_a{liu_shen_alpha}"

    # Save the flow field
    save_flow(U, V, os.path.join(output_dir, f'farneback_liushen_{param_str}.mat'))

    # Plot the flow field
    plot_flow_field(U, V, f"Farneback with Liu-Shen ({param_str})",
                   os.path.join(output_dir, f'farneback_liushen_{param_str}_flow.png'))

    # Extract horizontal profile
    x, u, v = extract_horizontal_profile(U, V)

    # Analyze horizontal profile
    popt, r_squared = analyze_horizontal_profile(x, u, v,
                                               f"Farneback with Liu-Shen ({param_str})",
                                               os.path.join(output_dir, f'farneback_liushen_{param_str}_profile.png'))

    # Load reference data
    ref_data = load_openpiv_reference()
    if ref_data is not None:
        ref_U = ref_data['U']
        ref_V = ref_data['V']

        # Compare with reference
        compare_with_reference(U, V, ref_U, ref_V,
                              f"Farneback with Liu-Shen ({param_str})",
                              os.path.join(output_dir, f'farneback_liushen_{param_str}_comparison.png'))

    print(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}, R²={r_squared:.4f}")

    return U, V, popt, r_squared, param_str

def main():
    # Load test images
    im1, im2 = load_parabolic_images()

    # Create output directory
    output_dir = 'farneback_liushen_optimization'
    os.makedirs(output_dir, exist_ok=True)

    # Parameter combinations to try
    parameter_sets = [
        # Initial parameters based on common settings
        {"window_size": 33, "iterations": 5, "poly_n": 7, "poly_sigma": 1.5, "pyr_levels": 2, "liu_shen_alpha": 0.1},

        # Smaller window size to match OpenPIV (15px - must be odd)
        {"window_size": 15, "iterations": 5, "poly_n": 7, "poly_sigma": 1.5, "pyr_levels": 2, "liu_shen_alpha": 0.1},

        # Adjust polynomial parameters
        {"window_size": 15, "iterations": 5, "poly_n": 5, "poly_sigma": 1.2, "pyr_levels": 2, "liu_shen_alpha": 0.1},

        # Increase iterations
        {"window_size": 15, "iterations": 10, "poly_n": 5, "poly_sigma": 1.2, "pyr_levels": 2, "liu_shen_alpha": 0.1},

        # Adjust Liu-Shen alpha
        {"window_size": 15, "iterations": 10, "poly_n": 5, "poly_sigma": 1.2, "pyr_levels": 2, "liu_shen_alpha": 0.5},

        # Fine-tuned parameters
        {"window_size": 15, "iterations": 10, "poly_n": 5, "poly_sigma": 1.5, "pyr_levels": 2, "liu_shen_alpha": 0.5},
    ]

    # Run all parameter combinations
    results = []

    for params in parameter_sets:
        U, V, popt, r_squared, param_str = run_farneback_liushen(
            im1, im2,
            window_size=params["window_size"],
            iterations=params["iterations"],
            poly_n=params["poly_n"],
            poly_sigma=params["poly_sigma"],
            pyr_levels=params["pyr_levels"],
            liu_shen_alpha=params["liu_shen_alpha"],
            output_dir=output_dir
        )

        results.append({
            "params": params,
            "popt": popt,
            "r_squared": r_squared,
            "param_str": param_str
        })

    # Print summary of results
    print("\nSummary of Results:")
    print("-" * 100)
    print(f"{'Window':<8} {'Iter':<6} {'PolyN':<6} {'PolySig':<8} {'PyrLvl':<8} {'LSAlpha':<8} {'a (x²)':<12} {'b (x)':<12} {'c (const)':<12} {'R²':<8}")
    print("-" * 100)

    for result in results:
        params = result["params"]
        popt = result["popt"]
        r_squared = result["r_squared"]

        print(f"{params['window_size']:<8} {params['iterations']:<6} {params['poly_n']:<6} {params['poly_sigma']:<8.2f} {params['pyr_levels']:<8} {params['liu_shen_alpha']:<8.2f} {popt[0]:<12.6f} {popt[1]:<12.6f} {popt[2]:<12.6f} {r_squared:<8.4f}")

    print("-" * 100)
    print("\nResults saved to", output_dir)

if __name__ == "__main__":
    main()
