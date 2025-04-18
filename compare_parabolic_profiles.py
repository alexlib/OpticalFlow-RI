#!/usr/bin/env python
"""
Compare parabolic profiles from different optical flow methods.

This script runs all the optimized optical flow methods on the parabolic test images
and compares the resulting horizontal velocity profiles to verify they are parabolic.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.optimize import curve_fit
from scipy.io import savemat
import time
from tqdm import tqdm

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the optimized implementations
from denseLucasKanade_Numba_optimized import denseLucasKanade_Numba
from Farneback_Numba_optimized import Farneback_Numba
from HornSchunck_optimized import HSOpticalFlowAlgoAdapter
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

def compare_profiles(profiles, titles, output_filename):
    """Compare vertical velocity profiles from different methods."""
    plt.figure(figsize=(12, 8))
    
    for i, (x, v, popt, r_squared) in enumerate(profiles):
        # Generate fitted curve
        x_fit = np.linspace(x.min(), x.max(), 1000)
        v_fit = parabolic_function(x_fit, *popt)
        
        # Plot the data and fit
        plt.plot(x, v, '.', alpha=0.3, label=f'{titles[i]} (data)')
        plt.plot(x_fit, v_fit, '-', 
                 label=f'{titles[i]}: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f} (R²={r_squared:.4f})')
    
    plt.title('Comparison of Vertical Velocity Profiles')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def run_algorithm(algo, algo_name, im1, im2, output_dir, common_params):
    """Run an optical flow algorithm and analyze the results."""
    print(f"\nRunning {algo_name}...")
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Compute optical flow
    start_time = time.time()
    
    # Show progress bar
    with tqdm(total=100, desc=f"Computing {algo_name}", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        if hasattr(algo, 'compute'):
            # For class-based algorithms
            if algo_name == 'Liu-Shen':
                # Liu-Shen has a different parameter order
                V_out, U_out, _ = algo.compute(im1, im2, U, V)
            else:
                U_out, V_out, _ = algo.compute(im1, im2, U, V)
        else:
            # For GenericPyramidalOpticalFlow
            U_out, V_out = genericPyramidalOpticalFlow(
                im1, im2, common_params['FILTER'], algo, common_params['pyramidalLevels'], 
                common_params['kLevels'], common_params['FILTER_OPT'], None, 
                warping=common_params.get('warping', False)
            )
        pbar.update(100)  # Update to 100% when done
    
    elapsed_time = time.time() - start_time
    print(f"{algo_name} completed in {elapsed_time:.2f} seconds")
    
    # Save the flow field
    save_flow(U_out, V_out, os.path.join(output_dir, f'{algo_name.lower().replace(" ", "_")}.mat'))
    
    # Plot the flow field
    plot_flow_field(U_out, V_out, algo_name, 
                   os.path.join(output_dir, f'{algo_name.lower().replace(" ", "_")}_flow.png'))
    
    # Extract horizontal profile at the middle of the image
    x, u, v = extract_horizontal_profile(U_out, V_out)
    
    # Analyze horizontal profile
    popt, r_squared = analyze_horizontal_profile(x, u, v, algo_name,
                                               os.path.join(output_dir, f'{algo_name.lower().replace(" ", "_")}_profile.png'))
    
    return x, v, popt, r_squared

def main():
    # Load test images
    im1, im2 = load_parabolic_images()
    
    # Create output directory
    output_dir = 'parabolic_profile_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # Common parameters for all algorithms
    common_params = {
        'FILTER': 2,
        'FILTER_OPT': 0.48,
        'pyramidalLevels': 2,
        'kLevels': 1,
        'warping': False
    }
    
    # Define algorithms to run
    algorithms = [
        (denseLucasKanade_Numba(Niter=5, halfWindow=13), "Dense Lucas-Kanade"),
        (Farneback_Numba(windowSize=33, Niters=5, polyN=7, polySigma=1.5), "Farneback"),
        (HSOpticalFlowAlgoAdapter(alphas=[0.1], Niter=100), "Horn-Schunck"),
        (LiuShenOpticalFlowAlgoAdapter(alpha=0.1), "Liu-Shen")
    ]
    
    # Run all algorithms and collect profiles
    profiles = []
    titles = []
    
    for algo, algo_name in algorithms:
        x, v, popt, r_squared = run_algorithm(algo, algo_name, im1, im2, output_dir, common_params)
        profiles.append((x, v, popt, r_squared))
        titles.append(algo_name)
    
    # Compare profiles
    compare_profiles(profiles, titles, os.path.join(output_dir, 'profile_comparison.png'))
    
    # Print summary of results
    print("\nSummary of Parabolic Fits:")
    print("-" * 80)
    print(f"{'Algorithm':<20} {'a (x²)':<15} {'b (x)':<15} {'c (const)':<15} {'R²':<10}")
    print("-" * 80)
    
    for i, (_, _, popt, r_squared) in enumerate(profiles):
        print(f"{titles[i]:<20} {popt[0]:<15.6f} {popt[1]:<15.6f} {popt[2]:<15.6f} {r_squared:<10.4f}")
    
    print("-" * 80)
    print("\nResults saved to", output_dir)

if __name__ == "__main__":
    main()
