#!/usr/bin/env python
"""
Optimize optical flow methods based on OpenPIV results.

This script runs the optimized optical flow methods on the parabolic test images
and compares the results to the OpenPIV reference.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.optimize import curve_fit
import time
from tqdm import tqdm

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the optimized implementations
from denseLucasKanade_Numba_optimized import denseLucasKanade_Numba
from Farneback_Numba_optimized import Farneback_Numba
from HornSchunck_optimized import HSOpticalFlowAlgoAdapter
from PhysicsBasedOpticalFlowLiuShen_optimized import LiuShenOpticalFlowAlgoAdapter

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

def run_dense_lucas_kanade(im1, im2, output_dir):
    """Run the Dense Lucas-Kanade algorithm with optimized parameters."""
    print("\nRunning Dense Lucas-Kanade...")
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Try different parameters
    window_sizes = [13, 21, 33, 65]  # Similar to PIV window sizes
    iterations = [5, 10, 20]
    
    best_r_squared = 0
    best_params = None
    best_results = None
    
    for window_size in window_sizes:
        for niter in iterations:
            try:
                # Create the Dense Lucas-Kanade object
                lk = denseLucasKanade_Numba(Niter=niter, halfWindow=window_size//2)
                
                # Compute optical flow
                start_time = time.time()
                U_out, V_out, _ = lk.compute(im1, im2, U, V)
                elapsed_time = time.time() - start_time
                
                # Extract horizontal profile
                profile_x, profile_u, profile_v = extract_horizontal_profile(U_out, V_out)
                
                # Fit parabola
                popt, r_squared = fit_parabola(profile_x, profile_v)
                
                print(f"Window size: {window_size}, Iterations: {niter}, R²: {r_squared:.4f}, Time: {elapsed_time:.2f}s")
                
                # Save if this is the best fit so far
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_params = (window_size, niter)
                    best_results = (U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared)
                    
            except Exception as e:
                print(f"Error with window_size={window_size}, niter={niter}: {e}")
    
    # Use the best parameters
    if best_params is not None:
        window_size, niter = best_params
        U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared = best_results
        
        print(f"\nBest parameters: window_size={window_size}, iterations={niter}, R²={best_r_squared:.4f}")
        
        # Plot the flow field
        plot_flow_field(U_out, V_out, f"Dense Lucas-Kanade (window={window_size}, iterations={niter})",
                       os.path.join(output_dir, 'dense_lucas_kanade_flow.png'))
        
        # Analyze horizontal profile
        analyze_horizontal_profile(profile_x, profile_u, profile_v, 
                                 f"Dense Lucas-Kanade (window={window_size}, iterations={niter})",
                                 os.path.join(output_dir, 'dense_lucas_kanade_profile.png'))
        
        # Save the best parameters to a file for reference
        with open(os.path.join(output_dir, 'dense_lucas_kanade_parameters.txt'), 'w') as f:
            f.write(f"Window size: {window_size}\n")
            f.write(f"Iterations: {niter}\n")
            f.write(f"R-squared: {best_r_squared:.4f}\n")
            f.write(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}\n")
        
        return profile_x, profile_v, popt, r_squared
    
    return None

def run_farneback(im1, im2, output_dir):
    """Run the Farneback algorithm with optimized parameters."""
    print("\nRunning Farneback...")
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Try different parameters
    window_sizes = [15, 33, 65]
    iterations = [5, 10, 20]
    poly_n_values = [5, 7, 9]
    poly_sigma_values = [1.1, 1.5, 2.0]
    
    best_r_squared = 0
    best_params = None
    best_results = None
    
    for window_size in window_sizes:
        for niter in iterations:
            for poly_n in poly_n_values:
                for poly_sigma in poly_sigma_values:
                    try:
                        # Create the Farneback object
                        fb = Farneback_Numba(windowSize=window_size, Niters=niter, 
                                           polyN=poly_n, polySigma=poly_sigma)
                        
                        # Compute optical flow
                        start_time = time.time()
                        U_out, V_out, _ = fb.compute(im1, im2, U, V)
                        elapsed_time = time.time() - start_time
                        
                        # Extract horizontal profile
                        profile_x, profile_u, profile_v = extract_horizontal_profile(U_out, V_out)
                        
                        # Fit parabola
                        popt, r_squared = fit_parabola(profile_x, profile_v)
                        
                        print(f"Window: {window_size}, Iter: {niter}, PolyN: {poly_n}, PolySigma: {poly_sigma}, R²: {r_squared:.4f}, Time: {elapsed_time:.2f}s")
                        
                        # Save if this is the best fit so far
                        if r_squared > best_r_squared:
                            best_r_squared = r_squared
                            best_params = (window_size, niter, poly_n, poly_sigma)
                            best_results = (U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared)
                            
                    except Exception as e:
                        print(f"Error with window_size={window_size}, niter={niter}, poly_n={poly_n}, poly_sigma={poly_sigma}: {e}")
    
    # Use the best parameters
    if best_params is not None:
        window_size, niter, poly_n, poly_sigma = best_params
        U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared = best_results
        
        print(f"\nBest parameters: window={window_size}, iter={niter}, polyN={poly_n}, polySigma={poly_sigma}, R²={best_r_squared:.4f}")
        
        # Plot the flow field
        plot_flow_field(U_out, V_out, f"Farneback (window={window_size}, iter={niter}, polyN={poly_n}, polySigma={poly_sigma})",
                       os.path.join(output_dir, 'farneback_flow.png'))
        
        # Analyze horizontal profile
        analyze_horizontal_profile(profile_x, profile_u, profile_v, 
                                 f"Farneback (window={window_size}, iter={niter}, polyN={poly_n}, polySigma={poly_sigma})",
                                 os.path.join(output_dir, 'farneback_profile.png'))
        
        # Save the best parameters to a file for reference
        with open(os.path.join(output_dir, 'farneback_parameters.txt'), 'w') as f:
            f.write(f"Window size: {window_size}\n")
            f.write(f"Iterations: {niter}\n")
            f.write(f"Polynomial degree (polyN): {poly_n}\n")
            f.write(f"Polynomial sigma: {poly_sigma}\n")
            f.write(f"R-squared: {best_r_squared:.4f}\n")
            f.write(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}\n")
        
        return profile_x, profile_v, popt, r_squared
    
    return None

def run_horn_schunck(im1, im2, output_dir):
    """Run the Horn-Schunck algorithm with optimized parameters."""
    print("\nRunning Horn-Schunck...")
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Try different parameters
    alpha_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    iterations = [50, 100, 200, 500]
    
    best_r_squared = 0
    best_params = None
    best_results = None
    
    for alpha in alpha_values:
        for niter in iterations:
            try:
                # Create the Horn-Schunck object
                hs = HSOpticalFlowAlgoAdapter(alphas=[alpha], Niter=niter)
                
                # Compute optical flow
                start_time = time.time()
                U_out, V_out, _ = hs.compute(im1, im2, U, V)
                elapsed_time = time.time() - start_time
                
                # Extract horizontal profile
                profile_x, profile_u, profile_v = extract_horizontal_profile(U_out, V_out)
                
                # Fit parabola
                popt, r_squared = fit_parabola(profile_x, profile_v)
                
                print(f"Alpha: {alpha}, Iterations: {niter}, R²: {r_squared:.4f}, Time: {elapsed_time:.2f}s")
                
                # Save if this is the best fit so far
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_params = (alpha, niter)
                    best_results = (U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared)
                    
            except Exception as e:
                print(f"Error with alpha={alpha}, niter={niter}: {e}")
    
    # Use the best parameters
    if best_params is not None:
        alpha, niter = best_params
        U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared = best_results
        
        print(f"\nBest parameters: alpha={alpha}, iterations={niter}, R²={best_r_squared:.4f}")
        
        # Plot the flow field
        plot_flow_field(U_out, V_out, f"Horn-Schunck (alpha={alpha}, iterations={niter})",
                       os.path.join(output_dir, 'horn_schunck_flow.png'))
        
        # Analyze horizontal profile
        analyze_horizontal_profile(profile_x, profile_u, profile_v, 
                                 f"Horn-Schunck (alpha={alpha}, iterations={niter})",
                                 os.path.join(output_dir, 'horn_schunck_profile.png'))
        
        # Save the best parameters to a file for reference
        with open(os.path.join(output_dir, 'horn_schunck_parameters.txt'), 'w') as f:
            f.write(f"Alpha: {alpha}\n")
            f.write(f"Iterations: {niter}\n")
            f.write(f"R-squared: {best_r_squared:.4f}\n")
            f.write(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}\n")
        
        return profile_x, profile_v, popt, r_squared
    
    return None

def run_liu_shen(im1, im2, output_dir):
    """Run the Liu-Shen algorithm with optimized parameters."""
    print("\nRunning Liu-Shen...")
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Try different parameters
    alpha_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    best_r_squared = 0
    best_params = None
    best_results = None
    
    for alpha in alpha_values:
        try:
            # Create the Liu-Shen object
            ls = LiuShenOpticalFlowAlgoAdapter(alpha=alpha)
            
            # Compute optical flow
            start_time = time.time()
            U_out, V_out, _ = ls.compute(im1, im2, U, V)
            elapsed_time = time.time() - start_time
            
            # Extract horizontal profile
            profile_x, profile_u, profile_v = extract_horizontal_profile(U_out, V_out)
            
            # Fit parabola
            popt, r_squared = fit_parabola(profile_x, profile_v)
            
            print(f"Alpha: {alpha}, R²: {r_squared:.4f}, Time: {elapsed_time:.2f}s")
            
            # Save if this is the best fit so far
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_params = alpha
                best_results = (U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared)
                
        except Exception as e:
            print(f"Error with alpha={alpha}: {e}")
    
    # Use the best parameters
    if best_params is not None:
        alpha = best_params
        U_out, V_out, profile_x, profile_u, profile_v, popt, r_squared = best_results
        
        print(f"\nBest parameters: alpha={alpha}, R²={best_r_squared:.4f}")
        
        # Plot the flow field
        plot_flow_field(U_out, V_out, f"Liu-Shen (alpha={alpha})",
                       os.path.join(output_dir, 'liu_shen_flow.png'))
        
        # Analyze horizontal profile
        analyze_horizontal_profile(profile_x, profile_u, profile_v, 
                                 f"Liu-Shen (alpha={alpha})",
                                 os.path.join(output_dir, 'liu_shen_profile.png'))
        
        # Save the best parameters to a file for reference
        with open(os.path.join(output_dir, 'liu_shen_parameters.txt'), 'w') as f:
            f.write(f"Alpha: {alpha}\n")
            f.write(f"R-squared: {best_r_squared:.4f}\n")
            f.write(f"Parabolic fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}\n")
        
        return profile_x, profile_v, popt, r_squared
    
    return None

def main():
    # Load test images
    im1, im2 = load_parabolic_images()
    
    # Create output directory
    output_dir = 'optimized_flow_methods'
    os.makedirs(output_dir, exist_ok=True)
    
    # Run all methods and collect profiles
    profiles = []
    titles = []
    
    # Run Dense Lucas-Kanade
    result = run_dense_lucas_kanade(im1, im2, output_dir)
    if result is not None:
        profiles.append(result)
        titles.append('Dense Lucas-Kanade')
    
    # Run Farneback
    result = run_farneback(im1, im2, output_dir)
    if result is not None:
        profiles.append(result)
        titles.append('Farneback')
    
    # Run Horn-Schunck
    result = run_horn_schunck(im1, im2, output_dir)
    if result is not None:
        profiles.append(result)
        titles.append('Horn-Schunck')
    
    # Run Liu-Shen
    result = run_liu_shen(im1, im2, output_dir)
    if result is not None:
        profiles.append(result)
        titles.append('Liu-Shen')
    
    # Compare profiles
    if profiles:
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
