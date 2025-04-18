#!/usr/bin/env python
"""
Analyze the horizontal velocity and horizontal profile from the optical flow results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit

def load_flow_field(filename):
    """Load flow field from a .mat file."""
    data = loadmat(filename)
    U = data['velocities']['u'][0, 0]
    V = data['velocities']['v'][0, 0]
    return U, V

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

def plot_horizontal_velocity(U, output_filename):
    """Plot the horizontal velocity component."""
    plt.figure(figsize=(10, 8))
    
    # Auto-calculate color limits
    u_abs_max = max(abs(np.percentile(U, 1)), abs(np.percentile(U, 99)))
    vmin = -u_abs_max
    vmax = u_abs_max
    
    # Plot horizontal velocity component as colormesh with high contrast colormap
    im = plt.imshow(U, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Horizontal Velocity (pixels/frame)')
    plt.title('Horizontal Velocity (U)')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_horizontal_profiles(x, u, v, output_filename):
    """Plot horizontal profiles of both velocity components."""
    plt.figure(figsize=(12, 6))
    
    # Plot horizontal velocity profile
    plt.subplot(1, 2, 1)
    plt.plot(x, u, 'b-', linewidth=2)
    plt.title('Horizontal Velocity Profile')
    plt.xlabel('X (pixels)')
    plt.ylabel('Horizontal Velocity (pixels/frame)')
    plt.grid(True)
    
    # Plot vertical velocity profile
    plt.subplot(1, 2, 2)
    plt.plot(x, v, 'r-', linewidth=2)
    plt.title('Vertical Velocity Profile')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()

def analyze_horizontal_profile(x, u, v, output_filename):
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
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    return popt_v, r_squared_v

def main():
    # Load flow field
    flow_file = 'denseLK_numba_optimized_results/denseLK_numba_optimized.mat'
    U, V = load_flow_field(flow_file)
    
    # Create output directory
    output_dir = 'horizontal_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot horizontal velocity
    plot_horizontal_velocity(U, os.path.join(output_dir, 'horizontal_velocity.png'))
    
    # Extract horizontal profile at the middle of the image
    x, u, v = extract_horizontal_profile(U, V)
    
    # Plot horizontal profiles
    plot_horizontal_profiles(x, u, v, os.path.join(output_dir, 'horizontal_profiles.png'))
    
    # Analyze horizontal profile
    popt_v, r_squared_v = analyze_horizontal_profile(x, u, v, 
                                                   os.path.join(output_dir, 'horizontal_profile_analysis.png'))
    
    # Print results
    print("Parabolic Fit to Vertical Velocity Profile:")
    print(f"  Equation: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}")
    print(f"  R-squared: {r_squared_v:.4f}")
    
    # Extract and analyze profiles at different vertical positions
    positions = [U.shape[0] // 4, U.shape[0] // 2, 3 * U.shape[0] // 4]
    
    plt.figure(figsize=(12, 8))
    
    for i, pos in enumerate(positions):
        x, u, v = extract_horizontal_profile(U, V, position=pos)
        popt, r_squared = fit_parabola(x, v)
        
        # Generate fitted curve
        x_fit = np.linspace(x.min(), x.max(), 1000)
        v_fit = parabolic_function(x_fit, *popt)
        
        # Plot the data and fit
        plt.plot(x, v, '.', label=f'Data at y={pos}')
        plt.plot(x_fit, v_fit, '-', 
                 label=f'Fit at y={pos}: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f} (R²={r_squared:.4f})')
    
    plt.title('Vertical Velocity Profiles at Different Vertical Positions')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiple_profiles.png'), dpi=300)
    plt.close()
    
    print(f"Analysis results saved to {output_dir}")

if __name__ == "__main__":
    main()
