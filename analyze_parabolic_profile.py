#!/usr/bin/env python
"""
Analyze the parabolic flow profile from the optical flow results.
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

def extract_parabolic_profile(U, V, axis='horizontal', position=None):
    """
    Extract a parabolic profile from the flow field.
    
    Args:
        U, V: Horizontal and vertical velocity components
        axis: 'horizontal' or 'vertical'
        position: Position along the perpendicular axis (default: middle)
        
    Returns:
        x: Position along the axis
        v: Velocity profile
    """
    if axis == 'horizontal':
        if position is None:
            position = U.shape[0] // 2
        x = np.arange(U.shape[1])
        v = V[position, :]
    else:  # vertical
        if position is None:
            position = U.shape[1] // 2
        x = np.arange(U.shape[0])
        v = V[:, position]
    
    return x, v

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

def analyze_profile(x, v, title, output_filename):
    """Analyze and plot the parabolic profile."""
    # Fit a parabola to the data
    popt, r_squared = fit_parabola(x, v)
    
    # Generate fitted curve
    x_fit = np.linspace(x.min(), x.max(), 1000)
    v_fit = parabolic_function(x_fit, *popt)
    
    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    plt.plot(x, v, 'b.', label='Data')
    plt.plot(x_fit, v_fit, 'r-', label=f'Fit: {popt[0]:.6f}x² + {popt[1]:.6f}x + {popt[2]:.6f}')
    plt.title(f'{title} (R² = {r_squared:.4f})')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    
    return popt, r_squared

def main():
    # Load flow field
    flow_file = 'denseLK_numba_optimized_results/denseLK_numba_optimized.mat'
    U, V = load_flow_field(flow_file)
    
    # Create output directory
    output_dir = 'parabolic_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and analyze horizontal profile
    x_h, v_h = extract_parabolic_profile(U, V, axis='horizontal')
    popt_h, r_squared_h = analyze_profile(x_h, v_h, 'Horizontal Parabolic Profile', 
                                         os.path.join(output_dir, 'horizontal_profile_fit.png'))
    
    # Extract and analyze vertical profile
    x_v, v_v = extract_parabolic_profile(U, V, axis='vertical')
    popt_v, r_squared_v = analyze_profile(x_v, v_v, 'Vertical Parabolic Profile', 
                                         os.path.join(output_dir, 'vertical_profile_fit.png'))
    
    # Print results
    print("Horizontal Profile Fit:")
    print(f"  Equation: {popt_h[0]:.6f}x² + {popt_h[1]:.6f}x + {popt_h[2]:.6f}")
    print(f"  R-squared: {r_squared_h:.4f}")
    print()
    print("Vertical Profile Fit:")
    print(f"  Equation: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}")
    print(f"  R-squared: {r_squared_v:.4f}")
    
    # Create a 3D visualization of the flow field
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of points
    x = np.arange(0, U.shape[1], 5)
    y = np.arange(0, U.shape[0], 5)
    X, Y = np.meshgrid(x, y)
    
    # Get the vertical velocity at each point
    Z = V[y[:, np.newaxis], x]
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=True)
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Vertical Velocity (pixels/frame)')
    
    # Set labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Vertical Velocity (pixels/frame)')
    ax.set_title('3D Visualization of Parabolic Flow Field')
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, '3d_flow_field.png'), dpi=300)
    plt.close()
    
    print(f"Analysis results saved to {output_dir}")

if __name__ == "__main__":
    main()
