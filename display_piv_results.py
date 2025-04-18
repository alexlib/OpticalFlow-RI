#!/usr/bin/env python
"""
Display PIV results from a .mat file according to user request.

Usage:
    python display_piv_results.py [options]

Options:
    --quiver             Display quiver plot
    --colormap           Display colormap of horizontal velocity
    --profile            Display left-to-right average profile
    --all                Display all plots (default if no option is specified)
    --output DIR         Save plots to directory DIR (default: 'piv_plots')
    --input FILE         Input .mat file (default: 'piv_results/piv_results.mat')
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit
import argparse

def load_piv_results(filename):
    """
    Load PIV results from a .mat file.

    Args:
        filename: Path to the .mat file

    Returns:
        Dictionary with PIV results
    """
    print(f"Loading PIV results from {filename}")

    try:
        data = loadmat(filename)
        return data
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        sys.exit(1)

def plot_quiver(data, output_dir=None):
    """
    Plot quiver of the PIV results.

    Args:
        data: Dictionary with PIV results
        output_dir: Directory to save the plot (optional)
    """
    print("Plotting quiver...")

    # Extract data
    x = data['x']
    y = data['y']
    u = data['u']
    v = data['v']
    window_size = data['window_size'][0, 0]
    overlap = data['overlap'][0, 0]
    search_area_size = data.get('search_area_size', np.array([[window_size]]))[0, 0]

    # Create figure
    plt.figure(figsize=(10, 8))

    # Calculate magnitude for coloring
    magnitude = np.sqrt(u**2 + v**2)

    # Plot quiver with colors based on magnitude
    quiv = plt.quiver(x, y, u, v, magnitude,
                     scale=50, scale_units='inches',
                     cmap='jet', clim=[0, np.percentile(magnitude, 95)])
    plt.colorbar(quiv, label='Magnitude (pixels/frame)')

    plt.title(f"PIV Vector Field (window={window_size}, overlap={overlap}, search={search_area_size})")
    plt.xlim(0, x.max())
    plt.ylim(y.max(), 0)  # Invert y-axis to match image coordinates
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_file = os.path.join(output_dir, 'piv_quiver.png')
        plt.savefig(output_file, dpi=200)
        print(f"Quiver plot saved to {output_file}")

    plt.show()

def plot_colormap(data, output_dir=None):
    """
    Plot colormap of horizontal velocity.

    Args:
        data: Dictionary with PIV results
        output_dir: Directory to save the plot (optional)
    """
    print("Plotting colormap of horizontal velocity...")

    # Extract data
    U = data['U']
    window_size = data['window_size'][0, 0]
    overlap = data['overlap'][0, 0]
    search_area_size = data.get('search_area_size', np.array([[window_size]]))[0, 0]

    # Create figure
    plt.figure(figsize=(10, 8))

    # Auto-calculate color limits for horizontal velocity
    u_abs_max = max(abs(np.percentile(U, 1)), abs(np.percentile(U, 99)))
    vmin = -u_abs_max
    vmax = u_abs_max

    # Plot horizontal velocity component as colormesh with high contrast colormap
    im = plt.imshow(U, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Horizontal Velocity (pixels/frame)')

    plt.title(f"Horizontal Velocity (window={window_size}, overlap={overlap}, search={search_area_size})")
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')

    plt.tight_layout()

    if output_dir:
        output_file = os.path.join(output_dir, 'piv_horizontal_velocity.png')
        plt.savefig(output_file, dpi=200)
        print(f"Colormap plot saved to {output_file}")

    plt.show()

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

def plot_profile(data, output_dir=None):
    """
    Plot left-to-right average profile.

    Args:
        data: Dictionary with PIV results
        output_dir: Directory to save the plot (optional)
    """
    print("Plotting left-to-right average profile...")

    # Extract data
    U = data['U']
    V = data['V']
    window_size = data['window_size'][0, 0]
    overlap = data['overlap'][0, 0]
    search_area_size = data.get('search_area_size', np.array([[window_size]]))[0, 0]

    # Calculate average profiles
    x = np.arange(U.shape[1])
    u_avg = np.mean(U, axis=1)  # Average along vertical axis
    v_avg = np.mean(V, axis=1)  # Average along vertical axis

    # Fit parabolas
    popt_u, r_squared_u = fit_parabola(x, u_avg)
    popt_v, r_squared_v = fit_parabola(x, v_avg)

    # Generate fitted curves
    x_fit = np.linspace(x.min(), x.max(), 1000)
    u_fit = parabolic_function(x_fit, *popt_u)
    v_fit = parabolic_function(x_fit, *popt_v)

    # Create figure with two subplots
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    # Plot horizontal velocity profile with parabolic fit
    plt.plot(x, u_avg, 'b.', label='Data')
    plt.plot(x_fit, u_fit, 'r-', linewidth=2,
             label=f'Parabolic Fit: {popt_u[0]:.6f}x² + {popt_u[1]:.6f}x + {popt_u[2]:.6f}')
    plt.title(f'Horizontal Velocity Profile with Parabolic Fit (R² = {r_squared_u:.4f})')
    plt.xlabel('X (pixels)')
    plt.ylabel('Horizontal Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    # Plot vertical velocity profile with parabolic fit
    plt.plot(x, v_avg, 'g.', label='Data')
    plt.plot(x_fit, v_fit, 'm-', linewidth=2,
             label=f'Parabolic Fit: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}')
    plt.title(f'Vertical Velocity Profile with Parabolic Fit (R² = {r_squared_v:.4f})')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()

    plt.suptitle(f"Average Velocity Profiles (window={window_size}, overlap={overlap}, search={search_area_size})")
    plt.tight_layout()

    if output_dir:
        output_file = os.path.join(output_dir, 'piv_average_profile.png')
        plt.savefig(output_file, dpi=200)
        print(f"Profile plot saved to {output_file}")

    plt.show()

    # Print fit parameters
    print("\nHorizontal Velocity Parabolic Fit:")
    print(f"Equation: {popt_u[0]:.6f}x² + {popt_u[1]:.6f}x + {popt_u[2]:.6f}")
    print(f"R-squared: {r_squared_u:.4f}")

    print("\nVertical Velocity Parabolic Fit:")
    print(f"Equation: {popt_v[0]:.6f}x² + {popt_v[1]:.6f}x + {popt_v[2]:.6f}")
    print(f"R-squared: {r_squared_v:.4f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Display PIV results from a .mat file.')
    parser.add_argument('--quiver', action='store_true', help='Display quiver plot')
    parser.add_argument('--colormap', action='store_true', help='Display colormap of horizontal velocity')
    parser.add_argument('--profile', action='store_true', help='Display left-to-right average profile')
    parser.add_argument('--all', action='store_true', help='Display all plots (default if no option is specified)')
    parser.add_argument('--output', type=str, default='piv_plots', help='Directory to save plots')
    parser.add_argument('--input', type=str, default='piv_results/piv_results.mat', help='Input .mat file')

    args = parser.parse_args()

    # If no plot type is specified, show all
    if not (args.quiver or args.colormap or args.profile):
        args.all = True

    # Create output directory if saving plots
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Load PIV results
    data = load_piv_results(args.input)

    # Display requested plots
    if args.quiver or args.all:
        plot_quiver(data, args.output)

    if args.colormap or args.all:
        plot_colormap(data, args.output)

    if args.profile or args.all:
        plot_profile(data, args.output)

if __name__ == "__main__":
    main()
