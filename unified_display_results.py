#!/usr/bin/env python
"""
Unified display script for optical flow and PIV results.

This script can display results from different optical flow and PIV methods
in a consistent format, allowing for easy comparison.

Usage:
    python unified_display_results.py [options]

Options:
    --quiver             Display quiver plot
    --colormap           Display colormap of horizontal velocity
    --profile            Display left-to-right average profile
    --all                Display all plots (default if no option is specified)
    --output DIR         Save plots to directory DIR (default: 'flow_plots')
    --input FILE         Input .mat file
    --method NAME        Method name for plot titles (default: derived from filename)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import curve_fit
import argparse
import re

def load_results(filename):
    """
    Load results from a .mat file.

    Args:
        filename: Path to the .mat file

    Returns:
        Dictionary with results and metadata
    """
    print(f"Loading results from {filename}")

    try:
        data = loadmat(filename)
        
        # Extract method name from filename if not provided
        method_name = os.path.basename(filename).split('.')[0]
        method_name = method_name.replace('_', ' ').title()
        
        # Create a metadata dictionary
        metadata = {
            'method_name': method_name,
            'filename': filename
        }
        
        # Check if this is a PIV result or optical flow result
        if 'x' in data and 'y' in data and 'u' in data and 'v' in data:
            # This is a PIV result with sparse grid
            metadata['type'] = 'piv_sparse'
        elif 'U' in data and 'V' in data:
            # This is an optical flow result with dense grid
            metadata['type'] = 'optical_flow'
        else:
            print("Warning: Unknown data format. Attempting to process anyway.")
            metadata['type'] = 'unknown'
        
        # Extract parameters if available
        if 'window_size' in data:
            metadata['window_size'] = data['window_size'][0, 0]
        else:
            metadata['window_size'] = None
            
        if 'overlap' in data:
            metadata['overlap'] = data['overlap'][0, 0]
        else:
            metadata['overlap'] = None
            
        if 'search_area_size' in data:
            metadata['search_area_size'] = data['search_area_size'][0, 0]
        elif 'window_size' in data:
            metadata['search_area_size'] = data['window_size'][0, 0]
        else:
            metadata['search_area_size'] = None
            
        if 'smooth_sigma' in data:
            metadata['smooth_sigma'] = data['smooth_sigma'][0, 0]
        else:
            metadata['smooth_sigma'] = None
        
        return data, metadata
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        sys.exit(1)

def plot_quiver(data, metadata, output_dir=None):
    """
    Plot quiver of the results.

    Args:
        data: Dictionary with results
        metadata: Dictionary with metadata
        output_dir: Directory to save the plot (optional)
    """
    print("Plotting quiver...")
    
    method_name = metadata['method_name']
    result_type = metadata['type']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    if result_type == 'piv_sparse':
        # Extract data for sparse PIV grid
        x = data['x']
        y = data['y']
        u = data['u']
        v = data['v']
        
        # Calculate magnitude for coloring
        magnitude = np.sqrt(u**2 + v**2)
        
        # Plot quiver with colors based on magnitude
        quiv = plt.quiver(x, y, u, v, magnitude,
                         scale=50, scale_units='inches',
                         cmap='jet', clim=[0, np.percentile(magnitude, 95)])
        plt.colorbar(quiv, label='Magnitude (pixels/frame)')
        
        plt.xlim(0, x.max())
        plt.ylim(y.max(), 0)  # Invert y-axis to match image coordinates
        
    elif result_type == 'optical_flow' or result_type == 'unknown':
        # Extract data for dense optical flow grid
        U = data['U']
        V = data['V']
        
        # Create a downsampled grid for quiver
        h, w = U.shape
        step = max(1, min(h, w) // 40)  # Adjust step for reasonable arrow density
        y, x = np.mgrid[0:h:step, 0:w:step]
        u = U[::step, ::step]
        v = V[::step, ::step]
        
        # Calculate magnitude for coloring
        magnitude = np.sqrt(u**2 + v**2)
        
        # Plot quiver with colors based on magnitude
        quiv = plt.quiver(x, y, u, v, magnitude,
                         scale=50, scale_units='inches',
                         cmap='jet', clim=[0, np.percentile(magnitude, 95)])
        plt.colorbar(quiv, label='Magnitude (pixels/frame)')
        
        plt.xlim(0, w)
        plt.ylim(h, 0)  # Invert y-axis to match image coordinates
    
    # Add parameter information to title if available
    title = f"{method_name} - Vector Field"
    if metadata['window_size'] is not None:
        title += f" (window={metadata['window_size']}"
        if metadata['overlap'] is not None:
            title += f", overlap={metadata['overlap']}"
        if metadata['search_area_size'] is not None:
            title += f", search={metadata['search_area_size']}"
        if metadata['smooth_sigma'] is not None:
            title += f", sigma={metadata['smooth_sigma']}"
        title += ")"
    
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        # Create a safe filename from the method name
        safe_name = re.sub(r'[^\w\-_]', '_', method_name.lower())
        output_file = os.path.join(output_dir, f'{safe_name}_quiver.png')
        plt.savefig(output_file, dpi=200)
        print(f"Quiver plot saved to {output_file}")
    
    plt.show()

def plot_colormap(data, metadata, output_dir=None):
    """
    Plot colormap of horizontal velocity.

    Args:
        data: Dictionary with results
        metadata: Dictionary with metadata
        output_dir: Directory to save the plot (optional)
    """
    print("Plotting colormap of horizontal velocity...")
    
    method_name = metadata['method_name']
    result_type = metadata['type']
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    if result_type == 'piv_sparse':
        # For sparse PIV, we need the interpolated grid
        if 'U' in data:
            U = data['U']
        else:
            print("Error: Interpolated grid (U) not found in PIV data.")
            return
    elif result_type == 'optical_flow' or result_type == 'unknown':
        # For optical flow, we already have the dense grid
        U = data['U']
    
    # Auto-calculate color limits for horizontal velocity
    u_abs_max = max(abs(np.percentile(U, 1)), abs(np.percentile(U, 99)))
    vmin = -u_abs_max
    vmax = u_abs_max
    
    # Plot horizontal velocity component as colormesh with high contrast colormap
    im = plt.imshow(U, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='Horizontal Velocity (pixels/frame)')
    
    # Add parameter information to title if available
    title = f"{method_name} - Horizontal Velocity"
    if metadata['window_size'] is not None:
        title += f" (window={metadata['window_size']}"
        if metadata['overlap'] is not None:
            title += f", overlap={metadata['overlap']}"
        if metadata['search_area_size'] is not None:
            title += f", search={metadata['search_area_size']}"
        if metadata['smooth_sigma'] is not None:
            title += f", sigma={metadata['smooth_sigma']}"
        title += ")"
    
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    plt.tight_layout()
    
    if output_dir:
        # Create a safe filename from the method name
        safe_name = re.sub(r'[^\w\-_]', '_', method_name.lower())
        output_file = os.path.join(output_dir, f'{safe_name}_colormap.png')
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

def plot_profile(data, metadata, output_dir=None):
    """
    Plot left-to-right average profile.

    Args:
        data: Dictionary with results
        metadata: Dictionary with metadata
        output_dir: Directory to save the plot (optional)
    """
    print("Plotting left-to-right average profile...")
    
    method_name = metadata['method_name']
    result_type = metadata['type']
    
    # Extract data
    if result_type == 'piv_sparse':
        # For sparse PIV, we need the interpolated grid
        if 'U' in data and 'V' in data:
            U = data['U']
            V = data['V']
        else:
            print("Error: Interpolated grid (U, V) not found in PIV data.")
            return
    elif result_type == 'optical_flow' or result_type == 'unknown':
        # For optical flow, we already have the dense grid
        U = data['U']
        V = data['V']
    
    # Calculate average profiles
    y_indices = np.arange(U.shape[0])
    x_indices = np.arange(U.shape[1])
    
    # Average along horizontal axis (for each y)
    u_avg_y = np.mean(U, axis=1)  # Average u for each y
    v_avg_y = np.mean(V, axis=1)  # Average v for each y
    
    # Fit parabolas to the horizontal profiles
    popt_u_y, r_squared_u_y = fit_parabola(y_indices, u_avg_y)
    popt_v_y, r_squared_v_y = fit_parabola(y_indices, v_avg_y)
    
    # Generate fitted curves
    y_fit = np.linspace(y_indices.min(), y_indices.max(), 1000)
    u_fit_y = parabolic_function(y_fit, *popt_u_y)
    v_fit_y = parabolic_function(y_fit, *popt_v_y)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    # Plot horizontal velocity profile with parabolic fit
    plt.plot(y_indices, u_avg_y, 'b.', label='Data')
    plt.plot(y_fit, u_fit_y, 'r-', linewidth=2,
             label=f'Parabolic Fit: {popt_u_y[0]:.6f}y² + {popt_u_y[1]:.6f}y + {popt_u_y[2]:.6f}')
    plt.title(f'Horizontal Velocity Profile (Y) with Parabolic Fit (R² = {r_squared_u_y:.4f})')
    plt.xlabel('Y (pixels)')
    plt.ylabel('Horizontal Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    # Plot vertical velocity profile with parabolic fit
    plt.plot(y_indices, v_avg_y, 'g.', label='Data')
    plt.plot(y_fit, v_fit_y, 'm-', linewidth=2,
             label=f'Parabolic Fit: {popt_v_y[0]:.6f}y² + {popt_v_y[1]:.6f}y + {popt_v_y[2]:.6f}')
    plt.title(f'Vertical Velocity Profile (Y) with Parabolic Fit (R² = {r_squared_v_y:.4f})')
    plt.xlabel('Y (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()
    
    # Add parameter information to title if available
    title = f"{method_name} - Average Velocity Profiles"
    if metadata['window_size'] is not None:
        title += f" (window={metadata['window_size']}"
        if metadata['overlap'] is not None:
            title += f", overlap={metadata['overlap']}"
        if metadata['search_area_size'] is not None:
            title += f", search={metadata['search_area_size']}"
        if metadata['smooth_sigma'] is not None:
            title += f", sigma={metadata['smooth_sigma']}"
        title += ")"
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_dir:
        # Create a safe filename from the method name
        safe_name = re.sub(r'[^\w\-_]', '_', method_name.lower())
        output_file = os.path.join(output_dir, f'{safe_name}_profile_y.png')
        plt.savefig(output_file, dpi=200)
        print(f"Y-Profile plot saved to {output_file}")
    
    plt.show()
    
    # Now create a plot for the X profiles (vertical average)
    
    # Average along vertical axis (for each x)
    u_avg_x = np.mean(U, axis=0)  # Average u for each x
    v_avg_x = np.mean(V, axis=0)  # Average v for each x
    
    # Fit parabolas to the vertical profiles
    popt_u_x, r_squared_u_x = fit_parabola(x_indices, u_avg_x)
    popt_v_x, r_squared_v_x = fit_parabola(x_indices, v_avg_x)
    
    # Generate fitted curves
    x_fit = np.linspace(x_indices.min(), x_indices.max(), 1000)
    u_fit_x = parabolic_function(x_fit, *popt_u_x)
    v_fit_x = parabolic_function(x_fit, *popt_v_x)
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    # Plot horizontal velocity profile with parabolic fit
    plt.plot(x_indices, u_avg_x, 'b.', label='Data')
    plt.plot(x_fit, u_fit_x, 'r-', linewidth=2,
             label=f'Parabolic Fit: {popt_u_x[0]:.6f}x² + {popt_u_x[1]:.6f}x + {popt_u_x[2]:.6f}')
    plt.title(f'Horizontal Velocity Profile (X) with Parabolic Fit (R² = {r_squared_u_x:.4f})')
    plt.xlabel('X (pixels)')
    plt.ylabel('Horizontal Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    # Plot vertical velocity profile with parabolic fit
    plt.plot(x_indices, v_avg_x, 'g.', label='Data')
    plt.plot(x_fit, v_fit_x, 'm-', linewidth=2,
             label=f'Parabolic Fit: {popt_v_x[0]:.6f}x² + {popt_v_x[1]:.6f}x + {popt_v_x[2]:.6f}')
    plt.title(f'Vertical Velocity Profile (X) with Parabolic Fit (R² = {r_squared_v_x:.4f})')
    plt.xlabel('X (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_dir:
        # Create a safe filename from the method name
        safe_name = re.sub(r'[^\w\-_]', '_', method_name.lower())
        output_file = os.path.join(output_dir, f'{safe_name}_profile_x.png')
        plt.savefig(output_file, dpi=200)
        print(f"X-Profile plot saved to {output_file}")
    
    plt.show()
    
    # Print fit parameters
    print("\nHorizontal Velocity Parabolic Fit (Y-Profile):")
    print(f"Equation: {popt_u_y[0]:.6f}y² + {popt_u_y[1]:.6f}y + {popt_u_y[2]:.6f}")
    print(f"R-squared: {r_squared_u_y:.4f}")
    
    print("\nVertical Velocity Parabolic Fit (Y-Profile):")
    print(f"Equation: {popt_v_y[0]:.6f}y² + {popt_v_y[1]:.6f}y + {popt_v_y[2]:.6f}")
    print(f"R-squared: {r_squared_v_y:.4f}")
    
    print("\nHorizontal Velocity Parabolic Fit (X-Profile):")
    print(f"Equation: {popt_u_x[0]:.6f}x² + {popt_u_x[1]:.6f}x + {popt_u_x[2]:.6f}")
    print(f"R-squared: {r_squared_u_x:.4f}")
    
    print("\nVertical Velocity Parabolic Fit (X-Profile):")
    print(f"Equation: {popt_v_x[0]:.6f}x² + {popt_v_x[1]:.6f}x + {popt_v_x[2]:.6f}")
    print(f"R-squared: {r_squared_v_x:.4f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Unified display script for optical flow and PIV results.')
    parser.add_argument('--quiver', action='store_true', help='Display quiver plot')
    parser.add_argument('--colormap', action='store_true', help='Display colormap of horizontal velocity')
    parser.add_argument('--profile', action='store_true', help='Display left-to-right average profile')
    parser.add_argument('--all', action='store_true', help='Display all plots (default if no option is specified)')
    parser.add_argument('--output', type=str, default='flow_plots', help='Directory to save plots')
    parser.add_argument('--input', type=str, required=True, help='Input .mat file')
    parser.add_argument('--method', type=str, help='Method name for plot titles')
    
    args = parser.parse_args()
    
    # If no plot type is specified, show all
    if not (args.quiver or args.colormap or args.profile):
        args.all = True
    
    # Create output directory if saving plots
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Load results
    data, metadata = load_results(args.input)
    
    # Override method name if provided
    if args.method:
        metadata['method_name'] = args.method
    
    # Display requested plots
    if args.quiver or args.all:
        plot_quiver(data, metadata, args.output)
    
    if args.colormap or args.all:
        plot_colormap(data, metadata, args.output)
    
    if args.profile or args.all:
        plot_profile(data, metadata, args.output)

if __name__ == "__main__":
    main()
