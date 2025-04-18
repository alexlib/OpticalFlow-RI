#!/usr/bin/env python
"""
Run the denseLucasKanade_Numba implementation on the Bits08 test case.
"""

import os
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.io import savemat
import time

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the implementation
from denseLucasKanade_Numba_simple import denseLucasKanade_Numba
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow

def load_bits08_images():
    """Load the Bits08 test images."""
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

def plot_flow_field(U, V, title, output_filename):
    """Plot optical flow as colormesh and quiver plots."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Auto-calculate color limits
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

def plot_parabolic_profiles(U, V, title, output_filename):
    """Plot parabolic profiles."""
    # Extract profiles
    x_h, v_h = extract_parabolic_profile(U, V, axis='horizontal')
    x_v, v_v = extract_parabolic_profile(U, V, axis='vertical')
    
    plt.figure(figsize=(12, 5))
    
    # Plot horizontal profile
    plt.subplot(1, 2, 1)
    plt.plot(x_h, v_h)
    plt.title('Horizontal Parabolic Profile')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    
    # Plot vertical profile
    plt.subplot(1, 2, 2)
    plt.plot(x_v, v_v)
    plt.title('Vertical Parabolic Profile')
    plt.xlabel('Position (pixels)')
    plt.ylabel('Vertical Velocity (pixels/frame)')
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close()
    
    return (x_h, v_h), (x_v, v_v)

def main():
    # Load test images
    im1, im2 = load_bits08_images()
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Create the Lucas-Kanade object
    lk = denseLucasKanade_Numba(Niter=5, halfWindow=13)
    
    # Parameters for GenericPyramidalOpticalFlow
    FILTER = 2
    FILTER_OPT = 0.48
    pyramidalLevels = 2
    kLevels = 1
    
    # Create output directory
    output_dir = 'denseLK_numba_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute optical flow using GenericPyramidalOpticalFlow
    start_time = time.time()
    
    U_out, V_out = genericPyramidalOpticalFlow(
        im1, im2, FILTER, lk, pyramidalLevels, kLevels, 
        FILTER_OPT, None, warping=False
    )
    
    elapsed_time = time.time() - start_time
    print(f"Optical flow computation completed in {elapsed_time:.2f} seconds")
    
    # Save the flow field
    save_flow(U_out, V_out, os.path.join(output_dir, 'denseLK_numba.mat'))
    
    # Plot the flow field
    plot_flow_field(U_out, V_out, 'Dense Lucas-Kanade (Numba)', 
                   os.path.join(output_dir, 'denseLK_numba_flow.png'))
    
    # Plot parabolic profiles
    plot_parabolic_profiles(U_out, V_out, 'Dense Lucas-Kanade (Numba)',
                           os.path.join(output_dir, 'denseLK_numba_profiles.png'))
    
    print("Results saved to", output_dir)

if __name__ == "__main__":
    main()
