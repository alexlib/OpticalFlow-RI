#!/usr/bin/env python
"""
Test script to compare Farneback_PyCL and Farneback_Numba implementations.

This script creates a simple test case with known motion, runs both implementations,
and compares the results to verify they produce similar optical flow.
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.data import camera
import matplotlib.pyplot as plt
from Farneback_PyCL import Farneback_PyCL
from Farneback_Numba import Farneback_Numba

def create_test_images(size=(200, 200), shift=(5, 8)):
    """Create a pair of test images with known shift."""
    print(f"Creating test images with size {size} and shift {shift}...")
    
    # Use a standard test image
    img1 = camera()
    img1 = img1.astype(np.float32)
    
    # Resize to desired size
    from skimage.transform import resize
    img1 = resize(img1, size, preserve_range=True).astype(np.float32)
    
    # Create a shifted version of the image
    shift_x, shift_y = shift
    img2 = np.zeros_like(img1)
    img2[shift_y:, shift_x:] = img1[:-shift_y, :-shift_x]
    
    return img1, img2

def plot_flow(U, V, title, filename):
    """Plot optical flow as a quiver plot and save to file."""
    plt.figure(figsize=(10, 8))
    
    # Create a grid of points
    y, x = np.mgrid[0:U.shape[0]:5, 0:U.shape[1]:5]
    
    # Subsample the flow for visualization
    u = U[::5, ::5]
    v = V[::5, ::5]
    
    # Calculate magnitude for coloring
    magnitude = np.sqrt(u**2 + v**2)
    
    # Plot quiver with colors based on magnitude
    plt.quiver(x, y, u, v, magnitude, 
               scale=50, scale_units='inches',
               cmap='jet')
    
    plt.colorbar(label='Magnitude (pixels/frame)')
    plt.title(title)
    plt.xlim(0, U.shape[1])
    plt.ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(filename, dpi=150)
    plt.close()

def compare_flows(U1, V1, U2, V2):
    """Compare two flow fields and return similarity metrics."""
    # Calculate mean absolute difference
    u_diff = np.abs(U1 - U2)
    v_diff = np.abs(V1 - V2)
    
    # Calculate mean and max differences
    mean_u_diff = np.mean(u_diff)
    mean_v_diff = np.mean(v_diff)
    max_u_diff = np.max(u_diff)
    max_v_diff = np.max(v_diff)
    
    # Calculate correlation coefficient
    u_corr = np.corrcoef(U1.flatten(), U2.flatten())[0, 1]
    v_corr = np.corrcoef(V1.flatten(), V2.flatten())[0, 1]
    
    return {
        'mean_u_diff': mean_u_diff,
        'mean_v_diff': mean_v_diff,
        'max_u_diff': max_u_diff,
        'max_v_diff': max_v_diff,
        'u_correlation': u_corr,
        'v_correlation': v_corr
    }

def plot_difference(U1, V1, U2, V2, title, filename):
    """Plot the difference between two flow fields."""
    diff_U = U1 - U2
    diff_V = V1 - V2
    
    plt.figure(figsize=(12, 10))
    
    # Plot U difference
    plt.subplot(2, 1, 1)
    plt.imshow(diff_U, cmap='RdBu_r')
    plt.colorbar(label='U difference')
    plt.title(f'{title} - U Component Difference')
    
    # Plot V difference
    plt.subplot(2, 1, 2)
    plt.imshow(diff_V, cmap='RdBu_r')
    plt.colorbar(label='V difference')
    plt.title(f'{title} - V Component Difference')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    # Create output directory
    output_dir = 'farneback_comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test images
    img1, img2 = create_test_images(size=(200, 200), shift=(5, 8))
    
    # Initialize flow fields
    U_opencl = np.zeros_like(img1)
    V_opencl = np.zeros_like(img1)
    U_numba = np.zeros_like(img1)
    V_numba = np.zeros_like(img1)
    
    # Common parameters for both implementations
    common_params = {
        'windowSize': 15,
        'Niters': 3,
        'polyN': 5,
        'polySigma': 1.2,
        'pyramidalLevels': 2
    }
    
    # Run OpenCL implementation
    print("\nRunning Farneback_PyCL implementation...")
    try:
        fb_opencl = Farneback_PyCL(**common_params)
        U_opencl, V_opencl, error_opencl = fb_opencl.compute(img1, img2, U_opencl, V_opencl)
        print(f"OpenCL implementation completed with error: {error_opencl}")
        print(f"Flow range U: {U_opencl.min():.2f} to {U_opencl.max():.2f}")
        print(f"Flow range V: {V_opencl.min():.2f} to {V_opencl.max():.2f}")
        opencl_success = True
    except Exception as e:
        print(f"Error running OpenCL implementation: {e}")
        opencl_success = False
    
    # Run Numba implementation
    print("\nRunning Farneback_Numba implementation...")
    fb_numba = Farneback_Numba(**common_params)
    U_numba, V_numba, error_numba = fb_numba.compute(img1, img2, U_numba, V_numba)
    print(f"Numba implementation completed with error: {error_numba}")
    print(f"Flow range U: {U_numba.min():.2f} to {U_numba.max():.2f}")
    print(f"Flow range V: {V_numba.min():.2f} to {V_numba.max():.2f}")
    
    # Plot Numba results
    plot_flow(U_numba, V_numba, "Farneback Numba Flow", f"{output_dir}/farneback_numba_flow.png")
    
    # If OpenCL implementation succeeded, compare results
    if opencl_success:
        # Plot OpenCL results
        plot_flow(U_opencl, V_opencl, "Farneback OpenCL Flow", f"{output_dir}/farneback_opencl_flow.png")
        
        # Compare the results
        comparison = compare_flows(U_opencl, V_opencl, U_numba, V_numba)
        print("\nComparison of OpenCL and Numba implementations:")
        for key, value in comparison.items():
            print(f"{key}: {value:.6f}")
        
        # Plot differences
        plot_difference(U_opencl, V_opencl, U_numba, V_numba, 
                       "OpenCL vs Numba", f"{output_dir}/farneback_difference.png")
        
        # Determine if implementations are similar
        if (comparison['u_correlation'] > 0.9 and comparison['v_correlation'] > 0.9 and
            comparison['mean_u_diff'] < 1.0 and comparison['mean_v_diff'] < 1.0):
            print("\nCONCLUSION: The implementations produce similar results!")
        else:
            print("\nCONCLUSION: The implementations produce different results.")
    else:
        print("\nCannot compare implementations because OpenCL implementation failed.")
        print("This is expected if you don't have OpenCL support or a compatible GPU.")
        print("The Numba implementation should still work correctly.")
    
    print(f"\nResults saved to {output_dir} directory")

if __name__ == "__main__":
    main()
