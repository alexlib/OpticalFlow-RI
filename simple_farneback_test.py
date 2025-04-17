#!/usr/bin/env python
"""
Simple test script for Farneback_Numba implementation.

This script creates a test case with known translation and verifies
that the Farneback_Numba implementation correctly detects the motion.
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from Farneback_Numba import Farneback_Numba

def create_translation_test(size=(100, 100), shift=(5, 8)):
    """Create a pair of images with pure translation."""
    # Create a random pattern image
    np.random.seed(42)  # For reproducibility
    img1 = np.random.rand(*size).astype(np.float32) * 255
    
    # Apply Gaussian blur to make it more realistic
    from scipy.ndimage import gaussian_filter
    img1 = gaussian_filter(img1, sigma=1.0)
    
    # Create a shifted version
    shift_x, shift_y = shift
    img2 = np.zeros_like(img1)
    
    if shift_x >= 0 and shift_y >= 0:
        img2[shift_y:, shift_x:] = img1[:-shift_y if shift_y > 0 else size[0], 
                                        :-shift_x if shift_x > 0 else size[1]]
    elif shift_x < 0 and shift_y >= 0:
        img2[shift_y:, :shift_x] = img1[:-shift_y if shift_y > 0 else size[0], 
                                        -shift_x:]
    elif shift_x >= 0 and shift_y < 0:
        img2[:shift_y, shift_x:] = img1[-shift_y:, 
                                        :-shift_x if shift_x > 0 else size[1]]
    else:  # shift_x < 0 and shift_y < 0
        img2[:shift_y, :shift_x] = img1[-shift_y:, -shift_x:]
    
    return img1, img2, (-shift_x, -shift_y)  # Return expected flow

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

def main():
    # Create output directory
    output_dir = 'farneback_test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test images with known translation
    shift = (5, 8)
    img1, img2, expected_flow = create_translation_test(size=(100, 100), shift=shift)
    expected_u, expected_v = expected_flow
    
    print(f"Created test images with shift {shift}")
    print(f"Expected flow: U={expected_u}, V={expected_v}")
    
    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)
    
    # Create Farneback algorithm
    fb = Farneback_Numba(
        windowSize=15,
        Niters=3,
        polyN=5,
        polySigma=1.2,
        pyramidalLevels=2
    )
    
    # Run the algorithm
    print("\nRunning Farneback_Numba implementation...")
    U, V, error = fb.compute(img1, img2, U, V)
    
    # Calculate mean flow
    mean_u = np.mean(U)
    mean_v = np.mean(V)
    
    print(f"\nResults:")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")
    print(f"Mean flow U: {mean_u:.2f} (expected: {expected_u})")
    print(f"Mean flow V: {mean_v:.2f} (expected: {expected_v})")
    
    # Calculate error
    u_error = abs(mean_u - expected_u)
    v_error = abs(mean_v - expected_v)
    
    print(f"\nError:")
    print(f"U error: {u_error:.2f}")
    print(f"V error: {v_error:.2f}")
    
    # Plot the flow
    plot_flow(U, V, f"Farneback Numba Flow (Expected: U={expected_u}, V={expected_v})", 
              f"{output_dir}/farneback_numba_flow.png")
    
    # Determine if test passed
    threshold = 2.0  # Allow some error due to noise and algorithm limitations
    if u_error < threshold and v_error < threshold:
        print("\nTEST PASSED: The Farneback_Numba implementation correctly detects translation.")
        print("This confirms it produces results similar to what the OpenCL version is supposed to do.")
    else:
        print("\nTEST FAILED: The Farneback_Numba implementation did not correctly detect translation.")
    
    print(f"\nResults saved to {output_dir} directory")

if __name__ == "__main__":
    main()
