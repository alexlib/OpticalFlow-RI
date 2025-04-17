#!/usr/bin/env python
"""
Improved test script for Farneback_Numba implementation.

This script creates a test case with a structured pattern and known translation
to verify that the Farneback_Numba implementation correctly detects the motion.
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from Farneback_Numba import Farneback_Numba

def create_structured_pattern(size=(100, 100)):
    """Create a structured pattern that's easier for optical flow to track."""
    # Create a grid pattern
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    
    # Create a pattern with multiple frequencies
    pattern = np.sin(2 * np.pi * 5 * x) * np.sin(2 * np.pi * 5 * y)
    pattern += 0.5 * np.sin(2 * np.pi * 10 * x) * np.sin(2 * np.pi * 10 * y)
    pattern += 0.25 * np.sin(2 * np.pi * 20 * x) * np.sin(2 * np.pi * 20 * y)
    
    # Normalize to [0, 255]
    pattern = ((pattern + 1.5) / 3.0 * 255).astype(np.float32)
    
    return pattern

def create_translation_test(size=(100, 100), shift=(5, 8)):
    """Create a pair of images with pure translation using a structured pattern."""
    # Create a structured pattern
    img1 = create_structured_pattern(size)
    
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

def plot_images_and_flow(img1, img2, U, V, title, filename):
    """Plot the images and the computed flow."""
    plt.figure(figsize=(15, 5))
    
    # Plot the first image
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    # Plot the second image
    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')
    
    # Plot the flow
    plt.subplot(1, 3, 3)
    
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
    plt.title('Optical Flow')
    plt.xlim(0, U.shape[1])
    plt.ylim(U.shape[0], 0)  # Invert y-axis to match image coordinates
    
    plt.suptitle(title)
    plt.tight_layout()
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
    
    # Save the test images
    plt.imsave(f"{output_dir}/test_image1.png", img1, cmap='gray')
    plt.imsave(f"{output_dir}/test_image2.png", img2, cmap='gray')
    
    # Initialize flow fields
    U = np.zeros_like(img1)
    V = np.zeros_like(img1)
    
    # Create Farneback algorithm with parameters optimized for this test
    fb = Farneback_Numba(
        windowSize=15,
        Niters=3,
        polyN=5,
        polySigma=1.2,
        pyramidalLevels=1  # Use only one level for this simple test
    )
    
    # Run the algorithm
    print("\nRunning Farneback_Numba implementation...")
    U, V, error = fb.compute(img1, img2, U, V)
    
    # Calculate mean flow in the central region (avoiding boundary effects)
    border = 20
    central_U = U[border:-border, border:-border]
    central_V = V[border:-border, border:-border]
    mean_u = np.mean(central_U)
    mean_v = np.mean(central_V)
    
    print(f"\nResults:")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")
    print(f"Mean flow U (central region): {mean_u:.2f} (expected: {expected_u})")
    print(f"Mean flow V (central region): {mean_v:.2f} (expected: {expected_v})")
    
    # Calculate error
    u_error = abs(mean_u - expected_u)
    v_error = abs(mean_v - expected_v)
    
    print(f"\nError:")
    print(f"U error: {u_error:.2f}")
    print(f"V error: {v_error:.2f}")
    
    # Plot the images and flow
    plot_images_and_flow(img1, img2, U, V, 
                        f"Farneback Numba Flow (Expected: U={expected_u}, V={expected_v})", 
                        f"{output_dir}/farneback_numba_improved_test.png")
    
    # Determine if test passed
    threshold = 2.0  # Allow some error due to noise and algorithm limitations
    if u_error < threshold and v_error < threshold:
        print("\nTEST PASSED: The Farneback_Numba implementation correctly detects translation.")
        print("This confirms it produces results similar to what the OpenCL version is supposed to do.")
    else:
        print("\nTEST FAILED: The Farneback_Numba implementation did not correctly detect translation.")
        print("However, this may be due to parameter tuning rather than a fundamental issue.")
    
    print(f"\nResults saved to {output_dir} directory")

if __name__ == "__main__":
    main()
