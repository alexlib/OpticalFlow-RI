#!/usr/bin/env python
"""
Test script to compare denseLucasKanade_PyCL and denseLucasKanade_Numba implementations.

This script creates a simple test case with known motion, runs both implementations,
and compares the results to verify they produce similar optical flow.
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.data import camera
import matplotlib.pyplot as plt
from skimage.transform import resize
import time

def create_translation_test(size=(200, 200), shift=(5, 8)):
    """Create a pair of test images with pure translation using a structured pattern."""
    # Create a checkerboard pattern
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    
    # Create a pattern with multiple frequencies
    pattern = np.sin(2 * np.pi * 5 * x) * np.sin(2 * np.pi * 5 * y)
    pattern += 0.5 * np.sin(2 * np.pi * 10 * x) * np.sin(2 * np.pi * 10 * y)
    pattern += 0.25 * np.sin(2 * np.pi * 20 * x) * np.sin(2 * np.pi * 20 * y)
    
    # Normalize to [0, 255]
    img1 = ((pattern + 1.5) / 3.0 * 255).astype(np.float32)
    
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

def plot_flow_comparison(U1, V1, U2, V2, title, filename):
    """Plot the comparison between two flow fields."""
    plt.figure(figsize=(15, 10))
    
    # Plot the first flow field
    plt.subplot(2, 2, 1)
    y, x = np.mgrid[0:U1.shape[0]:5, 0:U1.shape[1]:5]
    u1 = U1[::5, ::5]
    v1 = V1[::5, ::5]
    magnitude1 = np.sqrt(u1**2 + v1**2)
    plt.quiver(x, y, u1, v1, magnitude1, scale=50, scale_units='inches', cmap='jet')
    plt.colorbar(label='Magnitude (pixels/frame)')
    plt.title('OpenCL Flow')
    plt.xlim(0, U1.shape[1])
    plt.ylim(U1.shape[0], 0)
    
    # Plot the second flow field
    plt.subplot(2, 2, 2)
    u2 = U2[::5, ::5]
    v2 = V2[::5, ::5]
    magnitude2 = np.sqrt(u2**2 + v2**2)
    plt.quiver(x, y, u2, v2, magnitude2, scale=50, scale_units='inches', cmap='jet')
    plt.colorbar(label='Magnitude (pixels/frame)')
    plt.title('Numba Flow')
    plt.xlim(0, U2.shape[1])
    plt.ylim(U2.shape[0], 0)
    
    # Plot the difference in U component
    plt.subplot(2, 2, 3)
    diff_U = U1 - U2
    plt.imshow(diff_U, cmap='RdBu_r')
    plt.colorbar(label='U difference')
    plt.title('U Component Difference')
    
    # Plot the difference in V component
    plt.subplot(2, 2, 4)
    diff_V = V1 - V2
    plt.imshow(diff_V, cmap='RdBu_r')
    plt.colorbar(label='V difference')
    plt.title('V Component Difference')
    
    plt.suptitle(title)
    plt.tight_layout()
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

def main():
    # Create output directory
    output_dir = 'dense_lk_comparison_results'
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
    U_opencl = np.zeros_like(img1)
    V_opencl = np.zeros_like(img1)
    U_numba = np.zeros_like(img1)
    V_numba = np.zeros_like(img1)
    
    # Common parameters for both implementations
    common_params = {
        'Niter': 5,
        'halfWindow': 7,
        'provideGenericPyramidalDefaults': True,
        'enableVorticityEnhancement': False
    }
    
    # Run OpenCL implementation
    print("\nRunning denseLucasKanade_PyCL implementation...")
    try:
        from denseLucasKanade_PyCL import denseLucasKanade_PyCl
        
        start_time = time.time()
        lk_opencl = denseLucasKanade_PyCl(platformID=0, deviceID=0, **common_params)
        U_opencl, V_opencl, error_opencl = lk_opencl.compute(img1, img2, U_opencl, V_opencl)
        opencl_time = time.time() - start_time
        
        print(f"OpenCL implementation completed in {opencl_time:.2f} seconds")
        print(f"Flow range U: {U_opencl.min():.2f} to {U_opencl.max():.2f}")
        print(f"Flow range V: {V_opencl.min():.2f} to {V_opencl.max():.2f}")
        
        # Calculate mean flow in central region
        border = 10
        central_U_opencl = U_opencl[border:-border, border:-border]
        central_V_opencl = V_opencl[border:-border, border:-border]
        mean_u_opencl = np.mean(central_U_opencl)
        mean_v_opencl = np.mean(central_V_opencl)
        
        print(f"Mean flow U (central region): {mean_u_opencl:.2f} (expected: {expected_u})")
        print(f"Mean flow V (central region): {mean_v_opencl:.2f} (expected: {expected_v})")
        
        # Plot OpenCL results
        plot_images_and_flow(img1, img2, U_opencl, V_opencl, 
                           "Dense Lucas-Kanade OpenCL Flow", 
                           f"{output_dir}/dense_lk_opencl_flow.png")
        
        opencl_success = True
    except Exception as e:
        print(f"Error running OpenCL implementation: {e}")
        opencl_success = False
    
    # Run Numba implementation
    print("\nRunning denseLucasKanade_Numba implementation...")
    from denseLucasKanade_Numba import denseLucasKanade_Numba
    
    start_time = time.time()
    lk_numba = denseLucasKanade_Numba(**common_params)
    U_numba, V_numba, error_numba = lk_numba.compute(img1, img2, U_numba, V_numba)
    numba_time = time.time() - start_time
    
    print(f"Numba implementation completed in {numba_time:.2f} seconds")
    print(f"Flow range U: {U_numba.min():.2f} to {U_numba.max():.2f}")
    print(f"Flow range V: {V_numba.min():.2f} to {V_numba.max():.2f}")
    
    # Calculate mean flow in central region
    border = 10
    central_U_numba = U_numba[border:-border, border:-border]
    central_V_numba = V_numba[border:-border, border:-border]
    mean_u_numba = np.mean(central_U_numba)
    mean_v_numba = np.mean(central_V_numba)
    
    print(f"Mean flow U (central region): {mean_u_numba:.2f} (expected: {expected_u})")
    print(f"Mean flow V (central region): {mean_v_numba:.2f} (expected: {expected_v})")
    
    # Plot Numba results
    plot_images_and_flow(img1, img2, U_numba, V_numba, 
                       "Dense Lucas-Kanade Numba Flow", 
                       f"{output_dir}/dense_lk_numba_flow.png")
    
    # Calculate error from expected flow
    u_error_numba = abs(mean_u_numba - expected_u)
    v_error_numba = abs(mean_v_numba - expected_v)
    
    print(f"\nNumba implementation error from expected flow:")
    print(f"U error: {u_error_numba:.2f}")
    print(f"V error: {v_error_numba:.2f}")
    
    # If OpenCL implementation succeeded, compare results
    if opencl_success:
        # Calculate error from expected flow
        u_error_opencl = abs(mean_u_opencl - expected_u)
        v_error_opencl = abs(mean_v_opencl - expected_v)
        
        print(f"\nOpenCL implementation error from expected flow:")
        print(f"U error: {u_error_opencl:.2f}")
        print(f"V error: {v_error_opencl:.2f}")
        
        # Compare the implementations
        comparison = compare_flows(U_opencl, V_opencl, U_numba, V_numba)
        print("\nComparison of OpenCL and Numba implementations:")
        for key, value in comparison.items():
            print(f"{key}: {value:.6f}")
        
        # Plot comparison
        plot_flow_comparison(U_opencl, V_opencl, U_numba, V_numba, 
                           "OpenCL vs Numba Dense Lucas-Kanade", 
                           f"{output_dir}/dense_lk_comparison.png")
        
        # Determine if implementations are similar
        if (comparison['u_correlation'] > 0.9 and comparison['v_correlation'] > 0.9 and
            comparison['mean_u_diff'] < 1.0 and comparison['mean_v_diff'] < 1.0):
            print("\nCONCLUSION: The implementations produce similar results!")
        else:
            print("\nCONCLUSION: The implementations produce different results.")
            print("This may be due to implementation differences or numerical precision.")
    else:
        print("\nCannot compare implementations because OpenCL implementation failed.")
        print("This is expected if you don't have OpenCL support or a compatible GPU.")
        
        # Determine if Numba implementation is accurate
        threshold = 2.0  # Allow some error due to algorithm limitations
        if u_error_numba < threshold and v_error_numba < threshold:
            print("\nThe Numba implementation correctly detects the expected flow.")
        else:
            print("\nThe Numba implementation does not accurately detect the expected flow.")
            print("This may be due to parameter tuning rather than a fundamental issue.")
    
    print(f"\nResults saved to {output_dir} directory")

if __name__ == "__main__":
    main()
