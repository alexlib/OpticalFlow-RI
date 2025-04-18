#!/usr/bin/env python
"""
Test script for Farneback Numba implementation with Liu-Shen enhancement
"""

import sys
import os
sys.path.insert(0, os.path.join('.', 'src'))
import numpy as np
from skimage.data import camera
import matplotlib.pyplot as plt
from GenericPyramidalOpticalFlowWrapper import GenericPyramidalOpticalFlowWrapper
from PhysicsBasedOpticalFlowLiuShen import LiuShenOpticalFlowAlgoAdapter
from Farneback_Numba import Farneback_Numba

def plot_flow(U, V, title):
    """Plot optical flow as a quiver plot"""
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
    output_dir = 'farneback_numba_test_results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png", dpi=150)
    plt.close()

def main():
    # Create a simple test case with a small image
    print("Creating test images...")
    img1 = camera()  # Use a standard test image
    img1 = img1.astype(np.float32)
    
    # Create a slightly shifted version of the image
    shift_x = 2
    shift_y = 3
    img2 = np.zeros_like(img1)
    img2[shift_y:, shift_x:] = img1[:-shift_y, :-shift_x]
    
    # Resize to make it even smaller and faster
    small_shape = (100, 100)
    from skimage.transform import resize
    img1_small = resize(img1, small_shape, preserve_range=True).astype(np.float32)
    img2_small = resize(img2, small_shape, preserve_range=True).astype(np.float32)
    
    print(f"Test images created with shape {img1_small.shape}")
    print(f"Ground truth shift: ({shift_x}, {shift_y})")
    
    # Initialize flow fields
    U = np.zeros_like(img1_small)
    V = np.zeros_like(img1_small)
    
    # Create Farneback algorithm with small parameters for quick testing
    print("\nInitializing Farneback Numba algorithm...")
    fb_adapter = Farneback_Numba(
        windowSize=7,
        Niters=2,
        polyN=5,
        polySigma=1.1,
        pyramidalLevels=1
    )
    
    # Create pyramidal optical flow wrapper
    print("\nRunning Farneback without Liu-Shen enhancement...")
    pyr_of = GenericPyramidalOpticalFlowWrapper(
        fb_adapter,
        filter_sigma=0.0,
        pyr_levels=1
    )
    
    # Run the algorithm
    U, V = pyr_of.calculateFlow(img1_small, img2_small)
    
    # Print results
    print("\nFarneback Results:")
    print(f"Mean flow U: {np.mean(U):.4f} (expected negative value around -{shift_x})")
    print(f"Mean flow V: {np.mean(V):.4f} (expected negative value around -{shift_y})")
    print(f"Flow range U: {U.min():.2f} to {U.max():.2f}")
    print(f"Flow range V: {V.min():.2f} to {V.max():.2f}")
    
    # Plot the flow
    plot_flow(U, V, "Farneback Numba Flow")
    
    # Run Farneback with Liu-Shen enhancement
    print("\nRunning Farneback with Liu-Shen enhancement...")
    
    # Create Liu-Shen adapter
    ls_adapter = LiuShenOpticalFlowAlgoAdapter(0.1)  # Alpha parameter for regularization
    
    # Create pyramidal optical flow wrapper
    pyr_of = GenericPyramidalOpticalFlowWrapper(
        fb_adapter,
        filter_sigma=0.0,
        pyr_levels=1,
        optional_algo_adapter=ls_adapter,
        filter_opt=0.48  # Add filter_opt parameter for Liu-Shen
    )
    
    # Run the algorithm
    U_ls, V_ls = pyr_of.calculateFlow(img1_small, img2_small)
    
    # Print results
    print("\nFarneback with Liu-Shen Results:")
    print(f"Mean flow U: {np.mean(U_ls):.4f} (expected negative value around -{shift_x})")
    print(f"Mean flow V: {np.mean(V_ls):.4f} (expected negative value around -{shift_y})")
    print(f"Flow range U: {U_ls.min():.2f} to {U_ls.max():.2f}")
    print(f"Flow range V: {V_ls.min():.2f} to {V_ls.max():.2f}")
    
    # Plot the flow
    plot_flow(U_ls, V_ls, "Farneback Numba with Liu-Shen Flow")
    
    # Plot difference between Liu-Shen and regular Farneback
    diff_U = U_ls - U
    diff_V = V_ls - V
    
    print("\nDifference (Liu-Shen - Regular):")
    print(f"Mean diff U: {np.mean(diff_U):.4f}")
    print(f"Mean diff V: {np.mean(diff_V):.4f}")
    print(f"Diff range U: {diff_U.min():.2f} to {diff_U.max():.2f}")
    print(f"Diff range V: {diff_V.min():.2f} to {diff_V.max():.2f}")
    
    # Plot the difference
    plot_flow(diff_U, diff_V, "Difference (Liu-Shen - Regular)")
    
    print("\nTest completed successfully!")
    print(f"Results saved to farneback_numba_test_results directory")

if __name__ == "__main__":
    main()
