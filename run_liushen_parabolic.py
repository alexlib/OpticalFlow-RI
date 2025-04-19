"""
Run the Liu-Shen physics-based optical flow algorithm on parabolic flow images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import scipy.io as sio
from tqdm import tqdm

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the implementation
from PhysicsBasedOpticalFlowLiuShen_numba import physicsBasedOpticalFlowLiuShen

def run_liushen_parabolic():
    """Run the Liu-Shen algorithm on parabolic flow images."""
    # Load test images
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'testImages', 'Bits08', 'Ni06'))
    im1_path = os.path.join(test_dir, 'parabolic01_0.tif')
    im2_path = os.path.join(test_dir, 'parabolic01_1.tif')
    
    # Check if test images exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        print("Test images not found")
        return
    
    # Load images
    im1 = imread(im1_path).astype(np.float32)
    im2 = imread(im2_path).astype(np.float32)
    
    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)
    
    # Set parameters
    h = 0.1  # Regularization parameter
    
    # Run the algorithm
    u, v, err = physicsBasedOpticalFlowLiuShen(im1, im2, h, U, V)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'flow_results', 'liushen_parabolic')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    results = {
        'u': u,
        'v': v,
        'error': err
    }
    sio.savemat(os.path.join(output_dir, 'flow_results.mat'), results)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot horizontal velocity field
    plt.subplot(2, 2, 1)
    plt.imshow(u, cmap='jet')
    plt.colorbar()
    plt.title('Horizontal Velocity (U)')
    
    # Plot vertical velocity field
    plt.subplot(2, 2, 2)
    plt.imshow(v, cmap='jet')
    plt.colorbar()
    plt.title('Vertical Velocity (V)')
    
    # Plot quiver plot
    plt.subplot(2, 2, 3)
    step = 16  # Skip every 16 pixels for clarity
    Y, X = np.mgrid[0:u.shape[0]:step, 0:u.shape[1]:step]
    U_quiver = u[::step, ::step]
    V_quiver = v[::step, ::step]
    plt.quiver(X, Y, U_quiver, V_quiver, scale=5)
    plt.title('Flow Field')
    
    # Plot horizontal velocity profile
    plt.subplot(2, 2, 4)
    u_profile = np.mean(u, axis=1)  # Average across columns
    plt.plot(u_profile, np.arange(len(u_profile)))
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.title('Horizontal Velocity Profile')
    plt.xlabel('Velocity')
    plt.ylabel('Y-coordinate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'flow_results.png'))
    plt.show()
    
    print(f"Results saved to {output_dir}")
    
    return u, v, err

if __name__ == "__main__":
    run_liushen_parabolic()
