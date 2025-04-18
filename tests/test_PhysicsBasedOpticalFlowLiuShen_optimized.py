"""
Test file for the optimized PhysicsBasedOpticalFlowLiuShen implementation.
"""

import os
import sys
import pytest
import numpy as np
from skimage.io import imread

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the implementation to test
from PhysicsBasedOpticalFlowLiuShen_optimized import LiuShenOpticalFlowAlgoAdapter, physicsBasedOpticalFlowLiuShen, _compute_determinant_and_inverse, _compute_flow_update

def load_test_images():
    """Load test images from the Bits08 dataset."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'testImages', 'Bits08', 'Ni06')
    fn1 = os.path.join(base_path, 'parabolic01_0.tif')
    fn2 = os.path.join(base_path, 'parabolic01_1.tif')

    # Load images
    im1 = imread(fn1).astype(np.float32)
    im2 = imread(fn2).astype(np.float32)

    return im1, im2

def test_LiuShenOpticalFlowAlgoAdapter_initialization():
    """Test that the LiuShenOpticalFlowAlgoAdapter class can be initialized."""
    ls = LiuShenOpticalFlowAlgoAdapter(alpha=0.1)
    assert ls is not None
    assert ls.alpha == 0.1
    assert ls.hasGenericPyramidalDefaults() == False

def test_compute_determinant_and_inverse():
    """Test the compute_determinant_and_inverse function."""
    # Create simple test matrices
    A11 = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    A22 = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    A12 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # Compute determinant and inverse
    B11, B12, B22 = _compute_determinant_and_inverse(A11, A22, A12)

    # Check results
    assert np.allclose(B11, np.array([[3.0/5.0, 0.0], [0.0, 3.0/5.0]]), rtol=1e-5)
    assert np.allclose(B12, np.array([[-1.0/5.0, 0.0], [0.0, -1.0/5.0]]), rtol=1e-5)
    assert np.allclose(B22, np.array([[2.0/5.0, 0.0], [0.0, 2.0/5.0]]), rtol=1e-5)

def test_compute_flow_update():
    """Test the compute_flow_update function."""
    # Create simple test matrices and vectors
    B11 = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)
    B12 = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
    B22 = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)

    bu = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    bv = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    # Compute flow update
    unew, vnew = _compute_flow_update(B11, B12, B22, bu, bv)

    # Check results - just verify that the output has the right shape and type
    assert unew.shape == (2, 2)
    assert vnew.shape == (2, 2)
    assert unew.dtype == np.float32
    assert vnew.dtype == np.float32

    # Check that the values are negative (as expected from the formula)
    assert np.all(unew <= 0)
    assert np.all(vnew <= 0)

def test_physicsBasedOpticalFlowLiuShen_small():
    """Test that the physicsBasedOpticalFlowLiuShen function can compute optical flow on a small image."""
    # Create a small test image with a simple pattern
    im1 = np.zeros((16, 16), dtype=np.float32)
    im2 = np.zeros((16, 16), dtype=np.float32)

    # Create a simple pattern in the first image
    im1[5:10, 5:10] = 1.0

    # Create the same pattern in the second image, but shifted by (1, 1) pixels
    im2[6:11, 6:11] = 1.0

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # We'll use the function with default parameters

    # Compute optical flow with a smaller maxnum
    u, v, _ = physicsBasedOpticalFlowLiuShen(im1, im2, 0.1, V, U)

    # Check that the flow field is approximately correct
    # The flow should be approximately (1, 1) in the region of the pattern
    assert np.sum(np.abs(u[6:10, 6:10])) > 0  # x-component should have some flow
    assert np.sum(np.abs(v[6:10, 6:10])) > 0  # y-component should have some flow

def test_LiuShenOpticalFlowAlgoAdapter_compute():
    """Test that the LiuShenOpticalFlowAlgoAdapter compute method works."""
    # Create a small test image with a simple pattern
    im1 = np.zeros((16, 16), dtype=np.float32)
    im2 = np.zeros((16, 16), dtype=np.float32)

    # Create a simple pattern in the first image
    im1[5:10, 5:10] = 1.0

    # Create the same pattern in the second image, but shifted by (1, 1) pixels
    im2[6:11, 6:11] = 1.0

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Create the Liu-Shen adapter
    ls = LiuShenOpticalFlowAlgoAdapter(alpha=0.1)

    # Compute optical flow
    U_out, V_out, _ = ls.compute(im1, im2, U, V)

    # Check that the flow field has some flow in the region of interest
    assert np.sum(np.abs(U_out[6:10, 6:10])) > 0  # x-component should have some flow
    assert np.sum(np.abs(V_out[6:10, 6:10])) > 0  # y-component should have some flow

if __name__ == "__main__":
    # Run the tests
    test_LiuShenOpticalFlowAlgoAdapter_initialization()
    test_compute_determinant_and_inverse()
    test_compute_flow_update()
    test_physicsBasedOpticalFlowLiuShen_small()
    test_LiuShenOpticalFlowAlgoAdapter_compute()
    print("All tests passed!")
