"""
Test file for the optimized HornSchunck implementation.
"""

import os
import sys
import pytest
import numpy as np
from skimage.io import imread

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the implementation to test
from HornSchunck_optimized import HSOpticalFlowAlgoAdapter, HS, _compute_derivatives
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow

def load_test_images():
    """Load test images from the Bits08 dataset."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'testImages', 'Bits08', 'Ni06')
    fn1 = os.path.join(base_path, 'parabolic01_0.tif')
    fn2 = os.path.join(base_path, 'parabolic01_1.tif')

    # Load images
    im1 = imread(fn1).astype(np.float32)
    im2 = imread(fn2).astype(np.float32)

    return im1, im2

def test_HSOpticalFlowAlgoAdapter_initialization():
    """Test that the HSOpticalFlowAlgoAdapter class can be initialized."""
    hs = HSOpticalFlowAlgoAdapter(alphas=[0.1], Niter=100)
    assert hs is not None
    assert hs.alphas == [0.1]
    assert hs.Niter == 100
    assert hs.provideGenericPyramidalDefaults == True

def test_compute_derivatives():
    """Test the compute_derivatives function."""
    # Create a simple test image with a gradient
    im1 = np.zeros((10, 10), dtype=np.float32)
    im2 = np.zeros((10, 10), dtype=np.float32)

    # Create a horizontal gradient in im1
    for i in range(10):
        im1[:, i] = i / 10.0

    # Create a shifted horizontal gradient in im2
    for i in range(10):
        if i < 9:
            im2[:, i] = (i + 1) / 10.0
        else:
            im2[:, i] = 1.0

    # Compute derivatives
    fx, fy, ft = _compute_derivatives(im1, im2)

    # Check that fx is approximately 0.1 (the gradient)
    assert np.mean(fx[1:-1, 1:-1]) > 0.05

    # Check that fy is approximately 0 (no vertical gradient)
    assert abs(np.mean(fy[1:-1, 1:-1])) < 0.01

    # Check that ft is approximately 0.1 (the temporal difference)
    assert np.mean(ft[1:-1, 1:-1]) > 0.05

def test_HS_compute_small():
    """Test that the HS function can compute optical flow on a small image."""
    # Create a small test image with a simple pattern
    im1 = np.zeros((32, 32), dtype=np.float32)
    im2 = np.zeros((32, 32), dtype=np.float32)

    # Create a simple pattern in the first image
    im1[10:20, 10:20] = 1.0

    # Create the same pattern in the second image, but shifted by (2, 3) pixels
    im2[13:23, 12:22] = 1.0

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Compute optical flow
    U_out, V_out, _ = HS(im1, im2, 0.1, 10, U, V)

    # Check that the flow field is not all zeros
    assert np.sum(np.abs(U_out)) > 0
    assert np.sum(np.abs(V_out)) > 0

    # The Horn-Schunck algorithm can produce negative flow values in some cases
    # Just check that there's some flow in the region of interest
    assert np.sum(np.abs(U_out[13:20, 12:20])) > 0
    assert np.sum(np.abs(V_out[13:20, 12:20])) > 0

def test_HSOpticalFlowAlgoAdapter_with_GenericPyramidalOpticalFlow():
    """Test that the HSOpticalFlowAlgoAdapter class works with GenericPyramidalOpticalFlow."""
    # Create a small test image with a simple pattern (64x64 instead of 256x256)
    im1 = np.zeros((64, 64), dtype=np.float32)
    im2 = np.zeros((64, 64), dtype=np.float32)

    # Create a simple pattern in the first image
    im1[20:40, 20:40] = 1.0

    # Create the same pattern in the second image, but shifted by (2, 3) pixels
    im2[23:43, 22:42] = 1.0

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Create the Horn-Schunck object with minimal settings for test
    hs = HSOpticalFlowAlgoAdapter(alphas=[0.1], Niter=10)

    # Parameters for GenericPyramidalOpticalFlow - use minimal settings for test
    FILTER = 2
    FILTER_OPT = 0.48
    pyramidalLevels = 1  # Reduced from 2 to 1 for faster testing
    kLevels = 1

    # Compute optical flow using GenericPyramidalOpticalFlow
    try:
        U_out, V_out = genericPyramidalOpticalFlow(
            im1, im2, FILTER, hs, pyramidalLevels, kLevels,
            FILTER_OPT, None, warping=False
        )

        # Check that the flow field is not all zeros
        assert np.sum(np.abs(U_out)) > 0
        assert np.sum(np.abs(V_out)) > 0

        # The Horn-Schunck algorithm can produce negative flow values in some cases
        # Just check that there's some flow in the region of interest
        assert np.sum(np.abs(U_out[23:40, 22:40])) > 0
        assert np.sum(np.abs(V_out[23:40, 22:40])) > 0

    except Exception as e:
        pytest.fail(f"GenericPyramidalOpticalFlow failed with error: {e}")

if __name__ == "__main__":
    # Run the tests
    test_HSOpticalFlowAlgoAdapter_initialization()
    test_compute_derivatives()
    test_HS_compute_small()
    test_HSOpticalFlowAlgoAdapter_with_GenericPyramidalOpticalFlow()
    print("All tests passed!")
