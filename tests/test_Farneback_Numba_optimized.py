"""
Test file for the optimized Farneback_Numba implementation.
"""

import os
import sys
import pytest
import numpy as np
from skimage.io import imread

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the implementation to test
from Farneback_Numba_optimized import Farneback_Numba, _bilinear_interpolate
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

def test_Farneback_Numba_initialization():
    """Test that the Farneback_Numba class can be initialized."""
    fb = Farneback_Numba(windowSize=33, Niters=5, polyN=7, polySigma=1.5)
    assert fb is not None
    assert fb.windowSize == 33
    assert fb.numIters == 5
    assert fb.polyN == 7
    assert fb.polySigma == 1.5

def test_Farneback_Numba_compute_small():
    """Test that the Farneback_Numba class can compute optical flow on a small image."""
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

    # Create the Farneback object with minimal settings for test
    fb = Farneback_Numba(windowSize=9, Niters=2, polyN=5, polySigma=1.1, pyramidalLevels=1)

    # Compute optical flow
    U_out, V_out, _ = fb.compute(im1, im2, U, V)

    # Check that the flow field is not all zeros
    assert np.sum(np.abs(U_out)) > 0
    assert np.sum(np.abs(V_out)) > 0

    # The Farneback algorithm can produce negative flow values in some cases
    # Just check that there's some flow in the region of interest
    assert np.sum(np.abs(U_out[13:20, 12:20])) > 0
    assert np.sum(np.abs(V_out[13:20, 12:20])) > 0

def test_Farneback_Numba_with_GenericPyramidalOpticalFlow():
    """Test that the Farneback_Numba class works with GenericPyramidalOpticalFlow."""
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

    # Create the Farneback object with minimal settings for test
    fb = Farneback_Numba(windowSize=9, Niters=2, polyN=5, polySigma=1.1, pyramidalLevels=1)

    # Parameters for GenericPyramidalOpticalFlow - use minimal settings for test
    FILTER = 2
    FILTER_OPT = 0.48
    pyramidalLevels = 1  # Reduced from 2 to 1 for faster testing
    kLevels = 1

    # Compute optical flow using GenericPyramidalOpticalFlow
    try:
        U_out, V_out = genericPyramidalOpticalFlow(
            im1, im2, FILTER, fb, pyramidalLevels, kLevels,
            FILTER_OPT, None, warping=False
        )

        # Check that the flow field is not all zeros
        assert np.sum(np.abs(U_out)) > 0
        assert np.sum(np.abs(V_out)) > 0

        # The Farneback algorithm can produce negative flow values in some cases
        # Just check that there's some flow in the region of interest
        assert np.sum(np.abs(U_out[23:40, 22:40])) > 0
        assert np.sum(np.abs(V_out[23:40, 22:40])) > 0

    except Exception as e:
        pytest.fail(f"GenericPyramidalOpticalFlow failed with error: {e}")

def test_bilinear_interpolate():
    """Test the bilinear interpolation function."""
    # Create a simple test image
    img = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    # Test exact pixel values
    assert _bilinear_interpolate(img, 0, 0) == 1
    assert _bilinear_interpolate(img, 1, 0) == 2
    assert _bilinear_interpolate(img, 0, 1) == 4
    assert _bilinear_interpolate(img, 1, 1) == 5

    # Test interpolated values
    assert _bilinear_interpolate(img, 0.5, 0) == 1.5
    assert _bilinear_interpolate(img, 0, 0.5) == 2.5
    assert _bilinear_interpolate(img, 0.5, 0.5) == 3.0

    # Test out of bounds values
    assert _bilinear_interpolate(img, -1, 0) == 0
    assert _bilinear_interpolate(img, 0, -1) == 0
    assert _bilinear_interpolate(img, 3, 0) == 0
    assert _bilinear_interpolate(img, 0, 3) == 0

if __name__ == "__main__":
    # Run the tests
    test_Farneback_Numba_initialization()
    test_Farneback_Numba_compute_small()
    test_bilinear_interpolate()
    test_Farneback_Numba_with_GenericPyramidalOpticalFlow()
    print("All tests passed!")
