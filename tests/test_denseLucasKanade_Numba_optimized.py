"""
Test file for the optimized denseLucasKanade_Numba implementation.
"""

import os
import sys
import pytest
import numpy as np
from skimage.io import imread

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the implementation to test
from denseLucasKanade_Numba_optimized import denseLucasKanade_Numba, _bilinear_interpolate
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow

def test_denseLucasKanade_Numba_initialization():
    """Test that the denseLucasKanade_Numba class can be initialized."""
    lk = denseLucasKanade_Numba(Niter=5, halfWindow=13)
    assert lk is not None
    assert lk.Niter == 5
    assert lk.windowHalfWidth == 13
    assert lk.windowHalfHeight == 13
    assert lk.windowWidth == 27
    assert lk.windowHeight == 27

def test_denseLucasKanade_Numba_compute_small():
    """Test that the denseLucasKanade_Numba class can compute optical flow on a small image."""
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
    
    # Create the Lucas-Kanade object
    lk = denseLucasKanade_Numba(Niter=5, halfWindow=5)
    
    # Compute optical flow
    U_out, V_out, _ = lk.compute(im1, im2, U, V)
    
    # Check that the flow field is approximately correct
    # The flow should be approximately (2, 3) in the region of the pattern
    assert np.mean(U_out[13:20, 12:20]) > 0.5  # x-component should be positive
    assert np.mean(V_out[13:20, 12:20]) > 0.5  # y-component should be positive

def test_denseLucasKanade_Numba_with_GenericPyramidalOpticalFlow():
    """Test that the denseLucasKanade_Numba class works with GenericPyramidalOpticalFlow."""
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
    
    # Create the Lucas-Kanade object with smaller window for faster computation
    lk = denseLucasKanade_Numba(Niter=3, halfWindow=5)
    
    # Parameters for GenericPyramidalOpticalFlow - use minimal settings for test
    FILTER = 2
    FILTER_OPT = 0.48
    pyramidalLevels = 1  # Reduced from 2 to 1 for faster testing
    kLevels = 1
    
    # Compute optical flow using GenericPyramidalOpticalFlow
    try:
        U_out, V_out = genericPyramidalOpticalFlow(
            im1, im2, FILTER, lk, pyramidalLevels, kLevels, 
            FILTER_OPT, None, warping=False
        )
        
        # Check that the flow field is not all zeros
        assert np.sum(np.abs(U_out)) > 0
        assert np.sum(np.abs(V_out)) > 0
        
        # Check that the flow field has reasonable values in the region of interest
        # The flow should be approximately (2, 3) in the region of the pattern
        assert np.mean(U_out[23:40, 22:40]) > 0.5  # x-component should be positive
        assert np.mean(V_out[23:40, 22:40]) > 0.5  # y-component should be positive
        
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
    test_denseLucasKanade_Numba_initialization()
    test_denseLucasKanade_Numba_compute_small()
    test_bilinear_interpolate()
    test_denseLucasKanade_Numba_with_GenericPyramidalOpticalFlow()
    print("All tests passed!")
