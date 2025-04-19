"""
Test file for the optimized PhysicsBasedOpticalFlowLiuShen_numba implementation.
"""

import os
import sys
import pytest
import numpy as np
from skimage.io import imread
from scipy.ndimage import convolve as filter2

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import the implementation to test
from PhysicsBasedOpticalFlowLiuShen_numba import physicsBasedOpticalFlowLiuShen, LiuShenOpticalFlowAlgoAdapter

def test_liushen_small_image():
    """Test that the Liu-Shen algorithm works on a small test image."""
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

    # Set parameters
    h = 0.1  # Regularization parameter

    # Create a wrapper function with limited iterations
    def liushen_wrapper(im1, im2, h, U, V, max_iterations=5):
        # Override maxnum in the function body
        import types
        import PhysicsBasedOpticalFlowLiuShen_numba

        # Save the original function
        original_func = PhysicsBasedOpticalFlowLiuShen_numba.physicsBasedOpticalFlowLiuShen

        # Create a modified version of the function
        def modified_func(im1, im2, h, U, V):
            # Get the source code of the original function
            source = original_func.__code__

            # Create a new function with modified maxnum
            new_func = types.FunctionType(
                source,
                original_func.__globals__.copy(),
                original_func.__name__,
                original_func.__defaults__,
                original_func.__closure__
            )

            # Override maxnum in the globals
            new_func.__globals__['maxnum'] = max_iterations

            # Call the modified function
            return new_func(im1, im2, h, U, V)

        # Call the modified function
        return modified_func(im1, im2, h, U, V)

    # Run the algorithm with limited iterations
    u, v, err = liushen_wrapper(im1, im2, h, U, V, max_iterations=5)

    # Check that the flow field is approximately correct
    # The flow should be approximately (2, 3) in the region of the pattern
    assert np.mean(u[13:20, 12:20]) > 0.5  # x-component should be positive
    assert np.mean(v[13:20, 12:20]) > 0.5  # y-component should be positive

def test_liushen_adapter():
    """Test that the Liu-Shen adapter class works correctly."""
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

    # Set parameters
    h = 0.1  # Regularization parameter

    # Create adapter instance
    adapter = LiuShenOpticalFlowAlgoAdapter(h)

    # Create a wrapper function with limited iterations
    def adapter_compute_wrapper(im1, im2, U, V, max_iterations=5):
        # Override maxnum in the function body
        import types
        import PhysicsBasedOpticalFlowLiuShen_numba

        # Save the original function
        original_func = PhysicsBasedOpticalFlowLiuShen_numba.physicsBasedOpticalFlowLiuShen

        # Create a modified version of the function
        def modified_func(im1, im2, h, U, V):
            # Get the source code of the original function
            source = original_func.__code__

            # Create a new function with modified maxnum
            new_func = types.FunctionType(
                source,
                original_func.__globals__.copy(),
                original_func.__name__,
                original_func.__defaults__,
                original_func.__closure__
            )

            # Override maxnum in the globals
            new_func.__globals__['maxnum'] = max_iterations

            # Call the modified function
            return new_func(im1, im2, h, U, V)

        # Temporarily replace the original function
        PhysicsBasedOpticalFlowLiuShen_numba.physicsBasedOpticalFlowLiuShen = modified_func

        # Call the adapter's compute method
        result = adapter.compute(im1, im2, U, V)

        # Restore the original function
        PhysicsBasedOpticalFlowLiuShen_numba.physicsBasedOpticalFlowLiuShen = original_func

        return result

    # Run the adapter with limited iterations
    u, v, err = adapter_compute_wrapper(im1, im2, U, V, max_iterations=5)

    # Check that the flow field is approximately correct
    # The flow should be approximately (2, 3) in the region of the pattern
    assert np.mean(u[13:20, 12:20]) > 0.5  # x-component should be positive
    assert np.mean(v[13:20, 12:20]) > 0.5  # y-component should be positive

    # Check adapter properties
    assert adapter.getAlgoName() == 'Liu-Shen Physics based OF'
    assert adapter.hasGenericPyramidalDefaults() == False

def test_liushen_parabolic():
    """Test that the Liu-Shen algorithm works on parabolic flow images."""
    # Load test images
    test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testImages', 'Bits08', 'Ni06'))
    im1_path = os.path.join(test_dir, 'parabolic01_0.tif')
    im2_path = os.path.join(test_dir, 'parabolic01_1.tif')

    # Check if test images exist
    if not os.path.exists(im1_path) or not os.path.exists(im2_path):
        pytest.skip("Test images not found")

    # Load images
    im1 = imread(im1_path).astype(np.float32)
    im2 = imread(im2_path).astype(np.float32)

    # Downsample images for faster testing
    im1 = im1[::8, ::8]
    im2 = im2[::8, ::8]

    # Initialize flow fields
    U = np.zeros_like(im1)
    V = np.zeros_like(im1)

    # Set parameters
    h = 0.1  # Regularization parameter

    # Create a wrapper function with limited iterations
    def liushen_wrapper(im1, im2, h, U, V, max_iterations=3):
        # Override maxnum in the function body
        import types
        import PhysicsBasedOpticalFlowLiuShen_numba

        # Save the original function
        original_func = PhysicsBasedOpticalFlowLiuShen_numba.physicsBasedOpticalFlowLiuShen

        # Create a modified version of the function
        def modified_func(im1, im2, h, U, V):
            # Get the source code of the original function
            source = original_func.__code__

            # Create a new function with modified maxnum
            new_func = types.FunctionType(
                source,
                original_func.__globals__.copy(),
                original_func.__name__,
                original_func.__defaults__,
                original_func.__closure__
            )

            # Override maxnum in the globals
            new_func.__globals__['maxnum'] = max_iterations

            # Call the modified function
            return new_func(im1, im2, h, U, V)

        # Call the modified function
        return modified_func(im1, im2, h, U, V)

    # Run the algorithm with more iterations to get a proper parabolic profile
    u, v, err = liushen_wrapper(im1, im2, h, U, V, max_iterations=10)

    # Check that the flow field is not all zeros
    assert np.sum(np.abs(u)) > 0
    assert np.sum(np.abs(v)) > 0

    # Check that the flow field has reasonable values
    # For parabolic flow, we expect non-zero horizontal flow
    # and minimal vertical flow
    assert np.mean(np.abs(u)) > 0.001  # Mean absolute horizontal flow should be significant
    assert np.mean(np.abs(v)) < np.mean(np.abs(u))  # Vertical flow should be less than horizontal flow

if __name__ == "__main__":
    # Run the tests
    test_liushen_small_image()
    test_liushen_adapter()
    test_liushen_parabolic()
    print("All tests passed!")
