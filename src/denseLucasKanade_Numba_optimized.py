"""
MIT LicenseCopyright (c) [2021-2024] [Luís Mendes, luis <dot> mendes _at_ tecnico.ulisboa.pt]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:The above copyright notice and
this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from scipy.ndimage import convolve
from numba import njit, prange

# Standalone Numba-optimized functions
@njit(fastmath=True)
def _bilinear_interpolate(img, x, y):
    """Bilinear interpolation for float coordinates"""
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1
    y1 = y0 + 1

    # Get weights
    wx = x - x0
    wy = y - y0

    # Get pixel values with bounds checking
    v00 = img[y0, x0] if 0 <= y0 < img.shape[0] and 0 <= x0 < img.shape[1] else 0
    v01 = img[y0, x1] if 0 <= y0 < img.shape[0] and 0 <= x1 < img.shape[1] else 0
    v10 = img[y1, x0] if 0 <= y1 < img.shape[0] and 0 <= x0 < img.shape[1] else 0
    v11 = img[y1, x1] if 0 <= y1 < img.shape[0] and 0 <= x1 < img.shape[1] else 0

    # Interpolate
    return (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v01 + (1 - wx) * wy * v10 + wx * wy * v11

@njit(fastmath=True)
def _compute_flow_vector(img1, img2, x, y, window_half_width, window_half_height,
                       max_iterations, threshold,
                       assymetricWindowLeft, assymetricWindowRight, assymetricWindowTop, assymetricWindowBottom,
                       initial_flow_x, initial_flow_y, calc_err):
    """Compute optical flow vector at point (x, y)"""
    # Initial point with flow
    prev_x = x + initial_flow_x
    prev_y = y + initial_flow_y

    # Compute patch matrices directly instead of using a separate function
    # Initialize variables
    A11 = 0.0
    A12 = 0.0
    A22 = 0.0

    # Patch dimensions
    patch_width = 2 * window_half_width + 1
    patch_height = 2 * window_half_height + 1

    # Allocate arrays for patch data
    I_patch = np.zeros((patch_height, patch_width), dtype=np.float32)
    dx_patch = np.zeros((patch_height, patch_width), dtype=np.float32)
    dy_patch = np.zeros((patch_height, patch_width), dtype=np.float32)

    # Fill patch data
    for y_offset in range(-window_half_height, window_half_height + 1):
        for x_offset in range(-window_half_width, window_half_width + 1):
            # Apply asymmetric window weights
            weight = 1.0

            # Apply asymmetric window constraints
            if x_offset == -window_half_width and assymetricWindowLeft > 0:
                weight = 1.0 - assymetricWindowLeft
            if x_offset == window_half_width and assymetricWindowRight > 0:
                weight = 0.0
            if y_offset == -window_half_height and assymetricWindowTop > 0:
                weight = 1.0 - assymetricWindowTop
            if y_offset == window_half_height and assymetricWindowBottom > 0:
                weight = 0.0

            if weight > 0:
                # Get pixel coordinates
                px = x + x_offset
                py = y + y_offset

                # Check if within image bounds
                if 0 <= py < img1.shape[0] and 0 <= px < img1.shape[1]:
                    # Store pixel value
                    patch_y = y_offset + window_half_height
                    patch_x = x_offset + window_half_width
                    I_patch[patch_y, patch_x] = img1[py, px]

                    # Compute derivatives if not at border
                    if 1 <= py < img1.shape[0] - 1 and 1 <= px < img1.shape[1] - 1:
                        # Use Sobel-like operator (3x10 + 3) as in the OpenCL implementation
                        dx = 3.0 * (img1[py-1, px+1] + img1[py+1, px+1] - img1[py-1, px-1] - img1[py+1, px-1]) + 10.0 * (img1[py, px+1] - img1[py, px-1])
                        dy = 3.0 * (img1[py+1, px-1] + img1[py+1, px+1] - img1[py-1, px-1] - img1[py-1, px+1]) + 10.0 * (img1[py+1, px] - img1[py-1, px])

                        dx_patch[patch_y, patch_x] = dx * weight
                        dy_patch[patch_y, patch_x] = dy * weight

                        # Update matrices
                        A11 += dx * dx * weight
                        A12 += dx * dy * weight
                        A22 += dy * dy * weight

    # Check if matrix is invertible
    D = A11 * A22 - A12 * A12
    if D < 1.192092896e-07:
        return initial_flow_x, initial_flow_y, 0.0, False

    # Invert matrix
    A11 /= D
    A12 /= D
    A22 /= D

    # Iterative refinement
    for k in range(max_iterations):
        # Check if point is within image bounds
        if (prev_x - window_half_width < 0 or prev_x + window_half_width >= img2.shape[1] or
            prev_y - window_half_height < 0 or prev_y + window_half_height >= img2.shape[0]):
            return initial_flow_x, initial_flow_y, 0.0, False

        # Compute image mismatch
        b1 = 0.0
        b2 = 0.0

        for y_offset in range(-window_half_height, window_half_height + 1):
            for x_offset in range(-window_half_width, window_half_width + 1):
                # Apply asymmetric window weights
                weight = 1.0

                # Apply asymmetric window constraints
                if x_offset == -window_half_width and assymetricWindowLeft > 0:
                    weight = 1.0 - assymetricWindowLeft
                if x_offset == window_half_width and assymetricWindowRight > 0:
                    weight = 0.0
                if y_offset == -window_half_height and assymetricWindowTop > 0:
                    weight = 1.0 - assymetricWindowTop
                if y_offset == window_half_height and assymetricWindowBottom > 0:
                    weight = 0.0

                if weight > 0:
                    # Get patch coordinates
                    patch_y = y_offset + window_half_height
                    patch_x = x_offset + window_half_width

                    # Get pixel value from second image using bilinear interpolation
                    interp_x = prev_x + x_offset
                    interp_y = prev_y + y_offset
                    
                    # Inline bilinear interpolation for speed
                    x0 = int(interp_x)
                    y0 = int(interp_y)
                    x1 = x0 + 1
                    y1 = y0 + 1
                    
                    # Get weights
                    wx = interp_x - x0
                    wy = interp_y - y0
                    
                    # Get pixel values with bounds checking
                    v00 = img2[y0, x0] if 0 <= y0 < img2.shape[0] and 0 <= x0 < img2.shape[1] else 0
                    v01 = img2[y0, x1] if 0 <= y0 < img2.shape[0] and 0 <= x1 < img2.shape[1] else 0
                    v10 = img2[y1, x0] if 0 <= y1 < img2.shape[0] and 0 <= x0 < img2.shape[1] else 0
                    v11 = img2[y1, x1] if 0 <= y1 < img2.shape[0] and 0 <= x1 < img2.shape[1] else 0
                    
                    # Interpolate
                    img2_val = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v01 + (1 - wx) * wy * v10 + wx * wy * v11

                    # Compute difference
                    diff = (img2_val - I_patch[patch_y, patch_x]) * weight

                    # Update b vector
                    b1 += diff * dx_patch[patch_y, patch_x]
                    b2 += diff * dy_patch[patch_y, patch_x]

        # Compute flow update
        delta_x = (A12 * b2 - A22 * b1) * 32.0
        delta_y = (A12 * b1 - A11 * b2) * 32.0

        # Update flow
        prev_x += delta_x
        prev_y += delta_y

        # Check convergence
        if abs(delta_x) < threshold and abs(delta_y) < threshold:
            break

    # Compute error if needed
    error = 0.0
    if calc_err:
        total_weight = 0.0

        for y_offset in range(-window_half_height, window_half_height + 1):
            for x_offset in range(-window_half_width, window_half_width + 1):
                # Apply asymmetric window weights
                weight = 1.0

                # Apply asymmetric window constraints
                if x_offset == -window_half_width and assymetricWindowLeft > 0:
                    weight = 1.0 - assymetricWindowLeft
                if x_offset == window_half_width and assymetricWindowRight > 0:
                    weight = 0.0
                if y_offset == -window_half_height and assymetricWindowTop > 0:
                    weight = 1.0 - assymetricWindowTop
                if y_offset == window_half_height and assymetricWindowBottom > 0:
                    weight = 0.0

                if weight > 0:
                    # Get patch coordinates
                    patch_y = y_offset + window_half_height
                    patch_x = x_offset + window_half_width

                    # Get pixel value from second image using bilinear interpolation
                    interp_x = prev_x + x_offset
                    interp_y = prev_y + y_offset
                    
                    # Inline bilinear interpolation for speed
                    x0 = int(interp_x)
                    y0 = int(interp_y)
                    x1 = x0 + 1
                    y1 = y0 + 1
                    
                    # Get weights
                    wx = interp_x - x0
                    wy = interp_y - y0
                    
                    # Get pixel values with bounds checking
                    v00 = img2[y0, x0] if 0 <= y0 < img2.shape[0] and 0 <= x0 < img2.shape[1] else 0
                    v01 = img2[y0, x1] if 0 <= y0 < img2.shape[0] and 0 <= x1 < img2.shape[1] else 0
                    v10 = img2[y1, x0] if 0 <= y1 < img2.shape[0] and 0 <= x0 < img2.shape[1] else 0
                    v11 = img2[y1, x1] if 0 <= y1 < img2.shape[0] and 0 <= x1 < img2.shape[1] else 0
                    
                    # Interpolate
                    img2_val = (1 - wx) * (1 - wy) * v00 + wx * (1 - wy) * v01 + (1 - wx) * wy * v10 + wx * wy * v11

                    # Compute difference (using same scaling as OpenCL implementation)
                    img1_val = I_patch[patch_y, patch_x]
                    diff = ((img2_val * 16384 + 256) / 512) - ((img1_val * 16384 + 256) / 512)

                    # Update error
                    error += abs(diff) * weight
                    total_weight += weight

        # Normalize error
        if total_weight > 0:
            error /= (32.0 * total_weight)

    # Return flow vector and status
    return prev_x - x, prev_y - y, error, True

@njit(parallel=True)
def _compute_flow_parallel(im1, im2, U_init, V_init, height, width, 
                         window_half_width, window_half_height,
                         max_iterations, threshold,
                         assymetricWindowLeft, assymetricWindowRight, 
                         assymetricWindowTop, assymetricWindowBottom,
                         calc_err):
    """Numba-accelerated parallel implementation of optical flow computation."""
    # Create output arrays
    u_out = np.zeros_like(U_init)
    v_out = np.zeros_like(V_init)
    status = np.ones((height, width), dtype=np.float32)
    err = np.zeros((height, width), dtype=np.uint8)
    
    # Process each row in parallel
    for i in prange(height):
        for j in range(width):
            # Skip if outside image bounds
            if j < 0 or j >= width or i < 0 or i >= height:
                status[i, j] = 0
                continue
            
            # Compute flow vector
            flow_x, flow_y, error, success = _compute_flow_vector(
                im1, im2, j, i,
                window_half_width, window_half_height,
                max_iterations, threshold,
                assymetricWindowLeft, assymetricWindowRight, 
                assymetricWindowTop, assymetricWindowBottom,
                U_init[i, j], V_init[i, j], calc_err
            )
            
            # Update output
            if success:
                u_out[i, j] = flow_x
                v_out[i, j] = flow_y
                if calc_err:
                    err[i, j] = error
            else:
                status[i, j] = 0
    
    return u_out, v_out, status, err

class denseLucasKanade_Numba(object):
    def __init__(self, Niter=5, halfWindow=13, provideGenericPyramidalDefaults=True, enableVorticityEnhancement=False):
        self.provideGenericPyramidalDefaults = provideGenericPyramidalDefaults
        self.enableVorticityEnhancement = enableVorticityEnhancement

        self.windowHalfWidth = halfWindow
        self.windowHalfHeight = halfWindow
        self.windowWidth = 2 * halfWindow + 1
        self.windowHeight = 2 * halfWindow + 1
        self.Niter = Niter

        print('Running on CPU with Numba acceleration')

    def evaluateVorticityEnhancement(self, U, V):
        if not self.enableVorticityEnhancement:
            return [0, 0, 0, 0]

        D = np.array([[0, -1, 0],
                      [0,  0, 0],
                      [0,  1, 0]], dtype=np.float32) * np.float32(0.5)

        Dv = convolve(V, D.T, mode='reflect')
        Du = convolve(U, D,  mode='reflect')
        omega = Dv - Du
        if np.mean(omega) < -2e-3:
            #Left,Right,Top,Bottom
            return [0, 1, 0, 1]
        elif np.mean(omega) > 2e-3:
            return [1, 0, 0, 1]
        else:
            return [0, 0, 0, 0]

    def compute(self, im1, im2, U, V):
        """
        Compute dense optical flow using Lucas-Kanade method.

        Args:
            im1: First image
            im2: Second image
            U: Initial horizontal flow component
            V: Initial vertical flow component

        Returns:
            U, V: Updated flow components
            calcErr: Whether error was calculated
        """
        print(f"Computing dense Lucas-Kanade optical flow with Numba acceleration")
        print(f"Image size: {im1.shape}")
        print(f"Window size: {self.windowWidth}x{self.windowHeight}")
        print(f"Iterations: {self.Niter}")

        # Get image dimensions
        height, width = im1.shape

        # Evaluate vorticity enhancement
        assymetricWndCfg = self.evaluateVorticityEnhancement(U, V)

        # Determine if error should be calculated
        calcErr = True

        # Threshold for convergence
        threshold = 0.01

        # Use Numba's parallel processing capabilities
        u_out, v_out, status, err = _compute_flow_parallel(
            im1, im2, U, V, height, width,
            self.windowHalfWidth, self.windowHalfHeight,
            self.Niter, threshold,
            assymetricWndCfg[0], assymetricWndCfg[1], 
            assymetricWndCfg[2], assymetricWndCfg[3],
            calcErr
        )

        return u_out, v_out, calcErr

    def getAlgoName(self):
        return 'Numba Dense LK'

    def hasGenericPyramidalDefaults(self):
        return self.provideGenericPyramidalDefaults

    def getGenericPyramidalDefaults(self):
        parameters = {}
        parameters['warping'] = False
        parameters['intermediateScaling'] = True
        parameters['scaling'] = False
        return parameters
