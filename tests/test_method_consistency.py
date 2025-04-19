import sys
import os
import numpy as np
from skimage.transform import resize
from skimage.data import camera
from skimage.io import imread
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analyze_multipass_piv import analyze_with_multipass_piv

from scipy.interpolate import griddata

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from Farneback_numba import Farneback_Numba
from denseLucasKanade_numba import denseLucasKanade_numba
from HornSchunck_numba import HSOpticalFlowAlgoAdapter
from GenericPyramidalOpticalFlow import genericPyramidalOpticalFlow

# OpenPIV import
try:
    from openpiv import pyprocess
    openpiv_available = True
except ImportError:
    openpiv_available = False
    print("OpenPIV not installed. Skipping OpenPIV test.")

def postprocess_flow(U, V, u_scale=4.0, v_scale=1.0, smooth_sigma=1.5):
    # Gaussian smoothing
    U = gaussian_filter(U, sigma=smooth_sigma)
    V = gaussian_filter(V, sigma=smooth_sigma)
    # Scale to OpenPIV range
    u_max = max(abs(U.min()), abs(U.max()))
    v_max = max(abs(V.min()), abs(V.max()))
    if u_max > 0:
        U = U / u_max * u_scale
    if v_max > 0:
        V = V / v_max * v_scale
    # Invert U to match OpenPIV direction (if needed)
    U = -abs(U)
    return U, V

def compare_flows(U1, V1, U2, V2, name1, name2, border=10):
    # Compare only central region to avoid boundary effects
    U1c = U1[border:-border, border:-border]
    V1c = V1[border:-border, border:-border]
    U2c = U2[border:-border, border:-border]
    V2c = V2[border:-border, border:-border]
    du = U1c - U2c
    dv = V1c - V2c
    print(f"\n{name1} vs {name2} (central region):")
    print(f"  Mean abs diff U: {np.mean(np.abs(du)):.3f}, V: {np.mean(np.abs(dv)):.3f}")
    print(f"  Max abs diff U: {np.max(np.abs(du)):.3f}, V: {np.max(np.abs(dv)):.3f}")

def main():
    # Load parabolic test images from testImages/Bits08/Ni06
    base_path = os.path.join(os.path.dirname(__file__), '..', 'testImages', 'Bits08', 'Ni06')
    img1_path = os.path.join(base_path, 'parabolic01_0.tif')
    img2_path = os.path.join(base_path, 'parabolic01_1.tif')
    print(f"Loading images from {img1_path} and {img2_path}")
    img1 = imread(img1_path).astype(np.float32)
    img2 = imread(img2_path).astype(np.float32)

    # Use OpenPIV windef.simple_multipass (multi-grid) for reference
    from openpiv import windef
    settings = windef.PIVSettings()
    settings.windowsizes = (64, 32, 16)
    settings.overlap = (32, 16, 8)
    settings.num_iterations = 3
    x, y, u_piv, v_piv, flags = windef.simple_multipass(img1, img2, settings)
    points_grid = np.column_stack((y.flatten(), x.flatten()))

    # Farneback (large window, pyramidal)
    fb = Farneback_Numba(windowSize=33, Niters=5, polyN=7, polySigma=1.5, pyramidalLevels=2)
    U_fb = np.zeros_like(img1)
    V_fb = np.zeros_like(img1)
    U_fb, V_fb, _ = fb.compute(img1, img2, U_fb, V_fb)
    u_fb_interp = griddata((np.indices(U_fb.shape)[0].flatten(), np.indices(U_fb.shape)[1].flatten()), U_fb.flatten(), points_grid, method='linear', fill_value=0).reshape(u_piv.shape)
    v_fb_interp = griddata((np.indices(V_fb.shape)[0].flatten(), np.indices(V_fb.shape)[1].flatten()), V_fb.flatten(), points_grid, method='linear', fill_value=0).reshape(u_piv.shape)
    # Postprocess and compare to OpenPIV
    u_fb_post, v_fb_post = postprocess_flow(u_fb_interp, v_fb_interp)
    u_piv_post, v_piv_post = postprocess_flow(u_piv, v_piv)
    compare_flows(u_fb_post, v_fb_post, u_piv_post, v_piv_post, 'Farneback', 'OpenPIV')

    # Lucas-Kanade (large window, pyramidal)
    lk = denseLucasKanade_numba(Niter=5, halfWindow=16, provideGenericPyramidalDefaults=True)
    U_lk = np.zeros_like(img1)
    V_lk = np.zeros_like(img1)
    U_lk, V_lk, _ = lk.compute(img1, img2, U_lk, V_lk)
    u_lk_interp = griddata((np.indices(U_lk.shape)[0].flatten(), np.indices(U_lk.shape)[1].flatten()), U_lk.flatten(), points_grid, method='linear', fill_value=0).reshape(u_piv.shape)
    v_lk_interp = griddata((np.indices(V_lk.shape)[0].flatten(), np.indices(V_lk.shape)[1].flatten()), V_lk.flatten(), points_grid, method='linear', fill_value=0).reshape(u_piv.shape)
    # Postprocess and compare to OpenPIV
    u_lk_post, v_lk_post = postprocess_flow(u_lk_interp, v_lk_interp)
    compare_flows(u_lk_post, v_lk_post, u_piv_post, v_piv_post, 'Lucas-Kanade', 'OpenPIV')

    # Horn-Schunck (pyramidal)
    hs_adapter = HSOpticalFlowAlgoAdapter(alphas=[0.1], Niter=100, provideGenericPyramidalDefaults=True)
    U_hs = np.zeros_like(img1)
    V_hs = np.zeros_like(img1)
    U_hs, V_hs, _ = hs_adapter.compute(img1, img2, U_hs, V_hs)
    u_hs_interp = griddata((np.indices(U_hs.shape)[0].flatten(), np.indices(U_hs.shape)[1].flatten()), U_hs.flatten(), points_grid, method='linear', fill_value=0).reshape(u_piv.shape)
    v_hs_interp = griddata((np.indices(V_hs.shape)[0].flatten(), np.indices(V_hs.shape)[1].flatten()), V_hs.flatten(), points_grid, method='linear', fill_value=0).reshape(u_piv.shape)
    # Postprocess and compare to OpenPIV
    u_hs_post, v_hs_post = postprocess_flow(u_hs_interp, v_hs_interp)
    compare_flows(u_hs_post, v_hs_post, u_piv_post, v_piv_post, 'Horn-Schunck', 'OpenPIV')

    # Plot all results on the 16x16 grid
    def show_quiver_on_grid(x, y, u, v, title, scale=0.5):
        plt.figure(figsize=(7, 7))
        plt.imshow(img1, cmap='gray', alpha=0.5)
        plt.quiver(x, y, u, v, color='r', angles='xy', scale_units='xy', scale=scale)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    show_quiver_on_grid(x, y, u_piv, v_piv, 'OpenPIV (16x16 grid)')
    show_quiver_on_grid(x, y, u_fb_interp, v_fb_interp, 'Farneback (interpolated to 16x16 grid)')
    show_quiver_on_grid(x, y, u_lk_interp, v_lk_interp, 'Lucas-Kanade (interpolated to 16x16 grid)')
    show_quiver_on_grid(x, y, u_hs_interp, v_hs_interp, 'Horn-Schunck (interpolated to 16x16 grid)')

if __name__ == "__main__":
    main()
