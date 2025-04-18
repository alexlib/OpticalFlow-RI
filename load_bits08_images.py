#!/usr/bin/env python
"""
Simple script to load and display Bits08 test images.
"""

import sys
import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

def main():
    print("Starting image loading test...")
    sys.stdout.flush()
    
    # Load Bits08 test images
    img1_path = "examples/testImages/Bits08/Ni06/parabolic01_0.tif"
    img2_path = "examples/testImages/Bits08/Ni06/parabolic01_1.tif"
    
    print(f"Loading images: {img1_path} and {img2_path}")
    sys.stdout.flush()
    
    try:
        img1 = imread(img1_path).astype(np.float32)
        print("First image loaded successfully.")
        sys.stdout.flush()
        
        img2 = imread(img2_path).astype(np.float32)
        print("Second image loaded successfully.")
        sys.stdout.flush()
        
        # Print image information
        print(f"Image shape: {img1.shape}")
        print(f"Image dtype: {img1.dtype}")
        print(f"Image min/max values: {img1.min()}/{img1.max()}")
        sys.stdout.flush()
        
        # Create output directory
        output_dir = 'image_test_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot images
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img1, cmap='gray')
        plt.title('Image 1')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2, cmap='gray')
        plt.title('Image 2')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "bits08_images.png"), dpi=150)
        plt.close()
        
        print(f"Images saved to {output_dir}/bits08_images.png")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"Error loading images: {e}")
        sys.stdout.flush()
    
    print("Test completed.")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
