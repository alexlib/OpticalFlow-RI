#!/usr/bin/env python
"""
Display the OpenPIV results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.io import loadmat
import matplotlib.image as mpimg

def display_flow_image():
    """Display the flow image from the OpenPIV analysis."""
    img_path = os.path.join('simple_piv_analysis', 'simple_piv_flow.png')
    
    if not os.path.exists(img_path):
        print(f"Error: Image file {img_path} not found.")
        return
    
    img = mpimg.imread(img_path)
    
    plt.figure(figsize=(14, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('OpenPIV Flow Field (window=64, overlap=32, search_area=64)')
    plt.tight_layout()
    plt.show()

def display_profile_image():
    """Display the profile image from the OpenPIV analysis."""
    img_path = os.path.join('simple_piv_analysis', 'simple_piv_profile.png')
    
    if not os.path.exists(img_path):
        print(f"Error: Image file {img_path} not found.")
        return
    
    img = mpimg.imread(img_path)
    
    plt.figure(figsize=(14, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('OpenPIV Velocity Profile (window=64, overlap=32, search_area=64)')
    plt.tight_layout()
    plt.show()

def display_vector_image():
    """Display the vector image from the OpenPIV analysis."""
    img_path = os.path.join('simple_piv_analysis', 'simple_piv_vectors.png')
    
    if not os.path.exists(img_path):
        print(f"Error: Image file {img_path} not found.")
        return
    
    img = mpimg.imread(img_path)
    
    plt.figure(figsize=(14, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('OpenPIV Vector Field (window=64, overlap=32, search_area=64)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Display all results
    display_flow_image()
    display_profile_image()
    display_vector_image()
