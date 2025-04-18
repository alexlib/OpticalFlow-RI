#!/usr/bin/env python
"""
Display the results of the parabolic profile comparison.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def display_image(image_path, title=None, figsize=(12, 8)):
    """Display an image."""
    plt.figure(figsize=figsize)
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    # Base directory for results
    base_dir = 'parabolic_profile_comparison'
    
    # Display the comparison image
    comparison_path = os.path.join(base_dir, 'profile_comparison.png')
    display_image(comparison_path, "Comparison of Parabolic Profiles")
    
    # Display individual flow fields
    for algo in ['dense_lucas-kanade', 'farneback', 'horn-schunck', 'liu-shen']:
        flow_path = os.path.join(base_dir, f'{algo}_flow.png')
        display_image(flow_path, f"{algo.replace('-', ' ').title()} Flow Field")
    
    # Display individual profiles
    for algo in ['dense_lucas-kanade', 'farneback', 'horn-schunck', 'liu-shen']:
        profile_path = os.path.join(base_dir, f'{algo}_profile.png')
        display_image(profile_path, f"{algo.replace('-', ' ').title()} Profile")

if __name__ == "__main__":
    main()
