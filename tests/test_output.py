#!/usr/bin/env python
"""
Simple test script to verify that output is working correctly.
"""

import sys
import time
from tqdm import tqdm

def main():
    print("Starting test script...")
    sys.stdout.flush()
    
    print("Testing direct print statements...")
    sys.stdout.flush()
    
    print("Testing tqdm progress bar...")
    sys.stdout.flush()
    
    for i in tqdm(range(10), desc="Progress"):
        print(f"Step {i}")
        sys.stdout.flush()
        time.sleep(0.5)
    
    print("Test completed successfully!")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
