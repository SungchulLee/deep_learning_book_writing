"""
EBM Applications: Out-of-Distribution Detection, Denoising, Compositional Generation
================================================================================

Practical applications of energy-based models beyond generation.

Duration: 90-120 minutes
"""

import torch
import torch.nn as nn
import numpy as np

def out_of_distribution_detection():
    """Use EBM energy for OOD detection."""
    print("\nOut-of-Distribution Detection with EBMs")
    print("Lower energy → In-distribution")
    print("Higher energy → Out-of-distribution")
    print("✓ Can identify anomalies and novel inputs")

def image_denoising_with_ebm():
    """Denoise images using energy minimization."""
    print("\nImage Denoising via Energy Minimization")
    print("Start with noisy image, minimize energy")
    print("✓ Preserves structure while removing noise")

def compositional_generation():
    """Combine multiple concepts using energy addition."""
    print("\nCompositional Generation")
    print("E_combined = E_concept1 + E_concept2")
    print("✓ Generate images with multiple attributes")

def main():
    print("="*70)
    print("EBM APPLICATIONS")
    print("="*70)
    
    out_of_distribution_detection()
    image_denoising_with_ebm()
    compositional_generation()
    
    print("\nKey Applications:")
    print("  ✓ Anomaly/OOD detection")
    print("  ✓ Image denoising and inpainting")
    print("  ✓ Compositional generation")
    print("  ✓ Adversarial robustness")

if __name__ == "__main__":
    main()
