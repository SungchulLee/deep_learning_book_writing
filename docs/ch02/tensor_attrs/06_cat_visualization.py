#!/usr/bin/env python3
# ========================================================
# 03_tensor_attributes_and_methods_6_cat_visualization.py
# ========================================================
import tensor_features as tfs

tfs.download_cat_images()
    
batch = tfs.load_cat_images()
print(f"\nBatch tensor info:")
print(f"  Data Type: {type(batch)}")
print(f"  Shape    : {batch.shape}")
print(f"  Type     : {batch.dtype}")
print(f"  Range    : [{batch.min():.3f}, {batch.max():.3f}]")
        
tfs.display_images(batch)