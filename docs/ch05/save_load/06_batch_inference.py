"""
Module 64.06: Batch Inference
==============================

This module demonstrates efficient batch inference pipelines for processing
large datasets. Students will learn to optimize throughput, manage memory,
and monitor progress for production batch processing.

Learning Objectives:
-------------------
1. Implement efficient batch inference pipelines
2. Optimize DataLoader for inference
3. Monitor and log batch processing
4. Handle errors in batch processing
5. Save and organize batch results

Time Estimate: 60 minutes
Difficulty: Intermediate
Prerequisites: Module 64.01, 13 (DataLoaders)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime
from tqdm import tqdm  # Progress bar
import csv


# ============================================================================
# PART 1: Model Definition
# ============================================================================

class SimpleClassifier(nn.Module):
    """Simple CNN for demonstration."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================================
# PART 2: Dataset for Batch Inference
# ============================================================================

class InferenceDataset(Dataset):
    """
    Dataset for batch inference.
    
    Key differences from training datasets:
    - No labels needed
    - Focus on efficiency
    - May include metadata (filenames, IDs)
    - Often processes files from disk
    """
    
    def __init__(self, image_paths: List[str], transform=None):
        """
        Initialize inference dataset.
        
        Args:
            image_paths: List of paths to images
            transform: Optional preprocessing transform
        """
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and preprocess single image.
        
        Returns:
            tuple: (image_tensor, metadata_dict)
        """
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            image = image.resize((28, 28))
            
            # Convert to tensor
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Add channel dim
            
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            # Metadata to track results
            metadata = {
                'image_path': str(image_path),
                'filename': Path(image_path).name,
                'index': idx
            }
            
            return image_tensor, metadata
            
        except Exception as e:
            # Return error information
            print(f"Error loading {image_path}: {e}")
            return None, {'error': str(e), 'image_path': str(image_path)}


# ============================================================================
# PART 3: Batch Inference Engine
# ============================================================================

class BatchInferenceEngine:
    """
    Engine for efficient batch inference.
    
    Features:
    - GPU utilization optimization
    - Progress tracking
    - Error handling
    - Result aggregation
    - Performance monitoring
    """
    
    def __init__(self, model: nn.Module, device: str = 'auto',
                 batch_size: int = 32, num_workers: int = 4):
        """
        Initialize batch inference engine.
        
        Args:
            model: PyTorch model for inference
            device: Device to use ('auto', 'cpu', 'cuda')
            batch_size: Batch size for inference
            num_workers: Number of workers for data loading
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0.0,
            'inference_time': 0.0,
            'loading_time': 0.0
        }
    
    def run(self, image_paths: List[str], show_progress: bool = True) -> Dict[str, Any]:
        """
        Run batch inference on list of image paths.
        
        Args:
            image_paths: List of paths to images
            show_progress: Show progress bar
            
        Returns:
            Dictionary with results and statistics
        """
        print("\n" + "="*70)
        print("RUNNING BATCH INFERENCE")
        print("="*70)
        print(f"Total images: {len(image_paths)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        print(f"Workers: {self.num_workers}")
        
        start_time = time.time()
        
        # Create dataset and dataloader
        dataset = InferenceDataset(image_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Keep order for matching results to inputs
            pin_memory=True if torch.cuda.is_available() else False  # Faster GPU transfer
        )
        
        # Storage for results
        all_results = []
        failed_items = []
        
        # Inference loop with progress bar
        with torch.no_grad():  # Disable gradient computation
            pbar = tqdm(dataloader, desc="Processing batches") if show_progress else dataloader
            
            for batch_idx, (images, metadata_list) in enumerate(pbar):
                batch_start = time.time()
                
                # Filter out failed loads (None values)
                valid_indices = [i for i, img in enumerate(images) if img is not None]
                if len(valid_indices) == 0:
                    continue
                
                # Stack valid images
                valid_images = [images[i] for i in valid_indices]
                valid_metadata = [metadata_list[i] for i in valid_indices]
                
                try:
                    # Stack into batch tensor
                    batch_tensor = torch.stack(valid_images).to(self.device)
                    
                    # Run inference
                    inference_start = time.time()
                    outputs = self.model(batch_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    self.stats['inference_time'] += time.time() - inference_start
                    
                    # Get predictions
                    probs, preds = torch.max(probabilities, dim=1)
                    
                    # Store results
                    for i, (prob, pred, meta) in enumerate(zip(probs, preds, valid_metadata)):
                        result = {
                            'filename': meta['filename'],
                            'image_path': meta['image_path'],
                            'predicted_class': int(pred.item()),
                            'confidence': float(prob.item()),
                            'batch_index': batch_idx,
                            'timestamp': datetime.now().isoformat()
                        }
                        all_results.append(result)
                        self.stats['successful'] += 1
                        
                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {e}")
                    self.stats['failed'] += len(valid_images)
                    for meta in valid_metadata:
                        failed_items.append({
                            'filename': meta['filename'],
                            'error': str(e)
                        })
                
                self.stats['loading_time'] += time.time() - batch_start - self.stats['inference_time']
        
        self.stats['total_time'] = time.time() - start_time
        self.stats['total_images'] = len(image_paths)
        
        return {
            'results': all_results,
            'failed': failed_items,
            'statistics': self.stats
        }
    
    def print_statistics(self):
        """Print detailed statistics."""
        print("\n" + "="*70)
        print("BATCH INFERENCE STATISTICS")
        print("="*70)
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Success rate: {self.stats['successful'] / max(self.stats['total_images'], 1) * 100:.2f}%")
        print(f"\nTiming:")
        print(f"  Total time: {self.stats['total_time']:.2f}s")
        print(f"  Inference time: {self.stats['inference_time']:.2f}s")
        print(f"  Loading time: {self.stats['loading_time']:.2f}s")
        print(f"  Throughput: {self.stats['successful'] / max(self.stats['total_time'], 0.001):.2f} images/sec")
        if self.stats['inference_time'] > 0:
            print(f"  Inference throughput: {self.stats['successful'] / self.stats['inference_time']:.2f} images/sec")


# ============================================================================
# PART 4: Result Saving and Organization
# ============================================================================

class ResultSaver:
    """
    Save batch inference results in various formats.
    
    Supports:
    - JSON (full details)
    - CSV (tabular)
    - Organized by confidence
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize result saver.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, results: Dict[str, Any], filename: str = "results.json"):
        """Save results as JSON."""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved JSON results to: {output_path}")
    
    def save_csv(self, results: List[Dict], filename: str = "results.csv"):
        """Save results as CSV."""
        if not results:
            print("No results to save")
            return
        
        output_path = self.output_dir / filename
        
        # Get all keys from first result
        fieldnames = list(results[0].keys())
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ Saved CSV results to: {output_path}")
    
    def save_summary(self, stats: Dict[str, Any], filename: str = "summary.txt"):
        """Save summary statistics."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("Batch Inference Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total images: {stats['total_images']}\n")
            f.write(f"Successful: {stats['successful']}\n")
            f.write(f"Failed: {stats['failed']}\n")
            f.write(f"Success rate: {stats['successful'] / max(stats['total_images'], 1) * 100:.2f}%\n")
            f.write(f"\nTotal time: {stats['total_time']:.2f}s\n")
            f.write(f"Inference time: {stats['inference_time']:.2f}s\n")
            f.write(f"Throughput: {stats['successful'] / max(stats['total_time'], 0.001):.2f} images/sec\n")
        
        print(f"✓ Saved summary to: {output_path}")
    
    def organize_by_confidence(self, results: List[Dict], thresholds: List[float] = [0.5, 0.7, 0.9]):
        """
        Organize results by confidence levels.
        
        Args:
            results: List of prediction results
            thresholds: Confidence thresholds for categorization
        """
        # Create directories
        high_conf_dir = self.output_dir / "high_confidence"
        medium_conf_dir = self.output_dir / "medium_confidence"
        low_conf_dir = self.output_dir / "low_confidence"
        
        for dir_path in [high_conf_dir, medium_conf_dir, low_conf_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Categorize results
        categories = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        for result in results:
            conf = result['confidence']
            if conf >= thresholds[2]:
                categories['high'].append(result)
            elif conf >= thresholds[1]:
                categories['medium'].append(result)
            else:
                categories['low'].append(result)
        
        # Save categorized results
        for category, items in categories.items():
            if items:
                self.save_csv(items, f"{category}_confidence.csv")
        
        print(f"\n✓ Organized {len(results)} results by confidence")
        print(f"  High (>{thresholds[2]}): {len(categories['high'])}")
        print(f"  Medium ({thresholds[1]}-{thresholds[2]}): {len(categories['medium'])}")
        print(f"  Low (<{thresholds[1]}): {len(categories['low'])}")


# ============================================================================
# PART 5: Creating Synthetic Test Data
# ============================================================================

def create_test_images(num_images: int = 100, output_dir: str = "data/test_images"):
    """
    Create synthetic test images for batch inference demonstration.
    
    Args:
        num_images: Number of test images to create
        output_dir: Directory to save images
    """
    print(f"\nCreating {num_images} synthetic test images...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        # Create random 28x28 grayscale image
        img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(output_path / f"test_image_{i:04d}.png")
    
    print(f"✓ Created {num_images} test images in {output_dir}")
    return [str(p) for p in output_path.glob("*.png")]


# ============================================================================
# PART 6: Main Demonstration
# ============================================================================

def main():
    """
    Main demonstration of batch inference.
    """
    print("\n" + "="*70)
    print("MODULE 64.06: BATCH INFERENCE")
    print("="*70)
    
    # Create test images
    image_paths = create_test_images(num_images=100, output_dir="data/test_images")
    
    # Initialize model
    print("\nInitializing model...")
    model = SimpleClassifier()
    
    # Create inference engine
    print("Creating batch inference engine...")
    engine = BatchInferenceEngine(
        model=model,
        device='auto',
        batch_size=16,
        num_workers=4
    )
    
    # Run batch inference
    results = engine.run(image_paths, show_progress=True)
    
    # Print statistics
    engine.print_statistics()
    
    # Save results
    print("\nSaving results...")
    saver = ResultSaver(output_dir="results/batch_inference")
    saver.save_json(results, "full_results.json")
    saver.save_csv(results['results'], "predictions.csv")
    saver.save_summary(results['statistics'], "summary.txt")
    
    if results['failed']:
        saver.save_csv(results['failed'], "failed_items.csv")
    
    # Organize by confidence
    saver.organize_by_confidence(
        results['results'],
        thresholds=[0.5, 0.7, 0.9]
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. Batch inference is much more efficient than single predictions:
       ✓ Better GPU utilization
       ✓ Amortized preprocessing costs
       ✓ Higher throughput
       
    2. Use DataLoader for efficient batching:
       ✓ Multi-worker data loading
       ✓ Automatic batching
       ✓ Memory pinning for GPU
       
    3. Monitor and log progress:
       ✓ Progress bars for UX
       ✓ Track timing statistics
       ✓ Handle errors gracefully
       
    4. Organize results systematically:
       ✓ Multiple output formats (JSON, CSV)
       ✓ Categorize by confidence
       ✓ Link results to inputs
       
    5. Production considerations:
       ✓ Checkpointing for resumability
       ✓ Distributed processing for large datasets
       ✓ Result validation and QA
       ✓ Error recovery mechanisms
    """)
    
    print("\n✅ Module 64.06 completed successfully!")
    print("Next: Module 64.07 - Model Versioning")


if __name__ == "__main__":
    main()
