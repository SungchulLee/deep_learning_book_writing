#!/usr/bin/env python3
'''
============================================================
Model Versioning and Metadata Management
============================================================

Learn best practices for versioning models and tracking
important metadata for reproducibility and deployment.

Topics:
- Model versioning strategies
- Metadata tracking
- Configuration management
- Reproducibility best practices
'''

import torch
import torch.nn as nn
import json
import hashlib
from datetime import datetime
from typing import Dict, Any

class VersionedModel:
    '''
    Wrapper for PyTorch models with versioning and metadata tracking.
    
    Features:
    - Automatic version numbering
    - Configuration tracking
    - Hyperparameter logging
    - Model fingerprinting (hash of weights)
    - Training history tracking
    '''
    
    def __init__(self, model, version="1.0.0", config=None):
        self.model = model
        self.version = version
        self.config = config or {}
        self.metadata = {
            'version': version,
            'created_at': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
        }
        self.training_history = []
    
    def get_model_hash(self):
        '''
        Generate a unique hash based on model weights.
        Useful for verifying model integrity.
        '''
        # Get all parameters as a single tensor
        params = torch.cat([p.flatten() for p in self.model.parameters()])
        
        # Convert to bytes and hash
        params_bytes = params.detach().cpu().numpy().tobytes()
        model_hash = hashlib.sha256(params_bytes).hexdigest()[:16]
        
        return model_hash
    
    def add_training_record(self, epoch, metrics):
        '''
        Add a training record to history.
        '''
        record = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.training_history.append(record)
    
    def save(self, filepath):
        '''
        Save model with full metadata.
        '''
        # Create comprehensive save dictionary
        save_dict = {
            # Model weights
            'model_state_dict': self.model.state_dict(),
            
            # Version information
            'version': self.version,
            'model_hash': self.get_model_hash(),
            
            # Configuration
            'config': self.config,
            'metadata': self.metadata,
            
            # Training history
            'training_history': self.training_history,
            
            # Additional info
            'save_timestamp': datetime.now().isoformat(),
        }
        
        # Save to file
        torch.save(save_dict, filepath)
        
        # Also save metadata as JSON for easy inspection
        json_path = filepath.replace('.pth', '_metadata.json')
        metadata_dict = {
            'version': self.version,
            'model_hash': save_dict['model_hash'],
            'config': self.config,
            'metadata': self.metadata,
            'training_history': self.training_history,
            'save_timestamp': save_dict['save_timestamp'],
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"💾 Model saved: {filepath}")
        print(f"   Version: {self.version}")
        print(f"   Hash: {save_dict['model_hash']}")
        print(f"📄 Metadata saved: {json_path}")
    
    @classmethod
    def load(cls, filepath, model_class, *model_args, **model_kwargs):
        '''
        Load a versioned model from file.
        '''
        # Load checkpoint
        checkpoint = torch.load(filepath)
        
        # Create model instance
        model = model_class(*model_args, **model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create VersionedModel instance
        versioned_model = cls(
            model=model,
            version=checkpoint.get('version', 'unknown'),
            config=checkpoint.get('config', {})
        )
        
        # Restore metadata and history
        versioned_model.metadata = checkpoint.get('metadata', {})
        versioned_model.training_history = checkpoint.get('training_history', [])
        
        # Verify integrity
        saved_hash = checkpoint.get('model_hash')
        current_hash = versioned_model.get_model_hash()
        
        print(f"📂 Model loaded: {filepath}")
        print(f"   Version: {versioned_model.version}")
        print(f"   Hash match: {saved_hash == current_hash}")
        
        if saved_hash != current_hash:
            print("   ⚠️  Warning: Model hash mismatch!")
        
        return versioned_model
    
    def print_info(self):
        '''
        Print comprehensive model information.
        '''
        print("\n" + "=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)
        
        print(f"\n📦 Version: {self.version}")
        print(f"🔑 Hash: {self.get_model_hash()}")
        
        print(f"\n⚙️  Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")
        
        print(f"\n📊 Metadata:")
        for key, value in self.metadata.items():
            print(f"   {key}: {value}")
        
        if self.training_history:
            print(f"\n📈 Training History: {len(self.training_history)} records")
            if self.training_history:
                last = self.training_history[-1]
                print(f"   Last epoch: {last['epoch']}")
                print(f"   Last metrics: {last['metrics']}")


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL VERSIONING DEMONSTRATION")
    print("=" * 60)
    
    # Define a simple model
    class MyModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # Create model with configuration
    config = {
        'input_size': 784,
        'hidden_size': 128,
        'output_size': 10,
        'learning_rate': 0.001,
        'batch_size': 64,
        'optimizer': 'Adam',
    }
    
    model = MyModel(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size']
    )
    
    # Create versioned model
    versioned_model = VersionedModel(
        model=model,
        version="1.0.0",
        config=config
    )
    
    # Simulate training
    print("\n🏋️  Simulating training...")
    for epoch in range(1, 4):
        metrics = {
            'loss': 1.0 / epoch,
            'accuracy': 0.5 + epoch * 0.1
        }
        versioned_model.add_training_record(epoch, metrics)
        print(f"   Epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
    
    # Print model info
    versioned_model.print_info()
    
    # Save model
    print("\n" + "=" * 60)
    save_path = "versioned_model_v1.0.0.pth"
    versioned_model.save(save_path)
    
    # Load model
    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    
    loaded_model = VersionedModel.load(
        filepath=save_path,
        model_class=MyModel,
        input_size=784,
        hidden_size=128,
        output_size=10
    )
    
    # Verify
    loaded_model.print_info()
    
    # Cleanup
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
    json_path = save_path.replace('.pth', '_metadata.json')
    if os.path.exists(json_path):
        os.remove(json_path)
    print("\n🗑️  Cleaned up demo files")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
