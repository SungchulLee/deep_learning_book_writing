"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
Visualizes which parts of an image are important for CNN predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List


class GradCAM:
    """
    Implements Grad-CAM for visualizing CNN model decisions.
    
    Reference: "Grad-CAM: Visual Explanations from Deep Networks via 
    Gradient-based Localization" (Selvaraju et al., 2017)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model
            target_layer: The convolutional layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Target class index. If None, uses predicted class
            
        Returns:
            Heatmap as numpy array of shape (H, W)
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None,
                     original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate and visualize Grad-CAM overlay on original image.
        
        Args:
            input_image: Input tensor of shape (1, C, H, W)
            target_class: Target class index
            original_image: Original image as numpy array (H, W, 3) in RGB format
            
        Returns:
            Visualization as numpy array (H, W, 3)
        """
        # Generate CAM
        cam = self.generate_cam(input_image, target_class)
        
        # Resize CAM to match input image size
        if original_image is not None:
            h, w = original_image.shape[:2]
        else:
            h, w = input_image.shape[2:]
        
        cam = cv2.resize(cam, (w, h))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        if original_image is not None:
            # Normalize original image to [0, 255]
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            
            # Blend
            visualization = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        else:
            visualization = heatmap
        
        return visualization


class GradCAMPlusPlus(GradCAM):
    """
    Implements Grad-CAM++ for improved localization.
    
    Reference: "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"
    (Chattopadhyay et al., 2018)
    """
    
    def generate_cam(self, input_image: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, target_class]
        class_score.backward(retain_graph=True)
        
        # Get first order gradients
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Calculate second and third order gradients
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        # Calculate alpha weights
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + (grad_3 * activations).sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / alpha_denom
        
        # Calculate weights
        weights = (alpha * F.relu(gradients)).sum(dim=(1, 2))
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def get_target_layer(model: nn.Module, layer_name: Optional[str] = None) -> nn.Module:
    """
    Helper function to get the target layer from a model.
    
    Args:
        model: The neural network model
        layer_name: Name of the layer (e.g., 'layer4' for ResNet)
        
    Returns:
        The target layer module
    """
    if layer_name:
        return dict(model.named_modules())[layer_name]
    
    # Try to find last convolutional layer automatically
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    
    raise ValueError("Could not find a convolutional layer in the model")
