"""
Data augmentation utilities for the Ayna ML Assignment.
This module provides various augmentation techniques to improve model training.
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

class PolygonAugmentation:
    """Custom augmentation class for polygon segmentation with color conditioning."""
    
    def __init__(self, 
                 rotation_range=15,
                 scale_range=(0.8, 1.2),
                 flip_prob=0.5,
                 color_jitter_prob=0.3,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2)):
        """
        Initialize augmentation parameters.
        
        Args:
            rotation_range (int): Maximum rotation angle in degrees
            scale_range (tuple): Min/max scale factors
            flip_prob (float): Probability of horizontal flip
            color_jitter_prob (float): Probability of color jittering
            brightness_range (tuple): Min/max brightness factors
            contrast_range (tuple): Min/max contrast factors
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, image, target):
        """
        Apply augmentations to both input image and target.
        
        Args:
            image (PIL.Image): Input image
            target (PIL.Image): Target image
            
        Returns:
            tuple: (augmented_image, augmented_target)
        """
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = image.rotate(angle, resample=Image.BILINEAR)
            target = target.rotate(angle, resample=Image.NEAREST)
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
            image = image.resize(new_size, Image.BILINEAR)
            target = target.resize(new_size, Image.NEAREST)
            
            # Crop or pad to original size
            image = self._crop_or_pad(image, (256, 256))
            target = self._crop_or_pad(target, (256, 256))
        
        # Random horizontal flip
        if random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Color jittering (only on input image)
        if random.random() < self.color_jitter_prob:
            image = self._color_jitter(image)
        
        return image, target
    
    def _crop_or_pad(self, image, target_size):
        """Crop or pad image to target size."""
        current_size = image.size
        
        if current_size[0] > target_size[0] or current_size[1] > target_size[1]:
            # Crop
            left = max(0, (current_size[0] - target_size[0]) // 2)
            top = max(0, (current_size[1] - target_size[1]) // 2)
            right = left + target_size[0]
            bottom = top + target_size[1]
            image = image.crop((left, top, right, bottom))
        else:
            # Pad
            new_image = Image.new(image.mode, target_size, (0, 0, 0))
            paste_x = (target_size[0] - current_size[0]) // 2
            paste_y = (target_size[1] - current_size[1]) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        return image
    
    def _color_jitter(self, image):
        """Apply color jittering to image."""
        # Convert to numpy for easier manipulation
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Brightness
        brightness_factor = random.uniform(*self.brightness_range)
        img_array = img_array * brightness_factor
        
        # Contrast
        contrast_factor = random.uniform(*self.contrast_range)
        mean = np.mean(img_array)
        img_array = (img_array - mean) * contrast_factor + mean
        
        # Clamp values
        img_array = np.clip(img_array, 0, 1)
        
        # Convert back to PIL
        img_array = (img_array * 255).astype(np.uint8)
        return Image.fromarray(img_array)

def get_augmentation_transforms():
    """Get standard augmentation transforms for training."""
    return transforms.Compose([
        PolygonAugmentation(),
        transforms.ToTensor(),
    ])

def get_validation_transforms():
    """Get transforms for validation (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
    ])

# Example usage in training
if __name__ == "__main__":
    # Test augmentation
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Create dummy images
    input_img = Image.new('RGB', (256, 256), color='white')
    target_img = Image.new('RGB', (256, 256), color='black')
    
    # Apply augmentation
    aug = PolygonAugmentation()
    aug_input, aug_target = aug(input_img, target_img)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title('Original Input')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(target_img)
    axes[0, 1].set_title('Original Target')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(aug_input)
    axes[1, 0].set_title('Augmented Input')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(aug_target)
    axes[1, 1].set_title('Augmented Target')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show() 