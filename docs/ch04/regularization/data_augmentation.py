"""
Data Augmentation Example
==========================
Demonstrates how data augmentation prevents overfitting by artificially
expanding the training dataset with transformed versions of existing data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_synthetic_image_dataset(n_samples=1000, img_size=28):
    """Create a synthetic image dataset for demonstration."""
    # Generate simple geometric shapes as images
    np.random.seed(42)
    X = []
    y = []
    
    for i in range(n_samples):
        img = np.zeros((img_size, img_size))
        class_label = i % 3  # 3 classes
        
        if class_label == 0:  # Circle
            center = (np.random.randint(10, img_size-10), np.random.randint(10, img_size-10))
            radius = np.random.randint(5, 10)
            for x in range(img_size):
                for y_coord in range(img_size):
                    if (x - center[0])**2 + (y_coord - center[1])**2 <= radius**2:
                        img[y_coord, x] = 1.0
        
        elif class_label == 1:  # Square
            top_left = (np.random.randint(5, img_size-15), np.random.randint(5, img_size-15))
            size = np.random.randint(8, 15)
            img[top_left[1]:top_left[1]+size, top_left[0]:top_left[0]+size] = 1.0
        
        else:  # Triangle
            points = [
                (np.random.randint(5, img_size-5), np.random.randint(5, 15)),
                (np.random.randint(5, 15), np.random.randint(15, img_size-5)),
                (np.random.randint(15, img_size-5), np.random.randint(15, img_size-5))
            ]
            for x in range(img_size):
                for y_coord in range(img_size):
                    if point_in_triangle((x, y_coord), points):
                        img[y_coord, x] = 1.0
        
        # Add some noise
        img += np.random.normal(0, 0.1, (img_size, img_size))
        img = np.clip(img, 0, 1)
        
        X.append(img)
        y.append(class_label)
    
    return np.array(X), np.array(y)


def point_in_triangle(p, triangle):
    """Check if point is inside triangle (helper function)."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, triangle[0], triangle[1])
    d2 = sign(p, triangle[1], triangle[2])
    d3 = sign(p, triangle[2], triangle[0])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


def create_model(input_shape):
    """Create a CNN model."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def visualize_augmentation(original_img, datagen, n_examples=9):
    """Visualize augmented versions of an image."""
    # Prepare image for augmentation
    img = original_img.reshape((1,) + original_img.shape + (1,))
    
    plt.figure(figsize=(12, 12))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    # Augmented images
    i = 2
    for batch in datagen.flow(img, batch_size=1):
        plt.subplot(3, 3, i)
        plt.imshow(batch[0, :, :, 0], cmap='gray')
        plt.title(f'Augmented {i-1}', fontsize=10)
        plt.axis('off')
        i += 1
        if i > 9:
            break
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("Augmentation examples saved as 'augmentation_examples.png'")


def train_without_augmentation(X_train, y_train, X_val, y_val, epochs=50):
    """Train model without data augmentation."""
    print("Training WITHOUT data augmentation...")
    
    # Add channel dimension
    X_train_exp = X_train[..., np.newaxis]
    X_val_exp = X_val[..., np.newaxis]
    
    model = create_model(X_train_exp.shape[1:])
    history = model.fit(
        X_train_exp, y_train,
        validation_data=(X_val_exp, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0
    )
    
    return model, history


def train_with_augmentation(X_train, y_train, X_val, y_val, epochs=50):
    """Train model with data augmentation."""
    print("Training WITH data augmentation...")
    
    # Add channel dimension
    X_train_exp = X_train[..., np.newaxis]
    X_val_exp = X_val[..., np.newaxis]
    
    # Create data augmentation generator
    datagen = ImageDataGenerator(
        rotation_range=20,           # Random rotation up to 20 degrees
        width_shift_range=0.2,       # Random horizontal shift
        height_shift_range=0.2,      # Random vertical shift
        zoom_range=0.2,              # Random zoom
        horizontal_flip=True,        # Random horizontal flip
        fill_mode='nearest'          # Fill strategy for new pixels
    )
    
    # Fit the augmentation generator on training data
    datagen.fit(X_train_exp)
    
    # Create model
    model = create_model(X_train_exp.shape[1:])
    
    # Train with augmented data
    history = model.fit(
        datagen.flow(X_train_exp, y_train, batch_size=32),
        validation_data=(X_val_exp, y_val),
        epochs=epochs,
        steps_per_epoch=len(X_train) // 32,
        verbose=0
    )
    
    return model, history, datagen


def custom_augmentation_example():
    """Demonstrate custom augmentation functions."""
    print("\n" + "="*60)
    print("Custom Augmentation Functions")
    print("="*60)
    
    def add_noise(image, noise_factor=0.1):
        """Add random Gaussian noise."""
        noise = np.random.normal(0, noise_factor, image.shape)
        return np.clip(image + noise, 0, 1)
    
    def brightness_adjustment(image, factor_range=(0.7, 1.3)):
        """Randomly adjust brightness."""
        factor = np.random.uniform(*factor_range)
        return np.clip(image * factor, 0, 1)
    
    def random_crop(image, crop_size=0.8):
        """Randomly crop the image."""
        h, w = image.shape
        new_h, new_w = int(h * crop_size), int(w * crop_size)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        cropped = image[top:top+new_h, left:left+new_w]
        # Resize back to original size
        from scipy.ndimage import zoom
        zoom_factor = (h / new_h, w / new_w)
        return zoom(cropped, zoom_factor, order=1)
    
    # Generate sample image
    sample_img = np.zeros((28, 28))
    sample_img[10:18, 10:18] = 1.0
    
    # Apply augmentations
    augmented = {
        'Original': sample_img,
        'With Noise': add_noise(sample_img),
        'Brightness Up': brightness_adjustment(sample_img, (1.2, 1.5)),
        'Brightness Down': brightness_adjustment(sample_img, (0.5, 0.8))
    }
    
    # Visualize
    plt.figure(figsize=(12, 3))
    for idx, (name, img) in enumerate(augmented.items(), 1):
        plt.subplot(1, 4, idx)
        plt.imshow(img, cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('custom_augmentation.png', dpi=150)
    print("Custom augmentation examples saved as 'custom_augmentation.png'")


def main():
    # Generate dataset
    print("Generating synthetic dataset...")
    X, y = create_synthetic_image_dataset(n_samples=1000, img_size=28)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}\n")
    
    # Train without augmentation
    model_no_aug, history_no_aug = train_without_augmentation(
        X_train, y_train, X_val, y_val, epochs=50
    )
    
    # Train with augmentation
    model_aug, history_aug, datagen = train_with_augmentation(
        X_train, y_train, X_val, y_val, epochs=50
    )
    
    # Visualize augmentation
    visualize_augmentation(X_train[0], datagen)
    
    # Custom augmentation examples
    custom_augmentation_example()
    
    # Evaluate models
    print("\n" + "="*60)
    print("Test Set Performance")
    print("="*60)
    
    X_test_exp = X_test[..., np.newaxis]
    
    loss_no_aug, acc_no_aug = model_no_aug.evaluate(X_test_exp, y_test, verbose=0)
    loss_aug, acc_aug = model_aug.evaluate(X_test_exp, y_test, verbose=0)
    
    print(f"Without Augmentation:")
    print(f"  Test Loss: {loss_no_aug:.4f}")
    print(f"  Test Accuracy: {acc_no_aug:.4f}\n")
    
    print(f"With Augmentation:")
    print(f"  Test Loss: {loss_aug:.4f}")
    print(f"  Test Accuracy: {acc_aug:.4f}")
    print(f"  Improvement: {(acc_aug - acc_no_aug)*100:.2f}%\n")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_no_aug.history['loss'], label='Train (No Aug)', linewidth=2)
    plt.plot(history_no_aug.history['val_loss'], label='Val (No Aug)', linewidth=2)
    plt.plot(history_aug.history['loss'], label='Train (With Aug)', 
             linestyle='--', linewidth=2)
    plt.plot(history_aug.history['val_loss'], label='Val (With Aug)', 
             linestyle='--', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_no_aug.history['accuracy'], label='Train (No Aug)', linewidth=2)
    plt.plot(history_no_aug.history['val_accuracy'], label='Val (No Aug)', linewidth=2)
    plt.plot(history_aug.history['accuracy'], label='Train (With Aug)', 
             linestyle='--', linewidth=2)
    plt.plot(history_aug.history['val_accuracy'], label='Val (With Aug)', 
             linestyle='--', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('augmentation_training_comparison.png', dpi=150)
    print("Training comparison plot saved as 'augmentation_training_comparison.png'")
    
    # Summary
    print("\n" + "="*60)
    print("Data Augmentation Techniques")
    print("="*60)
    print("Common augmentation techniques include:")
    print("  • Geometric: Rotation, flipping, scaling, translation, shearing")
    print("  • Color: Brightness, contrast, saturation adjustment")
    print("  • Noise: Gaussian noise, salt-and-pepper noise")
    print("  • Cropping: Random crops, center crops")
    print("  • Advanced: Cutout, mixup, CutMix")
    print("\nBenefits:")
    print("  • Increases effective training set size")
    print("  • Improves model generalization")
    print("  • Reduces overfitting")
    print("  • Makes model robust to variations in input data")


if __name__ == "__main__":
    main()
