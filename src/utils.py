import os
import shutil
import random
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np

def create_data_directory(data_dir, class_names):
    """
    Create directory structure for the dataset
    
    Args:
        data_dir: Root directory for the dataset
        class_names: List of class names
    """
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                
    print(f"Created directory structure at {data_dir}")

def split_dataset(source_dir, data_dir, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split a dataset into train, validation, and test sets
    
    Args:
        source_dir: Source directory containing class folders with images
        data_dir: Target directory to create train/valid/test structure
        train_ratio: Proportion of data for training
        valid_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get class names
    class_names = [d for d in os.listdir(source_dir) 
                   if os.path.isdir(os.path.join(source_dir, d))]
    
    # Create directory structure
    create_data_directory(data_dir, class_names)
    
    # Process each class
    for class_name in class_names:
        # Get all image files
        source_class_dir = os.path.join(source_dir, class_name)
        image_files = [f for f in os.listdir(source_class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate splits
        n_total = len(image_files)
        n_train = int(train_ratio * n_total)
        n_valid = int(valid_ratio * n_total)
        
        # Split into train, validation, and test sets
        train_files = image_files[:n_train]
        valid_files = image_files[n_train:n_train + n_valid]
        test_files = image_files[n_train + n_valid:]
        
        # Copy files to respective directories
        for files, split in zip([train_files, valid_files, test_files], ['train', 'valid', 'test']):
            target_dir = os.path.join(data_dir, split, class_name)
            for file in files:
                source_path = os.path.join(source_class_dir, file)
                target_path = os.path.join(target_dir, file)
                shutil.copy2(source_path, target_path)
        
        print(f"Class {class_name}: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test")
    
    print("Dataset split complete!")

def show_batch(inputs, classes, class_names):
    """
    Show a batch of images with their labels
    
    Args:
        inputs: Batch of images as tensors
        classes: Labels for the images
        class_names: List of class names
    """
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    # Convert tensor to numpy for display
    inp = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    # Display images
    plt.figure(figsize=(12, 8))
    plt.imshow(inp)
    plt.title([class_names[x] for x in classes])
    plt.axis('off')
    plt.show()

def visualize_model_predictions(model, dataloaders, class_names, device, num_images=6):
    """
    Visualize model predictions on validation set
    
    Args:
        model: Trained model
        dataloaders: DataLoaders dictionary
        class_names: List of class names
        device: Device to run the model on
        num_images: Number of images to visualize
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['valid']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//3 + 1, 3, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\ntrue: {class_names[labels[j]]}')
                
                # Convert tensor to numpy for display
                inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                ax.imshow(inp)
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    return
    
    model.train(mode=was_training)
    plt.tight_layout()
    plt.show()