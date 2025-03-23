import numpy as np
import gzip
import os
import struct
from sklearn.datasets import fetch_openml

def load_mnist():
    """Load MNIST dataset.
    
    Returns:
        train_images: Training images (60000, 784)
        train_labels: Training labels (60000, 10)
        test_images: Test images (10000, 784)
        test_labels: Test labels (10000, 10)
    """
    # # Define file paths
    # files = {
    #     'train_img': 'train-images-idx3-ubyte.gz',
    #     'train_lbl': 'train-labels-idx1-ubyte.gz',
    #     'test_img': 't10k-images-idx3-ubyte.gz',
    #     'test_lbl': 't10k-labels-idx1-ubyte.gz'
    # }
    
    

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    train_images, train_labels = mnist.data[:60000], mnist.target[:60000]
    test_images, test_labels = mnist.data[60000:], mnist.target[60000:]

    
    # Convert grayscale images to binary
    train_images = (train_images > 127).astype(float)
    test_images = (test_images > 127).astype(float)
    
    # Convert labels to one-hot encoding
    train_labels = train_labels.astype(int)  # Ensure integer type
    test_labels = test_labels.astype(int)  # Ensure integer type
    # Convert labels to one-hot encoding
    train_labels_one_hot = np.zeros((train_labels.size, 10))
    train_labels_one_hot[np.arange(train_labels.size), train_labels] = 1

    test_labels_one_hot = np.zeros((test_labels.size, 10))
    test_labels_one_hot[np.arange(test_labels.size), test_labels] = 1



    
    return train_images, train_labels_one_hot, test_images, test_labels_one_hot


def _read_images(filename):
    """Read images from MNIST file.
    
    Args:
        filename: Name of the file
        
    Returns:
        Images as a numpy array
    """
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(-1, rows * cols)
    return images


def _read_labels(filename):
    """Read labels from MNIST file.
    
    Args:
        filename: Name of the file
        
    Returns:
        Labels as a numpy array
    """
    with gzip.open(filename, 'rb') as f:
        magic, num = struct.unpack('>2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def get_mnist_subset(train_images, train_labels, n_samples, balanced=True):
    """Get a subset of MNIST training data.
    
    Args:
        train_images: Full training images
        train_labels: Full training labels (one-hot encoded)
        n_samples: Number of samples to keep
        balanced: Whether to keep a balanced set of classes
        
    Returns:
        Subset of training images and labels
    """
    if balanced:
        # Get indices for each class
        indices_by_class = []
        for i in range(10):
            indices = np.where(train_labels[:, i] == 1)[0]
            indices_by_class.append(indices)
        
        # Determine number of samples per class
        samples_per_class = n_samples // 10
        
        # Select samples from each class
        selected_indices = []
        for indices in indices_by_class:
            if len(indices) > samples_per_class:
                selected_indices.extend(np.random.choice(indices, samples_per_class, replace=False))
            else:
                selected_indices.extend(indices)
        
        # Make sure we have exactly n_samples
        if len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            all_indices = np.arange(len(train_images))
            unused_indices = np.setdiff1d(all_indices, selected_indices)
            selected_indices.extend(np.random.choice(unused_indices, remaining, replace=False))
        
        selected_indices = np.array(selected_indices)
        
    else:
        # Randomly select n_samples
        selected_indices = np.random.choice(len(train_images), n_samples, replace=False)
    
    # Get the subset
    subset_images = train_images[selected_indices]
    subset_labels = train_labels[selected_indices]
    
    return subset_images, subset_labels
