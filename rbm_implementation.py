import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import gzip
import struct

class RBM:
    def __init__(self, n_visible, n_hidden):
        """Initialize a Restricted Boltzmann Machine with random weights and zero biases.
        
        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Initialize weights and biases
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)  # Bias for visible units
        self.b = np.zeros(n_hidden)   # Bias for hidden units
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def visible_to_hidden(self, v):
        """Compute activation probabilities of hidden units given visible units.
        
        Args:
            v: Visible units (batch_size, n_visible)
            
        Returns:
            Probabilities of hidden units being activated
        """
        return self.sigmoid(np.dot(v, self.W) + self.b)
    
    def hidden_to_visible(self, h):
        """Compute activation probabilities of visible units given hidden units.
        
        Args:
            h: Hidden units (batch_size, n_hidden)
            
        Returns:
            Probabilities of visible units being activated
        """
        return self.sigmoid(np.dot(h, self.W.T) + self.a)
    
    def sample_hidden(self, v):
        """Sample hidden units given visible units.
        
        Args:
            v: Visible units (batch_size, n_visible)
            
        Returns:
            Binary hidden units
        """
        p_h = self.visible_to_hidden(v)
        return (np.random.random(p_h.shape) < p_h).astype(float), p_h
    
    def sample_visible(self, h):
        """Sample visible units given hidden units.
        
        Args:
            h: Hidden units (batch_size, n_hidden)
            
        Returns:
            Binary visible units
        """
        p_v = self.hidden_to_visible(h)
        return (np.random.random(p_v.shape) < p_v).astype(float), p_v
    
    def contrastive_divergence(self, v_data, learning_rate, k=1):
        """Perform one step of contrastive divergence learning.
        
        Args:
            v_data: Training data (batch_size, n_visible)
            learning_rate: Learning rate
            k: Number of Gibbs sampling steps
            
        Returns:
            Reconstruction error
        """
        # Positive phase
        h_data, ph_data = self.sample_hidden(v_data)
        
        # Negative phase (k steps of Gibbs sampling)
        v_model = v_data.copy()
        for _ in range(k):
            h_model, _ = self.sample_hidden(v_model)
            v_model, pv_model = self.sample_visible(h_model)
        
        h_model, ph_model = self.sample_hidden(v_model)
        
        # Compute gradients and update parameters
        dW = np.dot(v_data.T, ph_data) - np.dot(v_model.T, ph_model)
        da = np.mean(v_data - v_model, axis=0)
        db = np.mean(ph_data - ph_model, axis=0)
        
        self.W += learning_rate * dW / len(v_data)
        self.a += learning_rate * da
        self.b += learning_rate * db
        
        # Compute reconstruction error
        v_recon, _ = self.sample_visible(h_data)
        return np.mean((v_data - v_recon) ** 2)
    
    def train(self, data, epochs, learning_rate, batch_size):
        """Train the RBM with contrastive divergence.
        
        Args:
            data: Training data (n_samples, n_visible)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Size of mini-batches
            
        Returns:
            List of reconstruction errors per epoch
        """
        n_samples = len(data)
        n_batches = n_samples // batch_size
        
        errors = []
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            batch_errors = []
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                batch_data = data[indices[start:end]]
                
                error = self.contrastive_divergence(batch_data, learning_rate)
                batch_errors.append(error)
            
            mean_error = np.mean(batch_errors)
            errors.append(mean_error)
            
            print(f"Epoch {epoch+1}/{epochs}, Reconstruction Error: {mean_error:.6f}")
        
        return errors
    
    def generate_samples(self, n_samples, n_gibbs_steps):
        """Generate samples from the RBM using Gibbs sampling.
        
        Args:
            n_samples: Number of samples to generate
            n_gibbs_steps: Number of Gibbs sampling steps
            
        Returns:
            Generated samples
        """
        # Start with random visible units
        v = np.random.randint(0, 2, (n_samples, self.n_visible)).astype(float)
        
        # Perform Gibbs sampling
        for _ in range(n_gibbs_steps):
            h, _ = self.sample_hidden(v)
            v, _ = self.sample_visible(h)
        
        return v


def lire_alpha_digit(characters=None):
    """Read binary alpha digit images from the dataset.
    
    Args:
        characters: List of character indices to read (0-35). If None, read all.
        
    Returns:
        Matrix of images (n_samples, n_pixels)
    """
    # Load .mat file
    data = loadmat('binaryalphadigs.mat')
    digits = data['dat']
    
    if characters is None:
        characters = range(36)  # All characters (0-9, A-Z)
    elif isinstance(characters, int):
        characters = [characters]
    
    images = []
    for idx in characters:
        for img in digits[idx]:
            # Flatten 20x16 images to 320-d vectors
            images.append(img.flatten())
    
    return np.array(images)


def plot_images(images, shape=(20, 16), n_cols=10, title=None):
    """Plot multiple binary images in a grid.
    
    Args:
        images: Array of images to plot (n_images, n_pixels)
        shape: Shape of each image (height, width)
        n_cols: Number of columns in the grid
        title: Title for the plot
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    plt.figure(figsize=(1.5 * n_cols, 1.5 * n_rows))
    
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i].reshape(shape), cmap='binary')
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    plt.show()


def generer_image_RBM(rbm, n_samples, n_gibbs_steps, shape=(20, 16)):
    """Generate and display images from an RBM.
    
    Args:
        rbm: Trained RBM
        n_samples: Number of samples to generate
        n_gibbs_steps: Number of Gibbs sampling steps
        shape: Shape of each image (height, width)
        
    Returns:
        Generated images
    """
    samples = rbm.generate_samples(n_samples, n_gibbs_steps)
    plot_images(samples, shape, title="Generated Images from RBM")
    return samples
