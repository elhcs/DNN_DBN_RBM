import numpy as np
import matplotlib.pyplot as plt
from rbm_implementation import RBM, plot_images

class DBN:
    def __init__(self, layer_sizes):
        """Initialize a Deep Belief Network with random weights.
        
        Args:
            layer_sizes: List of layer sizes (including input layer)
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize RBMs for each layer
        self.rbms = []
        for i in range(self.n_layers):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i+1]))
    
    def pretrain(self, data, epochs_per_layer, learning_rate, batch_size):
        """Pretrain the DBN using greedy layer-wise procedure.
        
        Args:
            data: Training data (n_samples, n_input)
            epochs_per_layer: Number of epochs for each layer
            learning_rate: Learning rate
            batch_size: Size of mini-batches
            
        Returns:
            List of reconstruction errors per layer
        """
        layer_input = data
        all_errors = []
        
        # Train each RBM layer
        for i, rbm in enumerate(self.rbms):
            print(f"\nPretraining layer {i+1}/{self.n_layers}")
            
            # Train the current RBM
            errors = rbm.train(layer_input, epochs_per_layer, learning_rate, batch_size)
            all_errors.append(errors)
            
            # Transform data for the next layer
            if i < self.n_layers - 1:
                layer_input = rbm.visible_to_hidden(layer_input)
                print(f"Transformed data shape for next layer: {layer_input.shape}")
        
        return all_errors
    
    def forward_pass(self, data):
        """Forward pass through the network.
        
        Args:
            data: Input data (n_samples, n_input)
            
        Returns:
            List of activations for each layer
        """
        activations = [data]
        layer_input = data
        
        for rbm in self.rbms:
            layer_output = rbm.visible_to_hidden(layer_input)
            activations.append(layer_output)
            layer_input = layer_output
        
        return activations
    
    def generate_samples(self, n_samples, n_gibbs_steps, shape=None):
        """Generate samples from the DBN.
        
        Args:
            n_samples: Number of samples to generate
            n_gibbs_steps: Number of Gibbs sampling steps
            shape: Shape of each image for plotting (height, width)
            
        Returns:
            Generated samples
        """
        # Start with random activation in the deepest layer
        h = np.random.randint(0, 2, (n_samples, self.layer_sizes[-1])).astype(float)
        
        # Propagate backward through the network
        for i in range(self.n_layers - 1, -1, -1):
            rbm = self.rbms[i]
            v, _ = rbm.sample_visible(h)
            
            # Perform Gibbs sampling at the lowest level
            if i == 0:
                for _ in range(n_gibbs_steps):
                    h, _ = rbm.sample_hidden(v)
                    v, _ = rbm.sample_visible(h)
            else:
                h = v
        
        samples = v
        
        # Plot samples if shape is provided
        if shape is not None:
            plot_images(samples, shape, title="Generated Images from DBN")
        
        return samples


def generer_image_DBN(dbn, n_samples, n_gibbs_steps, shape=(20, 16)):
    """Generate and display images from a DBN.
    
    Args:
        dbn: Trained DBN
        n_samples: Number of samples to generate
        n_gibbs_steps: Number of Gibbs sampling steps
        shape: Shape of each image (height, width)
        
    Returns:
        Generated images
    """
    return dbn.generate_samples(n_samples, n_gibbs_steps, shape)
