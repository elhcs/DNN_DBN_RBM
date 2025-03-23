import numpy as np
import matplotlib.pyplot as plt
from rbm_implementation import RBM
from dbn_implementation import DBN

class DNN:
    def __init__(self, layer_sizes):
        """Initialize a Deep Neural Network with random weights.
        
        Args:
            layer_sizes: List of layer sizes (including input and output layers)
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize RBMs for each layer
        self.rbms = []
        for i in range(self.n_layers):
            self.rbms.append(RBM(layer_sizes[i], layer_sizes[i+1]))
    
    def pretrain(self, data, epochs_per_layer, learning_rate, batch_size):
        """Pretrain the hidden layers of the DNN using RBMs.
        
        Args:
            data: Training data (n_samples, n_input)
            epochs_per_layer: Number of epochs for each layer
            learning_rate: Learning rate
            batch_size: Size of mini-batches
            
        Returns:
            Pretrained DNN
        """
        # Create a DBN with the hidden layers of the DNN
        hidden_layer_sizes = self.layer_sizes[:-1]  # Exclude output layer
        dbn = DBN(hidden_layer_sizes)
        
        # Pretrain the DBN
        dbn.pretrain(data, epochs_per_layer, learning_rate, batch_size)
        
        # Copy the pretrained weights to the DNN's hidden layers
        for i in range(len(dbn.rbms)):
            self.rbms[i].W = dbn.rbms[i].W.copy()
            self.rbms[i].a = dbn.rbms[i].a.copy()
            self.rbms[i].b = dbn.rbms[i].b.copy()
        
        return self
    
    def softmax(self, x):
        """Compute softmax activation.
        
        Args:
            x: Input array
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, data):
        """Forward pass through the network.
        
        Args:
            data: Input data (n_samples, n_input)
            
        Returns:
            List of activations for each layer and output probabilities
        """
        activations = [data]
        layer_input = data
        
        # Pass through hidden layers
        for i in range(self.n_layers - 1):
            rbm = self.rbms[i]
            layer_output = rbm.sigmoid(np.dot(layer_input, rbm.W) + rbm.b)
            activations.append(layer_output)
            layer_input = layer_output
        
        # Pass through output layer (softmax)
        output_rbm = self.rbms[-1]
        logits = np.dot(layer_input, output_rbm.W) + output_rbm.b
        probs = self.softmax(logits)
        activations.append(probs)
        
        return activations
    
    def backpropagation(self, data, labels, epochs, learning_rate, batch_size):
        """Train the DNN using backpropagation.
        
        Args:
            data: Training data (n_samples, n_input)
            labels: One-hot encoded labels (n_samples, n_output)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Size of mini-batches
            
        Returns:
            List of cross-entropy losses per epoch
        """
        n_samples = len(data)
        n_batches = n_samples // batch_size
        
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            batch_losses = []
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)
                batch_data = data[indices[start:end]]
                batch_labels = labels[indices[start:end]]
                
                # Forward pass
                activations = self.forward_pass(batch_data)
                output = activations[-1]
                
                # Compute cross-entropy loss
                batch_loss = -np.mean(np.sum(batch_labels * np.log(np.clip(output, 1e-10, 1.0)), axis=1))
                batch_losses.append(batch_loss)
                
                # Backpropagation
                # Output layer error
                delta = output - batch_labels
                
                # Update output layer weights
                output_rbm = self.rbms[-1]
                output_rbm.W -= learning_rate * np.dot(activations[-2].T, delta) / len(batch_data)
                output_rbm.b -= learning_rate * np.mean(delta, axis=0)
                
                # Propagate error backward through hidden layers
                for i in range(self.n_layers - 2, -1, -1):
                    rbm_next = self.rbms[i+1]
                    rbm = self.rbms[i]
                    
                    # Compute error for current layer
                    delta = np.dot(delta, rbm_next.W.T) * activations[i+1] * (1 - activations[i+1])
                    
                    # Update weights and biases
                    rbm.W -= learning_rate * np.dot(activations[i].T, delta) / len(batch_data)
                    rbm.b -= learning_rate * np.mean(delta, axis=0)
            
            # Compute average loss for the epoch
            mean_loss = np.mean(batch_losses)
            losses.append(mean_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Cross-Entropy Loss: {mean_loss:.6f}")
            
            # Evaluate performance on training data
            if (epoch + 1) % 10 == 0 or epoch == 0:
                preds = np.argmax(self.forward_pass(data)[-1], axis=1)
                true_labels = np.argmax(labels, axis=1)
                accuracy = np.mean(preds == true_labels)
                print(f"  Training Accuracy: {accuracy:.4f}")
        
        return losses
    
    def predict(self, data):
        """Predict class labels for the input data.
        
        Args:
            data: Input data (n_samples, n_input)
            
        Returns:
            Predicted labels (n_samples,)
        """
        probs = self.forward_pass(data)[-1]
        return np.argmax(probs, axis=1)
    
    def evaluate(self, data, labels):
        """Evaluate the DNN on the given data.
        
        Args:
            data: Input data (n_samples, n_input)
            labels: One-hot encoded labels or integer labels (n_samples, n_output) or (n_samples,)
            
        Returns:
            Error rate
        """
        preds = self.predict(data)
        
        if len(labels.shape) > 1:
            true_labels = np.argmax(labels, axis=1)
        else:
            true_labels = labels
        
        error_rate = 1 - np.mean(preds == true_labels)
        return error_rate


def init_DNN(layer_sizes):
    """Initialize a DNN with the given layer sizes.
    
    Args:
        layer_sizes: List of layer sizes (including input and output layers)
        
    Returns:
        Initialized DNN
    """
    return DNN(layer_sizes)


def pretrain_DNN(dnn, data, epochs_per_layer, learning_rate, batch_size):
    """Pretrain a DNN using RBMs.
    
    Args:
        dnn: DNN to pretrain
        data: Training data
        epochs_per_layer: Number of epochs for each layer
        learning_rate: Learning rate
        batch_size: Size of mini-batches
        
    Returns:
        Pretrained DNN
    """
    return dnn.pretrain(data, epochs_per_layer, learning_rate, batch_size)


def calcul_softmax(rbm, data):
    """Compute softmax probabilities for the output layer.
    
    Args:
        rbm: RBM of the output layer
        data: Input data to the output layer
        
    Returns:
        Softmax probabilities
    """
    logits = np.dot(data, rbm.W) + rbm.b
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def entree_sortie_reseau(dnn, data):
    """Compute activations for all layers of the DNN.
    
    Args:
        dnn: DNN
        data: Input data
        
    Returns:
        List of activations for each layer and output probabilities
    """
    return dnn.forward_pass(data)


def retropropagation(dnn, data, labels, epochs, learning_rate, batch_size):
    """Train a DNN using backpropagation.
    
    Args:
        dnn: DNN to train
        data: Training data
        labels: One-hot encoded labels
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Size of mini-batches
        
    Returns:
        Trained DNN and list of losses
    """
    losses = dnn.backpropagation(data, labels, epochs, learning_rate, batch_size)
    return dnn, losses


def test_DNN(dnn, test_data, test_labels):
    """Test a DNN on the given data.
    
    Args:
        dnn: DNN to test
        test_data: Test data
        test_labels: Test labels (one-hot encoded or integer labels)
        
    Returns:
        Error rate
    """
    return dnn.evaluate(test_data, test_labels)
