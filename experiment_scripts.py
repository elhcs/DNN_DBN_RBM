import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import gzip
import struct
import time

# Import our modules
from rbm_implementation import RBM, lire_alpha_digit, generer_image_RBM, plot_images
from dbn_implementation import DBN, generer_image_DBN
from dnn_implementation import DNN, init_DNN, pretrain_DNN, retropropagation, test_DNN
from mnist_loader import load_mnist, get_mnist_subset

# ===== Script principal_RBM_alpha =====
def principal_RBM_alpha():
    print("=== RBM on Binary Alpha Digits ===")
    
    # Parameters
    n_hidden = 100
    epochs = 100
    learning_rate = 0.1
    batch_size = 10
    n_gibbs_steps = 5
    
    # Load data
    print("Loading data...")
    # Use digits 0-9 for the experiment
    characters = list(range(10))
    data = lire_alpha_digit(characters)
    
    # Display some original images
    print("Original images:")
    plot_images(data[:10], title="Original Images")
    
    # Initialize RBM
    print("Initializing RBM...")
    n_visible = data.shape[1]  # 320 for 20x16 images
    rbm = RBM(n_visible, n_hidden)
    
    # Train RBM
    print("Training RBM...")
    errors = rbm.train(data, epochs, learning_rate, batch_size)
    
    # Plot training error
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), errors)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('RBM Training Error')
    plt.grid(True)
    plt.savefig('rbm_training_error.png')
    plt.show()
    
    # Generate images
    print("Generating images from RBM...")
    generated_images = generer_image_RBM(rbm, 10, n_gibbs_steps)
    
    print("RBM training and generation completed.")
    
    return rbm, errors


# ===== Script principal_DBN_alpha =====
def principal_DBN_alpha():
    print("\n=== DBN on Binary Alpha Digits ===")
    
    # Parameters
    layer_sizes = [320, 200, 100]  # [input, hidden1, hidden2]
    epochs_per_layer = 100
    learning_rate = 0.1
    batch_size = 10
    n_gibbs_steps = 10
    
    # Load data
    print("Loading data...")
    # Use digits 0-9 for the experiment
    characters = list(range(10))
    data = lire_alpha_digit(characters)
    
    # Display some original images
    print("Original images:")
    plot_images(data[:10], title="Original Images")
    
    # Initialize DBN
    print("Initializing DBN...")
    dbn = DBN(layer_sizes)
    
    # Pretrain DBN
    print("Pretraining DBN...")
    errors = dbn.pretrain(data, epochs_per_layer, learning_rate, batch_size)
    
    # Plot training errors for each layer
    plt.figure(figsize=(12, 8))
    for i, layer_errors in enumerate(errors):
        plt.subplot(1, len(errors), i+1)
        plt.plot(range(1, epochs_per_layer + 1), layer_errors)
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Error')
        plt.title(f'Layer {i+1} Training Error')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('dbn_training_errors.png')
    plt.show()
    
    # Generate images
    print("Generating images from DBN...")
    generated_images = generer_image_DBN(dbn, 10, n_gibbs_steps)
    
    print("DBN training and generation completed.")
    
    return dbn, errors


# ===== Script principal_DNN_MNIST =====
def principal_DNN_MNIST():
    print("\n=== DNN on MNIST Dataset ===")
    
    # Parameters
    layer_sizes = [784, 500, 500, 10]  # [input, hidden1, hidden2, output]
    pretrain_epochs = 50
    train_epochs = 100
    learning_rate_pretrain = 0.01
    learning_rate_train = 0.1
    batch_size = 100
    
    # Load MNIST data
    print("Loading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Create two identical networks
    print("Initializing networks...")
    dnn_pretrained = init_DNN(layer_sizes)
    dnn_random = init_DNN(layer_sizes)
    
    # Pretrain first network
    print("Pretraining first network...")
    dnn_pretrained = pretrain_DNN(dnn_pretrained, train_images, pretrain_epochs, learning_rate_pretrain, batch_size)
    
    # Train both networks with backpropagation
    print("Training pretrained network...")
    dnn_pretrained, losses_pretrained = retropropagation(dnn_pretrained, train_images, train_labels, train_epochs, learning_rate_train, batch_size)
    
    print("Training randomly initialized network...")
    dnn_random, losses_random = retropropagation(dnn_random, train_images, train_labels, train_epochs, learning_rate_train, batch_size)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, train_epochs + 1), losses_pretrained, label='Pretrained')
    plt.plot(range(1, train_epochs + 1), losses_random, label='Random Init')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('DNN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('dnn_training_losses.png')
    plt.show()
    
    # Test networks
    print("Testing networks...")
    error_pretrained = test_DNN(dnn_pretrained, test_images, test_labels)
    error_random = test_DNN(dnn_random, test_images, test_labels)
    
    print(f"Pretrained network error rate: {error_pretrained:.4f}")
    print(f"Random initialized network error rate: {error_random:.4f}")
    
    # View some predictions
    print("Sample predictions (pretrained network):")
    sample_indices = np.random.choice(len(test_images), 5)
    sample_images = test_images[sample_indices]
    sample_labels = np.argmax(test_labels[sample_indices], axis=1)
    
    outputs = dnn_pretrained.forward_pass(sample_images)[-1]
    preds = np.argmax(outputs, axis=1)
    
    for i in range(len(sample_indices)):
        print(f"True label: {sample_labels[i]}, Predicted: {preds[i]}")
        print(f"Output probabilities: {outputs[i]}")
    
    return dnn_pretrained, dnn_random, error_pretrained, error_random


# ===== Experiment 1: Effect of Network Depth =====
def experiment_network_depth():
    print("\n=== Experiment: Effect of Network Depth ===")
    
    # Load MNIST data
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Parameters
    max_hidden_layers = 4
    hidden_size = 200
    pretrain_epochs = 30
    train_epochs = 50
    learning_rate_pretrain = 0.01
    learning_rate_train = 0.1
    batch_size = 100
    
    # Results storage
    depths = range(1, max_hidden_layers + 1)
    pretrained_errors = []
    random_errors = []
    
    for n_hidden_layers in depths:
        print(f"\nTesting with {n_hidden_layers} hidden layers")
        
        # Create layer sizes
        layer_sizes = [784] + [hidden_size] * n_hidden_layers + [10]
        
        # Initialize networks
        dnn_pretrained = init_DNN(layer_sizes)
        dnn_random = init_DNN(layer_sizes)
        
        # Pretrain first network
        dnn_pretrained = pretrain_DNN(dnn_pretrained, train_images, pretrain_epochs, learning_rate_pretrain, batch_size)
        
        # Train both networks
        dnn_pretrained, _ = retropropagation(dnn_pretrained, train_images, train_labels, train_epochs, learning_rate_train, batch_size)
        dnn_random, _ = retropropagation(dnn_random, train_images, train_labels, train_epochs, learning_rate_train, batch_size)
        
        # Test networks
        error_pretrained = test_DNN(dnn_pretrained, test_images, test_labels)
        error_random = test_DNN(dnn_random, test_images, test_labels)
        
        pretrained_errors.append(error_pretrained)
        random_errors.append(error_random)
        
        print(f"Pretrained network error rate: {error_pretrained:.4f}")
        print(f"Random initialized network error rate: {error_random:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(depths, pretrained_errors, 'b-o', label='Pretrained')
    plt.plot(depths, random_errors, 'r-o', label='Random Init')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Error Rate')
    plt.title('Effect of Network Depth on Error Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_network_depth.png')
    plt.show()
    
    return depths, pretrained_errors, random_errors


# ===== Experiment 2: Effect of Hidden Layer Size =====
def experiment_hidden_size():
    print("\n=== Experiment: Effect of Hidden Layer Size ===")
    
    # Load MNIST data
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # Parameters
    hidden_sizes = [100, 300, 500, 700]
    n_hidden_layers = 2
    pretrain_epochs = 30
    train_epochs = 50
    learning_rate_pretrain = 0.01
    learning_rate_train = 0.1
    batch_size = 100
    
    # Results storage
    pretrained_errors = []
    random_errors = []
    
    for hidden_size in hidden_sizes:
        print(f"\nTesting with hidden size {hidden_size}")
        
        # Create layer sizes
        layer_sizes = [784] + [hidden_size] * n_hidden_layers + [10]
        
        # Initialize networks
        dnn_pretrained = init_DNN(layer_sizes)
        dnn_random = init_DNN(layer_sizes)
        
        # Pretrain first network
        dnn_pretrained = pretrain_DNN(dnn_pretrained, train_images, pretrain_epochs, learning_rate_pretrain, batch_size)
        
        # Train both networks
        dnn_pretrained, _ = retropropagation(dnn_pretrained, train_images, train_labels, train_epochs, learning_rate_train, batch_size)
        dnn_random, _ = retropropagation(dnn_random, train_images, train_labels, train_epochs, learning_rate_train, batch_size)
        
        # Test networks
        error_pretrained = test_DNN(dnn_pretrained, test_images, test_labels)
        error_random = test_DNN(dnn_random, test_images, test_labels)
        
        pretrained_errors.append(error_pretrained)
        random_errors.append(error_random)
        
        print(f"Pretrained network error rate: {error_pretrained:.4f}")
        print(f"Random initialized network error rate: {error_random:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_sizes, pretrained_errors, 'b-o', label='Pretrained')
    plt.plot(hidden_sizes, random_errors, 'r-o', label='Random Init')
    plt.xlabel('Number of Neurons per Hidden Layer')
    plt.ylabel('Error Rate')
    plt.title('Effect of Hidden Layer Size on Error Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_hidden_size.png')
    plt.show()
    
    return hidden_sizes, pretrained_errors, random_errors


# ===== Experiment 3: Effect of Training Data Size =====
def experiment_data_size():
    print("\n=== Experiment: Effect of Training Data Size ===")
    
    # Load MNIST data
    full_train_images, full_train_labels, test_images, test_labels = load_mnist()
    
    # Parameters
    data_sizes = [1000, 3000, 7000, 10000, 30000, 60000]
    layer_sizes = [784, 200, 200, 10]
    pretrain_epochs = 30
    train_epochs = 50
    learning_rate_pretrain = 0.01
    learning_rate_train = 0.1
    batch_size = 100
    
    # Results storage
    pretrained_errors = []
    random_errors = []
    
    for data_size in data_sizes:
        print(f"\nTesting with {data_size} training samples")
        
        # Get subset of training data
        if data_size < len(full_train_images):
            train_images, train_labels = get_mnist_subset(full_train_images, full_train_labels, data_size)
        else:
            train_images, train_labels = full_train_images, full_train_labels
        
        # Initialize networks
        dnn_pretrained = init_DNN(layer_sizes)
        dnn_random = init_DNN(layer_sizes)
        
        # Adjust batch size for small datasets
        actual_batch_size = min(batch_size, data_size // 10)
        
        # Pretrain first network
        dnn_pretrained = pretrain_DNN(dnn_pretrained, train_images, pretrain_epochs, learning_rate_pretrain, actual_batch_size)
        
        # Train both networks
        dnn_pretrained, _ = retropropagation(dnn_pretrained, train_images, train_labels, train_epochs, learning_rate_train, actual_batch_size)
        dnn_random, _ = retropropagation(dnn_random, train_images, train_labels, train_epochs, learning_rate_train, actual_batch_size)
        
        # Test networks
        error_pretrained = test_DNN(dnn_pretrained, test_images, test_labels)
        error_random = test_DNN(dnn_random, test_images, test_labels)
        
        pretrained_errors.append(error_pretrained)
        random_errors.append(error_random)
        
        print(f"Pretrained network error rate: {error_pretrained:.4f}")
        print(f"Random initialized network error rate: {error_random:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(data_sizes, pretrained_errors, 'b-o', label='Pretrained')
    plt.plot(data_sizes, random_errors, 'r-o', label='Random Init')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Error Rate')
    plt.title('Effect of Training Data Size on Error Rate')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiment_data_size.png')
    plt.show()
    
    return data_sizes, pretrained_errors, random_errors


# ===== Main function =====
def run_all_experiments():
    # Run simple demos first
    print("Running RBM on Binary Alpha Digits")
    principal_RBM_alpha()
    
    print("\nRunning DBN on Binary Alpha Digits")
    principal_DBN_alpha()
    
    print("\nRunning basic DNN on MNIST")
    principal_DNN_MNIST()
    
    # Run experiments
    print("\nRunning network depth experiment")
    depths, depth_pretrained, depth_random = experiment_network_depth()
    
    print("\nRunning hidden size experiment")
    sizes, size_pretrained, size_random = experiment_hidden_size()
    
    print("\nRunning data size experiment")
    data_sizes, data_pretrained, data_random = experiment_data_size()
    
    # Print summary
    print("\n===== Experiment Summary =====")
    
    print("\nEffect of Network Depth:")
    for i, depth in enumerate(depths):
        print(f"  {depth} layers: Pretrained = {depth_pretrained[i]:.4f}, Random = {depth_random[i]:.4f}")
    
    print("\nEffect of Hidden Layer Size:")
    for i, size in enumerate(sizes):
        print(f"  {size} neurons: Pretrained = {size_pretrained[i]:.4f}, Random = {size_random[i]:.4f}")
    
    print("\nEffect of Training Data Size:")
    for i, size in enumerate(data_sizes):
        print(f"  {size} samples: Pretrained = {data_pretrained[i]:.4f}, Random = {data_random[i]:.4f}")


if __name__ == "__main__":
    # Uncomment the specific function you want to run
    # principal_RBM_alpha()
    # principal_DBN_alpha()
    # principal_DNN_MNIST()
    # experiment_network_depth()
    # experiment_hidden_size()
    # experiment_data_size()
    
    # Or run all experiments
    run_all_experiments()
