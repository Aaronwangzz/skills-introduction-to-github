"""
Simple Neural Network with PyTorch
This script demonstrates building and training a basic neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class SimpleNN(nn.Module):
    """A simple feedforward neural network"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def generate_data(n_samples=1000):
    """Generate synthetic data for regression"""
    X = np.linspace(-10, 10, n_samples).reshape(-1, 1)
    y = 2 * X + 3 + np.random.randn(n_samples, 1) * 2
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    return X_tensor, y_tensor, X, y

def train_model(model, X, y, epochs=100, learning_rate=0.01):
    """Train the neural network"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    print("Training the model...")
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    return losses

def visualize_results(X, y, model, losses):
    """Visualize training results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Loss curve
    axes[0].plot(losses, color='blue')
    axes[0].set_title('Training Loss Over Time')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    model.eval()
    with torch.no_grad():
        predictions = model(X).numpy()
    
    axes[1].scatter(X.numpy(), y.numpy(), alpha=0.5, label='Actual', s=10)
    axes[1].plot(X.numpy(), predictions, color='red', linewidth=2, label='Predicted')
    axes[1].set_title('Actual vs Predicted Values')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pytorch_training_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'pytorch_training_results.png'")
    plt.show()

def main():
    """Main function to run the PyTorch example"""
    print("=" * 50)
    print("Simple Neural Network with PyTorch")
    print("=" * 50)
    
    # Generate data
    print("\nGenerating synthetic data...")
    X_tensor, y_tensor, X_np, y_np = generate_data()
    print(f"Data shape: X={X_tensor.shape}, y={y_tensor.shape}")
    
    # Create model
    input_size = 1
    hidden_size = 10
    output_size = 1
    model = SimpleNN(input_size, hidden_size, output_size)
    
    print(f"\nModel architecture:")
    print(model)
    
    # Train model
    losses = train_model(model, X_tensor, y_tensor, epochs=100, learning_rate=0.01)
    
    # Visualize results
    visualize_results(X_tensor, y_tensor, model, losses)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
