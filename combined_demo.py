"""
Combined Example: PyTorch, Pandas, and Matplotlib
This script demonstrates using all three libraries together for a complete ML workflow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ClassificationNN(nn.Module):
    """Neural network for binary classification"""
    
    def __init__(self, input_size, hidden_size):
        super(ClassificationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def create_dataset():
    """Create a dataset using pandas"""
    print("Creating dataset with Pandas...")
    
    # Generate synthetic data
    n_samples = 500
    class_0_x1 = np.random.randn(n_samples // 2) + 2
    class_0_x2 = np.random.randn(n_samples // 2) + 2
    
    class_1_x1 = np.random.randn(n_samples // 2) - 2
    class_1_x2 = np.random.randn(n_samples // 2) - 2
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_1': np.concatenate([class_0_x1, class_1_x1]),
        'feature_2': np.concatenate([class_0_x2, class_1_x2]),
        'label': np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    })
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df

def prepare_data(df):
    """Prepare data for PyTorch"""
    X = df[['feature_1', 'feature_2']].values
    y = df['label'].values.reshape(-1, 1)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Split into train and test
    train_size = int(0.8 * len(X_tensor))
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]
    
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, X_test, y_test, epochs=50):
    """Train the classification model"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses = []
    test_losses = []
    accuracies = []
    
    print("\nTraining the model...")
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
            
            # Calculate accuracy
            predictions = (test_outputs > 0.5).float()
            accuracy = (predictions == y_test).float().mean()
            accuracies.append(accuracy.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Test Loss: {test_loss.item():.4f}, '
                  f'Accuracy: {accuracy.item():.4f}')
    
    return train_losses, test_losses, accuracies

def visualize_results(df, model, X_train, X_test, y_test, train_losses, test_losses, accuracies):
    """Create comprehensive visualizations using Matplotlib"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Original data distribution
    ax1 = plt.subplot(2, 3, 1)
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    ax1.scatter(class_0['feature_1'], class_0['feature_2'], 
                alpha=0.5, label='Class 0', s=20)
    ax1.scatter(class_1['feature_1'], class_1['feature_2'], 
                alpha=0.5, label='Class 1', s=20)
    ax1.set_title('Data Distribution')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Test Loss
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(train_losses, label='Train Loss', color='blue')
    ax2.plot(test_losses, label='Test Loss', color='red')
    ax2.set_title('Training Progress')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(accuracies, color='green')
    ax3.set_title('Test Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Decision boundary
    ax4 = plt.subplot(2, 3, 4)
    h = 0.1
    x_min, x_max = df['feature_1'].min() - 1, df['feature_1'].max() + 1
    y_min, y_max = df['feature_2'].min() - 1, df['feature_2'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    model.eval()
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    
    ax4.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    ax4.scatter(class_0['feature_1'], class_0['feature_2'], 
                alpha=0.7, label='Class 0', s=20, edgecolor='k')
    ax4.scatter(class_1['feature_1'], class_1['feature_2'], 
                alpha=0.7, label='Class 1', s=20, edgecolor='k')
    ax4.set_title('Decision Boundary')
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Feature 2')
    ax4.legend()
    
    # Plot 5: Feature correlation
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(df['feature_1'], df['feature_2'], c=df['label'], 
                         cmap='RdYlBu', alpha=0.6, s=30, edgecolor='k')
    ax5.set_title('Feature Correlation')
    ax5.set_xlabel('Feature 1')
    ax5.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax5, label='Label')
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    Model Performance Summary
    ========================
    Final Train Loss: {train_losses[-1]:.4f}
    Final Test Loss: {test_losses[-1]:.4f}
    Final Accuracy: {accuracies[-1]:.4f}
    
    Dataset Statistics
    ==================
    Total Samples: {len(df)}
    Train Samples: {len(X_train)}
    Test Samples: {len(X_test)}
    Features: {df.shape[1] - 1}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, 
             family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('combined_analysis_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'combined_analysis_results.png'")
    plt.show()

def main():
    """Main function demonstrating all three libraries"""
    print("=" * 60)
    print("Combined Example: PyTorch, Pandas, and Matplotlib")
    print("=" * 60)
    
    # Step 1: Create dataset with Pandas
    df = create_dataset()
    
    # Step 2: Prepare data for PyTorch
    print("\nPreparing data for PyTorch...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 3: Create and train PyTorch model
    model = ClassificationNN(input_size=2, hidden_size=16)
    print(f"\nModel architecture:")
    print(model)
    
    train_losses, test_losses, accuracies = train_model(
        model, X_train, y_train, X_test, y_test, epochs=50
    )
    
    # Step 4: Visualize with Matplotlib
    print("\nCreating visualizations with Matplotlib...")
    visualize_results(df, model, X_train, X_test, y_test, 
                     train_losses, test_losses, accuracies)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Final accuracy: {accuracies[-1]:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    main()
