"""
Simple Data Analysis with Pandas and Matplotlib
This script demonstrates basic data manipulation and visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_sample_data():
    """Create a sample dataset for analysis"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    data = {
        'date': dates,
        'sales': np.random.randint(100, 500, size=100),
        'expenses': np.random.randint(50, 300, size=100),
        'profit': None
    }
    df = pd.DataFrame(data)
    df['profit'] = df['sales'] - df['expenses']
    return df

def analyze_data(df):
    """Perform basic data analysis"""
    print("=" * 50)
    print("Data Analysis Summary")
    print("=" * 50)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())
    print(f"\nTotal profit: ${df['profit'].sum():.2f}")
    print(f"Average daily profit: ${df['profit'].mean():.2f}")
    print(f"Best day profit: ${df['profit'].max():.2f}")
    print(f"Worst day profit: ${df['profit'].min():.2f}")

def visualize_data(df):
    """Create visualizations of the data"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Sales over time
    axes[0, 0].plot(df['date'], df['sales'], color='blue', label='Sales')
    axes[0, 0].set_title('Sales Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Sales ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Expenses over time
    axes[0, 1].plot(df['date'], df['expenses'], color='red', label='Expenses')
    axes[0, 1].set_title('Expenses Over Time')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Expenses ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Profit over time
    axes[1, 0].plot(df['date'], df['profit'], color='green', label='Profit')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Profit Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Profit ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Histogram of profit distribution
    axes[1, 1].hist(df['profit'], bins=20, color='purple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Profit Distribution')
    axes[1, 1].set_xlabel('Profit ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_analysis_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'data_analysis_results.png'")
    plt.show()

def main():
    """Main function to run the analysis"""
    print("Starting Data Analysis with Pandas and Matplotlib...\n")
    
    # Create sample data
    df = create_sample_data()
    
    # Analyze the data
    analyze_data(df)
    
    # Visualize the data
    visualize_data(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
