# Learning PyTorch, Pandas, and Matplotlib

A comprehensive learning project demonstrating the use of PyTorch, pandas, and matplotlib for data analysis and machine learning.

## Overview

This project contains three example scripts that demonstrate:
1. **Data Analysis** with pandas and matplotlib
2. **Neural Networks** with PyTorch
3. **Combined ML Workflow** using all three libraries together

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone this repository (or download the files)

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── requirements.txt              # Project dependencies
├── PROJECT_README.md            # This file
├── pandas_matplotlib_demo.py    # Data analysis demo
├── pytorch_demo.py              # Neural network demo
└── combined_demo.py             # Combined example
```

## Examples

### 1. Pandas and Matplotlib Demo (`pandas_matplotlib_demo.py`)

This script demonstrates:
- Creating and manipulating data with pandas DataFrames
- Performing basic statistical analysis
- Creating multiple types of visualizations with matplotlib

**Run it:**
```bash
python pandas_matplotlib_demo.py
```

**What it does:**
- Generates synthetic sales data
- Analyzes profit trends
- Creates 4 different plots showing sales, expenses, profit, and distributions
- Saves the visualization as `data_analysis_results.png`

### 2. PyTorch Demo (`pytorch_demo.py`)

This script demonstrates:
- Building a simple neural network with PyTorch
- Training the network on synthetic data
- Visualizing training progress and predictions

**Run it:**
```bash
python pytorch_demo.py
```

**What it does:**
- Creates a simple feedforward neural network
- Trains it on a linear regression task
- Shows loss curves and prediction accuracy
- Saves the visualization as `pytorch_training_results.png`

### 3. Combined Demo (`combined_demo.py`)

This script demonstrates:
- Complete ML workflow using all three libraries
- Data preparation with pandas
- Model training with PyTorch
- Comprehensive visualization with matplotlib

**Run it:**
```bash
python combined_demo.py
```

**What it does:**
- Creates a binary classification dataset with pandas
- Builds and trains a classification neural network with PyTorch
- Generates 6 different visualizations including decision boundaries
- Saves the visualization as `combined_analysis_results.png`

## Learning Objectives

### Pandas
- Creating DataFrames
- Data manipulation and transformation
- Statistical analysis
- Data filtering and selection

### Matplotlib
- Creating various plot types (line, scatter, histogram)
- Customizing plots (colors, labels, legends)
- Creating subplots and figures
- Saving visualizations

### PyTorch
- Building neural network architectures
- Defining forward passes
- Training loops with backpropagation
- Using optimizers and loss functions
- Model evaluation

## Expected Output

Each script will:
1. Print progress and results to the console
2. Generate and save a PNG visualization
3. Display interactive plots (if running in an interactive environment)

## Tips for Learning

1. **Start Simple**: Run each script individually to understand what each library does
2. **Experiment**: Modify parameters like learning rates, network sizes, or data ranges
3. **Read the Code**: Each script is well-commented to explain what's happening
4. **Combine Concepts**: The combined demo shows how these libraries work together

## Common Issues

**Issue**: Import errors
- **Solution**: Make sure you've installed all requirements: `pip install -r requirements.txt`

**Issue**: No display for plots
- **Solution**: The plots are saved as PNG files, you can view them even without a display

**Issue**: CUDA/GPU errors
- **Solution**: The scripts work fine on CPU. PyTorch will automatically use CPU if GPU is not available

## Next Steps

After running these examples, try:
- Modifying the network architectures
- Using different datasets
- Adding more visualization types
- Implementing different ML tasks (regression, multi-class classification)
- Exploring more advanced PyTorch features (CNNs, RNNs)

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## License

This project is provided as-is for educational purposes.
