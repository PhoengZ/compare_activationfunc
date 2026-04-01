# Neural Network Activation Function Comparison - Perceptron Behavior

## Project Overview

This project is an experimental study that compares the behavior and performance of different **activation functions** in a neural network learning task. The experiment trains a 2-layer neural network to approximate a **cosine function** and visualizes how different activation functions affect the learning process and final predictions.

## Experiment Description

### Objective
To analyze and compare how different activation functions impact neural network learning by training networks with identical architectures but different activation mechanisms on the same regression task.

### Task
- **Function to Learn**: Cosine function: $f(x) = \cos(x)$
- **Input Range**: $x \in [-10, 10]$
- **Data Points**: 300 evenly distributed samples

### Network Architecture
- **Input Layer**: 1 neuron
- **Hidden Layer**: 200 neurons (configurable)
- **Output Layer**: 1 neuron
- **Learning Method**: Backpropagation with gradient descent

### Activation Functions Tested

1. **ReLU (Rectified Linear Unit)**
   - Formula: $f(x) = \max(0, x)$
   - Derivative: 1 if x > 0, else 0
   - Characteristics: Fast, sparse activation, prone to dead neurons

2. **Sigmoid**
   - Formula: $f(x) = \frac{1}{1 + e^{-x}}$
   - Derivative: $f'(x) = f(x)(1 - f(x))$
   - Characteristics: S-shaped curve, smooth, outputs in range [0, 1]

3. **Tanh (Hyperbolic Tangent)**
   - Formula: $f(x) = \tanh(x)$
   - Derivative: $f'(x) = 1 - f(x)^2$
   - Characteristics: Smooth, outputs centered around 0 (range [-1, 1])

4. **Linear**
   - Formula: $f(x) = x$
   - Derivative: 1 (constant)
   - Characteristics: No transformation, baseline for comparison

5. **Clipper-Linear**
   - Formula: $f(x) = \text{clip}(x, -1, 1)$
   - Derivative: 1 if $-1 < x < 1$, else 0
   - Characteristics: Bounded linear activation

### Training Configuration
- **Learning Rate**: $1 \times 10^{-5}$
- **Number of Epochs**: 1,000
- **Training Method**: Backpropagation with weight and bias updates
- **Loss**: Mean Squared Error (implicit from gradient descent)

### Network Training Process

For each activation function, the network undergoes the following process:

1. **Forward Pass**: 
   - Hidden layer: $h = f_{activation}(x \cdot W_1 + b_1)$
   - Output layer: $y_{pred} = h \cdot W_2 + b_2$

2. **Backward Pass**:
   - Compute output error: $\delta_k = y_{pred} - y_{target}$
   - Compute hidden error: $\delta_h = (\delta_k \cdot W_2^T) \odot f'(h)$
   - Update weights and biases using gradient descent

3. **Results Storage**: Predictions are stored at every epoch to enable visualization

## Files in This Project

- **Perceptron_Behavior.ipynb**: Main Jupyter notebook containing the complete experiment code and visualizations
- **requirements.txt**: Python dependencies
- **README.md**: This file

## Key Findings

The experiment generates **animated visualizations** for each activation function showing:
- The target cosine curve (red dashed line)
- The neural network's learned approximation (blue line)
- Real-time epoch progression with training updates
- Performance difference at various stages of training

### Visualization Details
- Animation updates every 5 epochs for clarity
- Each frame displays the current epoch number and activation function name
- Horizontal range: [-10, 10]
- Vertical range: [-1.5, 1.5]

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook Perceptron_Behavior.ipynb
   ```

3. Execute all cells to:
   - Visualize individual activation functions
   - Train networks with each activation function
   - Generate animated comparisons

## Dependencies

- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Plotting and visualization
- **IPython**: HTML5 video generation for animations
- **Jupyter**: Interactive notebook environment

## Expected Observations

- **ReLU**: Fast learning but may suffer from dead neurons
- **Sigmoid**: Smooth learning with potential vanishing gradient issues
- **Tanh**: Similar to Sigmoid but better centered
- **Linear**: Limited expressive power, struggles with non-linear patterns
- **Clipper-Linear**: Bounded activation with limited gradient flow

## Research Implications

This experiment demonstrates:
1. The critical role of activation functions in neural network learning
2. How different activation functions affect convergence speed
3. The trade-offs between computational efficiency and learning capacity
4. Why modern architectures favor ReLU-based activations

## Author Notes

This is an educational project designed to provide intuitive understanding of how activation functions influence neural network behavior through visual comparison and empirical observation.

---

**Last Updated**: April 2026
