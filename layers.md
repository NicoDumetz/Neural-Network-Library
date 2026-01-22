**Layers**

This page summarizes the most important layers found in `library/Layers/`.

Linear
- File: `library/Layers/Linear/Linear.py`
- Math (forward): given input $X\in\mathbb{R}^{N\times D_{in}}$, weights $W\in\mathbb{R}^{D_{out}\times D_{in}}$, bias $b\in\mathbb{R}^{D_{out}}$:
  $$Y = X W^T + b$$
- Backprop:
  - $dW = \delta^T X$ (adjust depending on implementation layout)
  - $db = \sum_i \delta_{i,:}$
  - $dX = \delta W$

Conv2d
- File: `library/Layers/Conv2d/Conv2d.py`
- Inputs: $X\in\mathbb{R}^{N\times C_{in}\times H_{in}\times W_{in}}$
- Kernels: $K\in\mathbb{R}^{C_{out}\times C_{in}\times k_H\times k_W}$
- Stride and padding affect output shape. Implementation may use nested loops or `im2col`-style lowering.
- Backprop:
  - $dK$ accumulates upstream gradients times input patches across batch and spatial dims.
  - $dX$ computed via convolution of upstream gradient with flipped kernels.

MaxPool2d
- File: `library/Layers/MaxPool2d/MaxPool2d.py`
- Forward: partitions input into windows and outputs the max per window.
- Backward: routes gradient to the index of the max value in each pooling window.

BatchNorm2d
- File: `library/Layers/BatchNorm2d/BatchNorm2d.py`
- Per-channel normalization with running mean/variance when in `eval()` mode.
- Forward math uses batch mean/variance; backward must propagate through normalization steps.

Flatten
- File: `library/Layers/Flatten/Flatten.py`
- Reshape helper switching between convolutional and linear shapes.
- Backward reshapes gradients back to original shape.

Notes:
- Each layer registers `Parameter`s for trainable weights and biases so optimizers can update them.
- Pay attention to the weight layout in `Linear` and `Conv2d` when interpreting gradient formulas.
