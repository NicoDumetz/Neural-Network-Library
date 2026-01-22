# G-CNA Neural Network Library — Detailed Documentation

This document describes the structure, API concepts, and mathematics behind the library contained in the `library/` folder of this project. It is written to be implementation-agnostic but maps to the files in the repository (see "File map" near the end).

The document uses the following notation:
- Scalars: lowercase letters (e.g., $a$)
- Vectors: bold lowercase (e.g., $\mathbf{x}$)
- Matrices: bold uppercase (e.g., $\mathbf{W}$)
- Tensors: indexed objects (e.g., $X \in \mathbb{R}^{N\times C\times H\times W}$)
- $\odot$ denotes element-wise (Hadamard) product.

All formulas use standard deep-learning math notation. Inline math is written with $...$ and displayed equations with $$...$$.

**Conventions and shapes**
- Batch-first convention is used where applicable: an input mini-batch $X$ is often shaped $X \in \mathbb{R}^{N \times D}$ (dense) or $\mathbb{R}^{N \times C \times H \times W}$ (conv).
- For a `Linear` layer with input features $D_{in}$ and output features $D_{out}$:
  - Weight matrix $W$ shape: $D_{out} \times D_{in}$ (common layout: rows=output)
  - Bias $b$ shape: $D_{out}$

**High-level workflow (training loop)**
1. Forward pass: compute predictions by chaining `Module` components.
2. Compute loss $\mathcal{L}(y,\hat{y})$.
3. Backward pass: compute gradients via autodiff / backward ops.
4. Optimizer `step()` updates parameters (e.g., SGD, Adam).
5. Zero gradients for the next batch.

**Core building blocks**

**`Tensor` and `Parameter`**
- `Tensor` represents an n-dimensional array with:
  - `data`: numeric values
  - `grad`: gradient w.r.t. scalar loss (same shape)
  - `requires_grad`: boolean
  - `creator` / `grad_fn`: reference to operation that produced it for backward traversal

- `Parameter` is a thin wrapper for tensors that are module parameters (weights/biases).

Mathematical role: A tensor is the holder of numeric data $X$ and gradient $\frac{\partial \mathcal{L}}{\partial X}$.

**Autodiff primitives (BackwardOps)**
The library organizes backward ops into binary and unary backward classes. For each forward op $y=f(x, ...)$ there is a backward op implementing how gradients propagate.

Common gradients (forward op -> backward rule):
- Addition: $z = x + y$
  - $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z}$
  - $\frac{\partial \mathcal{L}}{\partial y} = \frac{\partial \mathcal{L}}{\partial z}$
- Multiplication (elementwise): $z = x \odot y$
  - $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z} \odot y$
  - $\frac{\partial \mathcal{L}}{\partial y} = \frac{\partial \mathcal{L}}{\partial z} \odot x$
- Matrix multiply (2D): $Z = XW$ (shapes: $X\in\mathbb{R}^{N\times D}, W\in\mathbb{R}^{D\times M}$)
  - $\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Z} W^T$  
  - $\frac{\partial \mathcal{L}}{\partial W} = X^T \frac{\partial \mathcal{L}}{\partial Z}$
- Elementwise divide: $z = x / y$
  - Use quotient rule piecewise: $\partial_x = \partial_z * (1 / y)$, $\partial_y = -\partial_z * (x / y^2)$

For each backward op, implementations typically handle broadcasting, shape inference, and reduce-sum to match shapes.

**Module system**
- `Module` is the base class for all layers and containers. Key responsibilities:
  - Hold child modules and parameters
  - Implement `forward()` method
  - Implement `parameters()` iterator
  - `train()` / `eval()` mode toggles

- `Sequential` composes modules in order and implements `forward(input)` by piping through children.

Mathematical contract: Each `Module` defines a function $f_{\theta}(x)$ where $\theta$ are its parameters. The autodiff engine records ops to compute $\nabla_{\theta}\mathcal{L}$.

**Layers**

**`Linear` (Fully connected)**
- Forward: given input $X \in \mathbb{R}^{N\times D_{in}}$, weights $W \in \mathbb{R}^{D_{out}\times D_{in}}$, bias $b \in \mathbb{R}^{D_{out}}$:
  $$Y = X W^T + b$$
  (some codebases use $Y = X W^T$ or $Y = X W + b$, check shapes in `library/Layers/Linear/Linear.py`)

- Backprop:
  - Given $\delta = \frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{N\times D_{out}}$,
  - $\frac{\partial \mathcal{L}}{\partial W} = \delta^T X$ (or $X^T \delta$ depending on next dimension conventions). Using $Y = X W^T$ form:
    $$\frac{\partial \mathcal{L}}{\partial W} = \delta^T X$$
  - $\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^N \delta_{i,:}$
  - $\frac{\partial \mathcal{L}}{\partial X} = \delta W$ (if $W$ layout matches previous equations).

**`Conv2d`**
- Convolution is usually implemented as cross-correlation (no kernel flip in forward). For an input $X \in \mathbb{R}^{N\times C_{in}\times H_{in}\times W_{in}}$, kernels $K \in \mathbb{R}^{C_{out}\times C_{in}\times k_H \times k_W}$, stride $s$, padding $p$:

  - Output height: $H_{out} = \left\lfloor \frac{H_{in} + 2p - k_H}{s} \right\rfloor + 1$
  - Output width analogous.

- Forward (element expression):
  $$Y_{n,co,h,w} = \sum_{ci=0}^{C_{in}-1} \sum_{kh=0}^{k_H-1} \sum_{kw=0}^{k_W-1} X_{n,ci, h*s+kh - p, w*s+kw - p} \cdot K_{co,ci,kh,kw} + b_{co}$$

- Backprop:
  - Gradient w.r.t. kernels:
    $$\frac{\partial \mathcal{L}}{\partial K_{co,ci,kh,kw}} = \sum_{n,h,w} \frac{\partial \mathcal{L}}{\partial Y_{n,co,h,w}} \cdot X_{n,ci,h*s+kh-p,w*s+kw-p}$$
  - Gradient w.r.t. input is convolution of output-grad with rotated kernels (flip spatial dims):
    $$\frac{\partial \mathcal{L}}{\partial X_{n,ci,:,:}} = \sum_{co} \text{full\\_conv}(\frac{\partial \mathcal{L}}{\partial Y_{n,co,:,:}}, K_{co,ci, : , :}^{rot})$$
  - Implementation note: many libraries compute these with im2col or nested loops, ensuring correct stride/pad.

**`MaxPool2d`**
- Forward: partitions input into windows and selects the maximum.
- Backward: routes gradient to the index of the maximum inside each pooling window. If ties exist, implementations may select the first index encountered.

**`BatchNorm2d`**
- Given mini-batch per-channel mean and variance:
  $$\mu = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu)^2$$
  $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
  $$y_i = \gamma \hat{x}_i + \beta$$
- Backprop is the multi-step chain rule through mean/variance normalization; a compact gradient formula for $\frac{\partial \mathcal{L}}{\partial x}$:
  (see many references for the full rearranged expression)

**`Flatten`**
- Forward: reshapes the tensor, typically from $\mathbb{R}^{N\times C\times H\times W}$ to $\mathbb{R}^{N\times (CHW)}$.
- Backward: reshapes the gradient back to original shape.

**Activations**

**ReLU**
- Forward: $y = \max(0, x)$
- Derivative: $y' = \mathbb{1}_{x > 0}$

**LeakyReLU**
- Forward: $y = \max(\alpha x, x)$ with small $\alpha$ (e.g., 0.01)
- Derivative: $y' = 1$ if $x>0$, $\alpha$ otherwise

**Sigmoid**
- Forward: $\sigma(x) = \frac{1}{1+e^{-x}}$
- Derivative: $\sigma'(x) = \sigma(x) (1 - \sigma(x))$

**Tanh**
- Forward: $\tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$
- Derivative: $1 - \tanh^2(x)$

**Softmax**
- For vector $z \in \mathbb{R}^C$:
  $$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$

- Jacobian: $J_{ij} = s_i (\delta_{ij} - s_j)$ where $s = \text{softmax}(z)$.
- Commonly combined with cross-entropy loss to produce a numerically stable gradient expression.

**Losses**

**MSE (Mean Squared Error)**
- $$\mathcal{L}(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$
- Gradient: $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \frac{2}{N} (\hat{y} - y)$$

**L1 Loss (Mean Absolute Error)**
- $$\mathcal{L} = \frac{1}{N} \sum_i |\hat{y}_i - y_i|$$
- Subgradient: $$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{N} \text{sign}(\hat{y}_i - y_i)$$

**Cross-Entropy Loss (multi-class)**
- If using softmax + CE and target is one-hot or index, for prediction probabilities $p_i$ and true class $t$:
  $$\mathcal{L} = -\log p_t$$
- Gradient (softmax + CE combined): if $z$ are logits and $s$ = softmax(z):
  $$\frac{\partial \mathcal{L}}{\partial z_i} = s_i - 1_{i=t}$$
  For a batch average divide by $N$.

**Binary Cross-Entropy (BCE)**
- For predictions $\hat{y} \in (0,1)$ and label $y \in \{0,1\}$:
  $$\mathcal{L} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$$
- Gradient: $$\frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$$
- More stable implementations operate on logits directly.

**Regularization**

**Dropout**
- Forward (train): multiply input by mask $m$ where $m_i \sim \text{Bernoulli}(p)$, then scale by $1/p$ if doing inverted dropout. So $y_i = x_i \cdot m_i / p$.
- Backward: gradient is multiplied by same mask and scaled.

**Optimizers**

**SGD**
- Basic update for parameter $\theta$ with learning rate $\eta$:
  $$\theta \,\leftarrow\, \theta - \eta \cdot g$$
  where $g$ is $\nabla_{\theta} \mathcal{L}$.

- Momentum variant (if implemented):
  $$v_{t} = \mu v_{t-1} + g_t$$
  $$\theta_{t} = \theta_{t-1} - \eta v_t$$

**Adam**
- Per-parameter running averages:
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
  Bias-corrected:
  $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
  Update:
  $$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Scheduler (StepLR)**
- Reduce learning rate by factor $\gamma$ every $k$ steps/epochs. If initial lr $\eta_0$ and `step_size` = $k$ then at epoch $t$:
  $$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

**Optimizer implementation notes**
- `step()` should iterate parameters and apply updates in-place.
- `zero_grad()` sets all parameter `.grad` to zero.
- Make sure gradients are accumulated across backward unless explicitly zeroed.

**Backward operation implementations (selected)**
These live under `Tensor/BackwardOP/` and the nested directories. Typical operations implemented include:
- `AddBackward`: simply routes upstream gradient equally.
- `MulBackward`: multiplies upstream gradient by the other operand.
- `MatMulBackward`: uses matrix rules shown earlier.
- `Conv2dBackward`: implements weight and input gradients using cross-correlation and full convolution.
- Unary ops like `ReLUBackward`, `SigmoidBackward`, `TanhBackward`, `ExpBackward`, `LogBackward` implement elementwise chain rule.

**Numerical stability tips used in the library**
- Softmax + cross entropy combined to avoid computing log of small numbers.
- Add small epsilon in denominators: $\sqrt{\sigma^2 + \epsilon}$ in batchnorm.
- Clip gradients if gradient clipping utility is present (not always implemented by default).

**Practical API patterns & examples**
Below are canonical usage patterns you can adapt to this codebase:

- Create a simple model (functional sketch):
```python
from library.Module.Module import Module
from library.Layers.Linear.Linear import Linear
from library.Sequential.Sequential import Sequential
from library.Activations.ReLU.ReLU import ReLU

model = Sequential([
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
])
```

- Training step sketch:
```python
# forward
pred = model.forward(x)
loss = loss_fn(pred, targets)
# backward
loss.backward()
# update
optimizer.step()
optimizer.zero_grad()
```

- Linear forward/backward minimal math (summary):
  - forward: $Y = X W^T + b$
  - grads: $dW = \delta^T X$, $db=\sum \delta$, $dX = \delta W$

**File map (matches repo layout)**
- `library/Module/Module.py`: base `Module` class, parameter registration
- `library/Tensor/Tensor.py`: `Tensor` core data structure and creation API
- `library/Parameter/Parameter.py`: parameter wrapper
- `library/Sequential/Sequential.py`: `Sequential` container
- `library/Layers/Linear/Linear.py`: fully connected layer
- `library/Layers/Conv2d/Conv2d.py`: 2D convolution layer
- `library/Layers/MaxPool2d/MaxPool2d.py`: max pooling
- `library/Layers/BatchNorm2d/BatchNorm2d.py`: batch normalization
- `library/Layers/Flatten/Flatten.py`: reshape helper
- `library/Activations/*`: `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`
- `library/Losses/*`: `MSELoss`, `L1Loss`, `CrossEntropyLoss`, `BCELoss`
- `library/Optimizer/SGD/SGD.py`: SGD optimizer
- `library/Optimizer/Adam/Adam.py`: Adam optimizer
- `library/Optimizer/Scheduler/StepLr/StepLr.py`: LR scheduler
- `library/Helper/Initializers.py`: weight initialization helpers
- `library/Helper/Prunning.py` and `Quantize.py`: pruning and quantization utilities
- `library/Tensor/BackwardOP/`: many backward op classes and specializations

**Appendix: Selected derivations**

1) Matrix multiply derivative (compact derivation)
Given $Z = XW$ with shapes $X(N\times D)$, $W(D\times M)$, $Z(N\times M)$ and scalar loss $\mathcal{L}$ where $\delta = \partial \mathcal{L} / \partial Z$:
- For $X$: treat each element $X_{ij}$ affects row $i$ of $Z$:
  $$\frac{\partial \mathcal{L}}{\partial X_{i,:}} = \delta_{i,:} W^T$$
  thus $\partial_X = \delta W^T$ (or $\delta W^T$ depending on layout; check whether forward uses $XW$ or $XW^T$).
- For $W$: each column j of $W$ multiplies all rows of $X$ to produce column j of $Z$:
  $$\frac{\partial \mathcal{L}}{\partial W} = X^T \delta$$

2) Convolution weight gradient (intuition)
- For each output position, the gradient w.r.t. kernel slices is the upstream gradient at that position multiplied by the corresponding input patch. Summation across batch and spatial positions yields the full kernel gradient.

**Notes about matching the repo's exact signatures**
This doc intentionally focuses on the mathematical semantics. The repository organizes code into many small files and classes (see File map). If you want, I can next:
- Open each implementation file and produce a per-file API reference (function/class signatures, args, defaults).
- Run a small script that prints actual signatures and docstrings derived from the code.

**Next steps (suggested)**
- Verify specifics: I can scan the actual `.py` files and map exact function signatures and argument order into the doc.
- Generate an HTML or Markdown API reference with auto-extracted docstrings.

---

If you'd like, I can now (pick one):
- Scan the codebase and produce a per-file reference (exact signatures + short examples).
- Add autodoc-style comments into the source files.
- Generate a `docs/` directory with separate pages for each module.

Tell me which next step you prefer and I will proceed.
