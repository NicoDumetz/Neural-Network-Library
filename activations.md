**Activations**

Activation functions and their forward / derivative formulas. Files are under `library/Activations/`.

ReLU
- File: `library/Activations/ReLU/ReLU.py`
- Forward: $y = \max(0, x)$
- Derivative: $\frac{dy}{dx} = \mathbb{1}_{x>0}$

LeakyReLU
- File: `library/Activations/LeakyReLU/LeakyReLU.py`
- Forward: $y = \max(\alpha x, x)$ with small $\alpha$ (e.g., 0.01)
- Derivative: $1$ if $x>0$, $\alpha$ otherwise

Sigmoid
- File: `library/Activations/Sigmoid/Sigmoid.py`
- Forward: $\sigma(x) = 1/(1+e^{-x})$
- Derivative: $\sigma(x)(1-\sigma(x))$

Tanh
- File: `library/Activations/Tanh/Tanh.py`
- Forward: $\tanh(x)$
- Derivative: $1-\tanh^2(x)$

Softmax
- File: `library/Activations/Softmax/Softmax.py`
- Forward: $\text{softmax}(z)_i = e^{z_i}/\sum_j e^{z_j}$ (apply per-sample along class dim)
- Jacobian: $J_{ij} = s_i(\delta_{ij} - s_j)$

Implementation notes:
- `Softmax` is typically combined with `CrossEntropyLoss` to compute numerically stable gradients.
- Activation backward ops should handle elementwise masking and preserve input shapes.
