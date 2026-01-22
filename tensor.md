**Tensor & Parameter**

- File: `library/Tensor/Tensor.py`
- File: `library/Parameter/Parameter.py`

Core concepts:
- `Tensor` stores numeric data, gradient, shape, `requires_grad` flag, and references to the operation (`grad_fn`) that created it.
- `Tensor.backward()` triggers the backward pass when called on a scalar loss (or with `gradient` passed if not scalar).
- `Parameter` is a thin wrapper indicating the tensor is a model parameter and should be updated by optimizers.

Typical fields and semantics:
- `tensor.data`: the raw numeric array (NumPy array or similar)
- `tensor.grad`: same-shaped array holding gradient values
- `tensor.requires_grad`: boolean
- `tensor.grad_fn` / `tensor.creator`: reference to the backward op or function that will compute gradients

Autodiff overview:
- Forward ops create new `Tensor`s and attach corresponding `BackwardOp` objects.
- During `backward()`, the engine traverses the graph in reverse-topological order, calling `BackwardOp.backward(upstream_grad)` to compute and accumulate gradients on inputs.
- Broadcasting and reduce-sum handling must be considered by each backward op.

Best practices:
- Use `Parameter` for weights and biases so `Module.parameters()` yields trainable tensors.
- Call `optimizer.zero_grad()` (or set `.grad` to zero) before or after each optimizer update to avoid unwanted gradient accumulation.
