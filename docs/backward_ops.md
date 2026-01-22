**Autodiff / Backward Operations**

Location: `library/Tensor/BackwardOP/`

Overview:
- Each forward operation registers a corresponding `BackwardOp` that knows how to compute gradients for its inputs given the upstream gradient.
- The backward graph is traversed in reverse topological order; `BackwardOp.backward(upstream_grad)` returns gradients for each input and accumulates them into `.grad` fields.

Common backward ops and formulas:
- `AddBackward`: upstream gradient is passed unchanged to both operands (or split if broadcasting was used).
- `MulBackward`: for elementwise multiply $z = x \odot y$:
  - $\partial_x = \partial_z \odot y$, $\partial_y = \partial_z \odot x$
- `MatMulBackward`: matrix rules discussed in `tensor.md` and `layers.md`.
- `Conv2dBackward`: computes `dK`, `dX`, and `db` by correlating upstream gradients with input patches.
- Unary ops (e.g., `ReLUBackward`, `SigmoidBackward`): compute elementwise derivative and multiply by upstream gradient.

Implementation notes:
- Broadcasting reductions: when an operation involved broadcasting, the backward pass must reduce summed gradients to match the original operand shapes.
- In-place ops: be careful with operations that modify tensors in-place; these can complicate gradient computation.
- Memory: the graph may hold references to intermediate tensors needed for backward; library may optionally support detaching or checkpointing to save memory.
