# G-CNA Neural Network Library

This repository hosts the documentation and sources for the G-CNA library, a collection of operators, layers, and optimizers meant to explore neural network ideas from scratch. The goal is to offer simple, understandable building blocks that are easy to extend without relying on a heavy external stack.

## Repository structure

- `docs/`: reference pages grouped by topic (modules, tensors, layers, etc.). Browse these files for detailed explanations.
- `Module/`, `Optimizer/`, `Tensor/`, `Configuration/`, `Helper/`: implementation code organized by responsibility, matching the related documentation.
- `README.md`: this onboarding file.

## Documentation

- [General guide](docs/DOCUMENTATION.md): overview of the library concepts, conventions, and workflows.
- [Modules](docs/modules.md): the `Module` base class, `Sequential`, and the forward-passing steps.
- [Tensors and parameters](docs/tensor.md): `Tensor`, `Parameter`, and a quick autodiff refresher.
- [Layers](docs/layers.md): `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten`, and usage notes.
- [Activations](docs/activations.md): activation functions (`ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`).
- [Losses](docs/losses.md): implementations of `MSE`, `L1`, `CrossEntropy`, `BCE`.
- [Optimizers](docs/optimizers.md): `SGD`, `Adam`, and the `StepLr` scheduler.
- [Backward operations](docs/backward_ops.md): overview of operations created during backprop.
- [Helpers](docs/helper.md): initializers, pruning techniques, and quantization helpers.
- [Analysis](docs/analyzer.md): tools for inspecting and debugging execution flows.

## Getting started

1. Read the files in `docs/` to understand the available primitives.
2. Dive into the `Module/`, `Tensor/`, etc. directories to see the actual implementations.
3. Add your own modules or extensions within these folders and update the relevant `docs/` page when needed.
