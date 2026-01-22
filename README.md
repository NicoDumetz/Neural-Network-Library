# Documentation — G-CNA Neural Network Library

This `docs/` directory contains per-module reference pages and usage examples for the neural network library in `library/`.

Files:
- `modules.md` — Module base class and `Sequential`
- `tensor.md` — `Tensor` and `Parameter` data structures and autodiff summary
- `layers.md` — `Linear`, `Conv2d`, `MaxPool2d`, `BatchNorm2d`, `Flatten`
- `activations.md` — `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`
- `losses.md` — `MSE`, `L1`, `CrossEntropy`, `BCE`
- `optimizers.md` — `SGD`, `Adam`, `Scheduler/StepLr`
- `backward_ops.md` — Autodiff backward op overview
- `helper.md` — Initializers, Pruning, Quantize helpers
- `examples.md` — Minimal example code and training step

Next steps:
- I can scan source files and insert exact function/class signatures into these pages.
- I can produce an HTML site from these Markdown files if you want.
