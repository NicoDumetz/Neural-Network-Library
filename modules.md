**Module system**

- File: `library/Module/Module.py`

Overview:
- `Module` is the base class for layers and containers. It holds child modules and parameters, provides `forward()` (to implement), `parameters()` iterator, and `train()`/`eval()` mode switches.
- `Sequential` is a convenience container that composes modules in order and implements `forward(x)` by piping input through children.

Key responsibilities and behavior:
- Parameter registration: when a `Parameter` is assigned as an attribute of a `Module`, it should be discoverable by `parameters()`.
- Registering child `Module` objects: attributes that are `Module` instances should be tracked as children.
- Mode toggles: `train()` sets training-specific behavior (e.g., dropout active, batchnorm uses batch stats). `eval()` disables those behaviors.

Usage example:
```python
from library.Module.Module import Module
from library.Sequential.Sequential import Sequential

class MyNet(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential([...])

    def forward(self, x):
        return self.layers.forward(x)
```

Notes:
- The `forward()` method is expected to perform forward computation and return a `Tensor`.
- Backpropagation is handled by the `Tensor` and BackwardOp classes; typically `loss.backward()` will trigger the graph traversal.
