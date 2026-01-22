**Helper utilities**

Files under `library/Helper/`:
- `Initializers.py` — weight initialization utilities
- `Prunning.py` — pruning helpers (note: spelling in repo `Prunning`)
- `Quantize.py` — quantization helpers

Initializers
- Typical functions: `zeros_`, `ones_`, `xavier_uniform`, `kaiming_normal`, etc.
- Use to initialize `Parameter.data` before training.

Pruning
- Methods may implement magnitude-based pruning (zeroing small weights) or structured pruning.
- Pruning should preserve parameter shapes and be reversible via masks if needed.

Quantize
- Implementations may simulate fixed-point behavior or reduce bit precision for weights/activations.
- Useful for model size reduction and inference speed tests.

Notes:
- Check helper docstrings to understand exact argument names and defaults.
- `Initializers` should be used before training; quantize/prune can be applied post-training or during fine-tuning.
