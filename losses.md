**Losses**

Loss implementations live under `library/Losses/`.

MSELoss (Mean Squared Error)
- File: `library/Losses/MSELoss/MSELoss.py`
- Loss: $\mathcal{L} = \frac{1}{N} \sum_i (\hat{y}_i - y_i)^2$
- Gradient: $\frac{2}{N} (\hat{y} - y)$

L1Loss (Mean Absolute Error)
- File: `library/Losses/L1Loss/L1Loss.py`
- Loss: $\mathcal{L} = \frac{1}{N} \sum_i |\hat{y}_i - y_i|$
- Subgradient: $\text{sign}(\hat{y} - y)/N$

CrossEntropyLoss
- File: `library/Losses/CrossEntropyLoss/CrossEntropyLoss.py`
- For logits + targets (class indices), using softmax + negative log-likelihood.
- Gradient (softmax + CE combined): $s_i - 1_{i=t}$ (per-sample, average over batch)

BCELoss (Binary Cross Entropy)
- File: `library/Losses/BCELoss/BCELoss.py`
- Loss: $- [y \log \hat{y} + (1-y) \log (1-\hat{y})]$
- Gradient: $-y/\hat{y} + (1-y)/(1-\hat{y})$ (implementations often operate on logits for stability)

Notes:
- For numerical stability, prefer combined implementations (e.g., `log_softmax` + `nll_loss`) or log-sum-exp tricks.
- Loss classes typically return a scalar `Tensor` to make `loss.backward()` straightforward.
