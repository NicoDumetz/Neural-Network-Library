**Optimizers**

This page documents optimizer behavior and expected fields.

SGD
- File: `library/Optimizer/SGD/SGD.py`
- Update rule: $\theta \leftarrow \theta - \eta g$
- Momentum variant (if implemented):
  $$v_t = \mu v_{t-1} + g_t; \quad \theta_t = \theta_{t-1} - \eta v_t$$

Adam
- File: `library/Optimizer/Adam/Adam.py`
- Equations:
  $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
  $$\hat{m}_t = m_t/(1-\beta_1^t), \ \hat{v}_t = v_t/(1-\beta_2^t)$$
  $$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}$$

Scheduler / StepLr
- File: `library/Optimizer/Scheduler/StepLr/StepLr.py`
- Reduces lr by factor `gamma` every `step_size` epochs: $\eta_t = \eta_0 \gamma^{\lfloor t/step\_size \rfloor}$

API expectations:
- Optimizers should accept an iterable of parameters (from `model.parameters()`), and maintain per-parameter state (e.g., momentum buffers, Adam's `m` and `v`).
- `step()` applies the update; `zero_grad()` clears accumulated gradients.

Notes on use:
- Always call `zero_grad()` between updates unless intentionally accumulating gradients.
- Use scheduler `step()` per epoch (or per batch depending on scheduler semantics).
