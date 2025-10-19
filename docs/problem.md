# Problem

The problem is to predict the integrated fuel burn, $\int_{t_{\text{start}}}^{t_{\text{end}}} \dot{m_f}(t) \, dt$, over a specified time interval. The challenge is to learn a model of the aircraft's continuous-time dynamics from a sparse, irregularly-sampled observation history, $X = \{(x_i, t_i)\}_{i=1}^N$. The prediction target is defined over a sub-interval $[t_{\text{start}}, t_{\text{end}}]$ which only covers a fraction of the total observation window.

A successful model must address four key challenges:

1. Learn a continuous latent state trajectory $z(t)$ that accurately represents the aircraft's physical state from the discrete observation set $X$.
2. Robustly handle non-uniform time gaps $\Delta t_i$ between observations.
3. Capture long-range, non-linear dependencies inherent in flight data.
4. Directly and efficiently compute the target integral, where the instantaneous fuel burn rate is a function of the latent state, $\dot{m_f}(t) = g(z(t))$.

This problem structure allows for two inference paradigms. Given all observations $X_{1:N}$, estimating the state $z(t)$ for $t \in [t_{\text{start}}, t_{\text{end}}]$ can be framed as:

- **Filtering:** Using only past and current data, $p(z(t) | X_{t_i \le t})$. This is a causal approach.
- **Smoothing:** Using the entire observation history, $p(z(t) | X_{1:N})$. This is more powerful for offline analysis as it uses "future" data (relative to the segment) to correct state estimates, likely yielding a more accurate representation.

## `v0.0.1`

This serves as the baseline. For simplicity, we use A20N data only, and discard trajectory information not within $[t_{\text{start}}, t_{\text{end}}]$.

To handle the positive-only, long-tailed distribution of the average fuel burn rate, we predict its log-transformed value. The target variable is:

$$
y = \log\left(\frac{\text{fuel\_kg}}{t_{\text{end}} - t_{\text{start}}} + 1\right)
$$

This simplifies the problem to a sequence-to-scalar regression task, avoiding direct integration of the instantaneous burn rate $\dot{m}_f(t)$.

To explicitly handle the irregular sampling, each input vector $x_i \in \mathbb{R}^{D+2}$ is an augmentation of the raw observation $o_i$ with two temporal features:

1. Time since takeoff ($\tau_i = t_i - t_{\text{takeoff}}$) for global temporal context.
2. Log-transformed time since the previous observation. The time gaps $\Delta t_i = t_i - t_{i-1}$ exhibit a long-tail distribution, spanning from seconds to hours - we use log transformation to stabilise training. For the first observation in the segment ($i=1$), we set $\Delta t_1 = 0$.

The final input vector at step $i$ is the concatenation:

$$
x_i = [o_i, \tau_i, \log(\Delta t_i + 1)]
$$

where $o_i$ are observations.

We use a **single layer** Gated DeltaNet, a linear RNN that processes the input sequence $X$ to update its hidden state matrix $S_t \in \mathbb{R}^{d_v \times d_k}$. The state transition from $t-1$ to $t$ is governed by the gated delta rule, a Householder-like transformation:

$$
S_t = S_{t-1} \odot \left( \alpha_t \odot (I - \beta_t k_t k_t^\top) \right) + \beta_t v_t k_t^\top
$$

| Dataset                     | Target                        | RMSE (kg/s) | RMSE (kg) |
| --------------------------- | ----------------------------- | ----------- | --------- |
| a320n, `seq_len` ∈ (1, 256] | `avg_fuel_burn_rate`          | 0.296       | ?         |
| a320n                       | `avg_fuel_burn_rate`          | 0.205       | 103.98    |
| a320n                       | log(`avg_fuel_burn_rate` + 1) | 0.198       | 98.30     |

## TODO

- [ ] `fuel_burn` is quantised (see [data](./data.md)): randomly perturb output by the estimated uncertainty (?)
- [ ] anomalous data points (sudden jumps in altitude or speed)

third party sources of information

- [ ] google-arco era5 for isa deviation, GS -> TAS conversion, temperature, pressure
- [ ] kinematic features ($p, q, r$, $\dot{x}, \dot{y}, \dot{z}$)
- [ ] specific energy (potential + kinetic)
- [ ] <https://github.com/DGAC/Acropole>