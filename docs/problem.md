# Problem

The problem is to predict the integrated fuel burn, $\int_{t_{\text{start}}}^{t_{\text{end}}} \dot{m_f}(t) \, dt$, over a specified time interval. The challenge is to learn a model of the aircraft's continuous-time dynamics from a sparse, irregularly-sampled observation history, $X = \{(x_i, t_i)\}_{i=1}^N$. The prediction target is defined over a sub-interval $[t_{\text{start}}, t_{\text{end}}]$ which only covers a fraction of the total observation window.

A successful model must address four key challenges:

1. Learn a continuous latent state trajectory $z(t)$ that accurately represents the aircraft's physical state from the discrete observation set $X$.
2. Robustly handle non-uniform time gaps $\Delta t_i$ between observations.
3. Capture long-range, non-linear dependencies inherent in flight data.
4. Directly and efficiently compute the target integral, where the instantaneous fuel burn rate is a function of the latent state, $\dot{m_f}(t) = g(z(t))$.

This problem structure allows for two inference paradigms. Given all observations $X_{1:N}$, estimating the state $z(t)$ for $t \in [t_{\text{start}}, t_{\text{end}}]$ can be framed as:

- **Filtering:** Using only past and current data, $p(z(t) | X_{t_i \le t})$. This is a causal approach.
- **Smoothing:** Using the entire observation history, $p(z(t) | X_{1:N})$. This is more powerful for offline analysis as it uses "future" data (relative to the segment) to correct state estimates. But adopting this is less helpful in real-world use cases because we may want to do real-time predictions.

## `v0.0.1`

### Architecture

For this sequence-to-scalar regression task, we directly use a simple **single layer** [Gated DeltaNet](https://arxiv.org/pdf/2412.06464) (GDN) for a baselien. Why:

- standard transformers are powerful but suffer from $O(L^2)$ complexity. GDN is a linear RNN belonging to the family of state space models (SSM) that offers near-linear complexity for both training and inference while approaching the expressivity of transformers.
- LSTM struggle to capture long-range dependencies due to vanishing gradient. GDN is far more expressive:
  - Delta Rule: allows model to make targeted corrections to its internal state with each new observation (effectively, fast weight programmer learning key-value associations from the flight history).
  - Gating: controls how much of the past state is preserved at each step, allowing model to forget information.

### Inputs

For simplicity, we discard trajectory information not within $[t_{\text{start}}, t_{\text{end}}]$.

The input vector at step $i$ is the concatenation:

$$
x_i = [o_i, \tau_i, \text{Embedding}(\text{aircraft\_type})]
$$

where $o_i$ are standardised observations and $\tau_i = t_i - t_{\text{takeoff}}$ provides global temporal context.

To explicitly handle the irregular sampling, each input vector $x_i \in \mathbb{R}^{D+1}$ is an augmentation of the raw observation $o_i$ with the time since takeoff $\tau_i = t_i = t_{\text{takeoff}}$ for global temporal context.

The time gaps $\Delta t_i = t_i - t_{i-1}$ exhibit a long-tail distribution, spanning from seconds to hours. While including $\log(t_i + 1)$ might help the model, dropping it altogether seemed to improve performance. Using [`time2vec`](https://arxiv.org/abs/1907.05321) on time since takeoff $\tau_i$ seemed to harm performance.
<!-- we already use use_short_conv=True (see fla/layers/gated_deltanet.py) -->
<!-- we already do L2 normalisation on query and key (see fla/ops/gated_delta_rule/chunk.py::use_qk_l2norm_in_kernel) -->

### Outputs

To handle the positive-only, long-tailed distribution of the average fuel burn rate, we predict its log-transformed value. The target variable is:

$$
y = \log\left(\frac{\text{fuel\_kg}}{t_{\text{end}} - t_{\text{start}}} + 1\right)
$$

which is pooled from the last token of the GDN. This simplifies the problem to a sequence-to-scalar regression task, avoiding direct integration of the instantaneous burn rate $\dot{m}_f(t)$.

| AC Type | Notes                        | Target                        | RMSE (kg/s) | RMSE (kg) | A20N RMSE (kg/s) | A20N RMSE (kg) |
| ------- | ---------------------------- | ----------------------------- | ----------- | --------- | ---------------- | -------------- |
| a320n   | seq len in (1, 256]          | `avg_fuel_burn_rate`          |             |           | 0.296¹           |                |
| a320n   | -                            | ..                            |             |           | 0.205            | 103.98         |
| a320n   | -                            | log(`avg_fuel_burn_rate` + 1) |             |           | 0.198            | 98.30          |
| all     | seq len in (1, 256]          | ..                            | 0.465¹      | 254.34¹   | 0.293¹           |                |
| all     | -                            | ..                            | 0.316       | 167.59    | 0.197            | 95.73          |
| all     | remove feature `log(dt + 1)` | ..                            | 0.303       | 163.63    | 0.196            | 96.52          |

- using log of target improves performance slightly.
- ¹ these values refer to sequence length <= 256 and should not be compared directly with the other runs
- loss spikes are crazy, possibly due to `fuel_burn` quantisation?

<!-- training on the entire flight (with `is_in_segment` bool flags / adding `start` and `end` to the features worsened performnace) -->

## `v0.0.2` onwards

the rmse of each version should not be compared between each other.

- v0.0.2
  - ² A bug that caused segments with `seq_len = 2` to be excluded is fixed. this means the training set has much more (+30%) datapoints and so performance should NOT be compared with `v0.0.1`
  - the `NB` triton autotune parameter was removed for dramatic speedup
- v0.0.3
  - implemented constant-velocity kalman filter / rts smoother for {barometric altitude, inertial vertical rate, ground speed}:
    (0.3861 ± 0.0142) kg/s | (212.58 ± 16.32) kg
    faster convergence, rmse for short segments improved due to smoother estimates, but performance did not appreciably improve for longer segments (in fact, slightly worsened!)
  - adding $\dot{VS}$ or $\dot{GS}$ did not seem to improve RMSE.
- v0.0.4
  - ³ A bug that caused nondeterministic runs was fixed, and also includes segments with zero trajectory points.

| notes                                                   | rmse(kg/s)        | rmse(kg)         |
| ------------------------------------------------------- | ----------------- | ---------------- |
| v0.0.2                                                  | 0.3915 ± 0.0146²³ | 212.76 ± 16.52²³ |
| v0.0.3                                                  | 0.3859 ± 0.0136²³ | 217.29 ± 15.90²³ |
| v0.0.3 + $t_\text{end} - t_i$                           | 0.3779 ± 0.0141³  | 212.45 ± 16.93³  |
| v0.0.3 + $t_\text{end} - t_i$ + $\dot{VS}$ + $\dot{GS}$ | 0.3794 ± 0.0136³  | 212.19 ± 16.25³  |
| v0.0.4 (includes `seq_len` < 2)                         | 0.4182 ± 0.0138   | 218.87 ± 18.55   |
| v0.0.4 + rmse(kg/s)                                     | 0.3876 ± 0.0134   | 220.17 ± 17.46   |
| v0.0.4 + rmse(kg/s) + finetuned on rmse(kg)             | 0.4040 ± 0.0134   | 211.80 ± 18.68   |

## TODO

### data

- split gs, track -> vew, vns then separately filter that. do the same for lat/lng. $\phi = \arctan(\frac{V\dot{\phi}}{g})$
- google-arco era5: temperature, pressure and uv at flight level; isa deviation, $V_{g} \rightarrow V$, $\rho = \frac{p}{\rho R T}$.
- $\theta = \arcsin(\frac{\text{VS}}{V})$
- assuming point mass, quasisteady flight, coordinated turn / no sideslip
  - $\frac{C_L S}{m} = \frac{g\cos\gamma}{\frac{1}{2}\rho V^2 \cos\phi}$
  - $\frac{T - D}{m} = \underbrace{V \frac{dV}{dt} + g \frac{dh}{dt}}_{\text{Specific Energy Rate}} + \text{wind effect}$
- handle outliers robustly:
  - `prc770867379`, `prc770868424`: weird drops in altitude
    - in cases where this happens, ground speed is also similarity affected: data corrupted?
  - `prc770844923`: weird peaks in vertical_rate
  - between large time gaps where altitude is at cruising level and vertical rate is missing, heuristic: we should make it zero instead of interpolating
- <https://github.com/DGAC/Acropole> - maybe use their final layer outputs?

misc

- add fraction of time elapsed in segment, not $\Delta t$ since it harms performance
- parameterise the sequence length being predicted

### model

- model has to compress everything, switch to mean pooling? increased convergence speed but had no effect on validation rmse.
- bidirectional?
- use `is_outlier` bool flags (check missingness first)
- learning rate scheduling: warm up, hold at peak LR, then decay (cosine?)
- stack more layers with residual connections and rmsnorm between (though preliminary tests yield negligible improvements, maybe revisit when we train on the entire flight where the model has to compress information)
- `fuel_burn` is quantised (see [data](./data.md)): also predict absolute uncertainty, but is this really needed? (gaussian_nll_loss probably isn't relevant)
- staged training: train on short sequences first -> longer sequences
- improve input projection.
- gradient clipping

