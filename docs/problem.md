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

## Baseline (v0.0)

For this sequence-to-scalar regression task, we use a simple **single layer** [Gated DeltaNet](https://arxiv.org/pdf/2412.06464) (GDN) as a baseline.

Standard transformers are powerful but suffer from $O(L^2)$ complexity. GDN belongs to the family of state space models (SSM) that offers near-linear complexity for both training and inference while approaching the expressivity of transformers.

the recurrence is:
$$
S_t = \underbrace{\alpha_t (I - \beta_t k_t k_t^T)}_{\text{multiplicative/homogeneous part}} S_{t-1} + \underbrace{\beta_t v_t k_t^T}_{\text{additive/input part}}
$$

this is a discrete-time, time-varying bilinear system, with the memory matrix $S_t \in \mathbb{R}^{d_k \times d_v}$, vectorized as $\text{vec}(S_t)$.

1. the multiplicative part:

   - the term $(I - \beta_t k_t k_t^T)$ is a form of a Householder transformation. if $\|k_t\|_2 = 1$ and $\beta_t = 2$, it's a reflection (orthogonal, norm-preserving).
   - $\beta_t$ is the output of a sigmoid, so $0 < \beta_t < 1$. assuming $k_t$ is normalized (see below), this matrix is a contraction. its eigenvalues are $1$ (with multiplicity $d_k - 1$) and $1 - \beta_t$ (with multiplicity 1). all are $\leq 1$.
   - $\alpha_t$ is the decay gate, $\exp(g)$. $g$ is parameterized to be negative, so $0 < \alpha_t < 1$.
   - therefore, the multiplicative operator $\alpha_t (I - \beta_t k_t k_t^T)$ is a contraction. repeated application to $S_{t-1}$ will decay its norm.

2. the additive part:

   - the state $S_t$ accumulates outer products $\beta_t v_t k_t^T$ at each step.
   - unrolling the recurrence for a few steps:
       $$
       S_t \approx \sum_{i=1}^{t} \left( \prod_{j=i+1}^{t} M_j \right) N_i
       $$
       where $M_j$ is the multiplicative matrix and $N_i$ is the additive term.
   - even though each $M_j$ is a contraction, we are summing $t$ terms. if the norms $\|N_i\|_F = \|\beta_t v_t k_t^T\|_F = \beta_t \|v_t\|_2 \|k_t\|_2$ are consistently large, the sum $\|S_t\|_F$ can grow linearly or faster, leading to explosion.

we would like to log:

- the frobenius norm of the intermediate states
- mean values of $\alpha_t$ (`exp(g)`) and $\beta_t$

but this is difficult without modifying the gateddeltanet. TODO: vendor it.

### Inputs & Feature Engineering

For simplicity, we only use trajectory information within the prediction interval $[t_{\text{start}}, t_{\text{end}}]$.

### Preprocessing

To handle noisy and missing data, we apply a constant-velocity Kalman filter and RTS smoother to the time series of altitude, vertical rate, and ground speed. For ground speed and track, we first decompose them into East-West ($v_{ew}$) and North-South ($v_{ns}$) velocity components and apply the smoother to each component separately before recombining them. This provides smoother estimates, particularly helpful for shorter segments.

### Input Features & Hypernetwork

To better handle the long-tailed aircraft type distribution, we use parameter conditioning via a hypernetwork.

1. an aircraft type embedding, $e_{\text{ac}}$, is generated for each aircraft.
2. this embedding is passed through a small mlp (the hypernetwork) to generate the weights $W_{\text{proj}}$ and bias $b_{\text{proj}}$ of a unique input projection layer for that specific aircraft type.
3. the input vector at each time step, $o_i$, consisting of standardised observations and temporal features, is then projected into the model's hidden space:
    $$
    h_i = W_{\text{proj}} o_i + b_{\text{proj}}
    $$

Several other features were experimented with but did not improve performance, including time gaps between observations ($\Delta t_i$), `time2vec` embeddings, and physics-informed features like $\frac{T - D}{m} = \underbrace{\frac{dV}{dt} + g \frac{dh}{dt}}_{\text{Specific Energy Rate}} + \text{wind effect}$ and a $\frac{C_L S}{m} = \frac{g\cos\gamma}{\frac{1}{2}\rho V^2 \cos\phi}$. Noisy sequences are usually the culprit (e.g. $\dot{V}$, $\ddot{h}$)

### Outputs

To handle the positive-only, long-tailed distribution of the average fuel burn rate, we predict its log-transformed value. The target variable is:

$$
y = \log\left(\frac{\text{fuel\_kg}}{t_{\text{end}} - t_{\text{start}}} + 1\right)
$$

The sequence output from the gdn is pooled (e.g., using the last token). this pooled vector is then concatenated with the original aircraft type embedding, $e_{\text{ac}}$, before being passed to the final linear regression head. the concatentation is essential for aircraft with large fuel burn (a388).

### Results

The model is trained to minimize the RMSE of the total fuel burn in kilograms. The table below shows the performance of the final model on the validation set and the test set (rank partition).

<!-- assuming point mass, quasisteady flight, coordinated turn / no sideslip (also: $V = V_\text{gs}$, $\phi = 0$, no wind effect -->

| Model Configuration            | Validation RMSE (kg) | Test RMSE (kg) |
| ------------------------------ | -------------------- | -------------- |
| v0.0.5 (baseline)              | 216.28^              | 247.15 (v1)    |
| v0.0.5 + finetuned on RMSE(kg) | 209.92^              | 245.56 (v0)    |

Finetuning with a loss function that directly weights by segment duration (`rmse_kg`) provides a slight improvement in the final metric.

^ this incorrectly refers to seed 13 of train/validation split constructed from random sampling. more runs reveal RMSE of between 200-350.

<!--
v0.0.6: introduced warmup and gradient clipping to stabilise training.
v0.0.7: used stratified sampling (by ac type) and CB loss
v0.0.8: used stratified sampling (by ac type & by duration quantile)
v0.0.9: switched to hypernetworks, seed 24 RMSE: 228.2 -> 212.68
v0.0.9+dev1: concat aircraft type embedding to final layer, seed 24 RMSE: 212.68 -> 208.58
-->

### Class Imbalance

The aircraft type distribution follows a power law, causing the model to be biased towards common aircraft types (e.g., A320, A20N) while performing poorly on rare types.

We introduce an optional [Class-Balanced (CB) loss](https://arxiv.org/abs/1901.05555), weighing the loss based on the "effective number of samples", which posits that the marginal benefit of new data diminishes as sample size increases.

The effective number of samples $E_n$ for a class with $n$ samples is:
$$
E_n = \frac{1 - \beta^n}{1 - \beta}
$$
where $\beta \in [0, 1)$ is a hyperparameter. The CB loss weights the loss for a class $y$ by $\frac{1}{E_{n_y}}$:
$$
\mathcal{L}_{\text{CB}}(\mathbf{p}, y) = \frac{1 - \beta}{1 - \beta^{n_y}} \mathcal{L}(\mathbf{p}, y)
$$
This up-weights the loss for rare classes, encouraging the model to learn their features more effectively.

## TODO

### data

- handle outliers robustly:
  - `prc770867379`, `prc770868424`: weird drops in altitude
    - in cases where this happens, ground speed is also similarity affected: data corrupted?
  - `prc770844923`: weird peaks in vertical_rate
- do the same for lat/lng. $\phi = \arctan(\frac{V\dot{\phi}}{g})$
- google-arco era5: temperature, pressure and uv at flight level; isa deviation, $V_{g} \rightarrow V$, $\rho = \frac{p}{\rho R T}$.
- between large time gaps where altitude is at cruising level and vertical rate is missing, heuristic: we should make it zero instead of interpolating
- <https://github.com/DGAC/Acropole> - maybe use their final layer outputs?
- dataloader seed does not update on each iter!

misc

- add fraction of time elapsed in segment, not $\Delta t$ since it harms performance
- roll the time since takeoff
- parameterise the sequence length being predicted

### model

- model has to compress everything, switch to mean pooling? increased convergence speed but worsened validation rmse.
- bidirectional? no effect.
- use `is_outlier` bool flags (check missingness first)
- stack more layers (though preliminary tests yield negligible improvements, maybe revisit when we train on the entire flight where the model has to compress information)
- `fuel_burn` is quantised (see [data](./data.md)): also predict absolute uncertainty, but is this really needed? (gaussian_nll_loss probably isn't relevant)
- staged training: train on short sequences first -> longer sequences
- multilayer -> stochastic depth / layerscale
- hyper networks for fast weights and/or components of gdn.